"""The RAG pipeline.

    corpus.jsonl ─► chunk ─► embed ─┬─► dense vectors ─┐
                                    └─► BM25 index  ───┴─► RRF fusion ─► rerank
                                                                            │
      question ─────────────────────────────────────────────────────────────┤
                                                                            ▼
                                                    abstention gate ─► grounded
                                                    (refuse if unsure)   prompt
                                                                            │
                                                                            ▼
                                                            LLM ─► citation
                                                                   resolution
                                                                            │
                                                                            ▼
                                                                        Answer

``build`` constructs everything from the corpus snapshot and persists it;
``load`` restores it, so the CLI, the evaluation script and the Streamlit app all
share one index.

Two guarantees worth stating:

* **The embedder that built an index is recorded in its metadata.** Loading that
  index with a different encoder would compare query vectors against stored
  vectors from a different space, making every score meaningless. ``load``
  refuses to do that unless explicitly overridden.
* **Every ``Answer`` carries its config snapshot and retrieval scores**, so any
  result can be traced back to the settings and passages that produced it.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from .chunking import chunk_documents, chunk_stats
from .config import Config
from .corpus import dedupe_documents, load_inbox, load_jsonl, load_text_dir
from .embeddings import LexicalEmbedder, build_embedder
from .generator import (
    AbstentionGate,
    Answer,
    audit,
    build_llm,
    confidence_signals,
    refusal_detected,
    refusal_reason,
    resolve_citations,
)
from .retriever import BM25Index, HybridRetriever, Reranker, Tokenizer, VectorStore, compile_filter
from .schema import Chunk, Document, Hit
from .utils import ensure_dir, get_logger, set_seed, timed, write_json

log = get_logger("neurorag.pipeline")

ROOT = Path(__file__).resolve().parent.parent

_FILLER = re.compile(
    r"^(hey|hi|hello|ok|okay|so|please|can you|could you|would you|i want to know|"
    r"i would like to know|tell me|explain to me)\b[\s,]*", re.IGNORECASE)


def clean_question(question: str) -> str:
    """Collapse whitespace and strip conversational filler.

    Filler contributes noise tokens to BM25 and shifts the dense embedding away
    from the actual information need, so it is removed before either channel sees
    the query.
    """
    text = re.sub(r"\s+", " ", question or "").strip()
    previous = None
    while previous != text:
        previous = text
        text = _FILLER.sub("", text).strip()
    return text


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return any(m in text for m in ("cannot allocate memory", "can't allocate memory",
                                   "out of memory", "defaultcpuallocator", "std::bad_alloc"))


class RAGPipeline:
    """Owns the corpus, the indexes, the retriever, the gate and the generator."""

    def __init__(self, config: Config, documents: Sequence[Document], chunks: Sequence[Chunk],
                 embedder, store: VectorStore, bm25: Optional[BM25Index],
                 reranker: Optional[Reranker], gate: AbstentionGate,
                 build_info: Optional[Dict[str, Any]] = None):
        self.config = config
        self.documents = list(documents)
        self.chunks = list(chunks)
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker
        self.gate = gate
        self.build_info: Dict[str, Any] = build_info or {}
        self.retriever = HybridRetriever(config, store, bm25, embedder, reranker)
        self._llm = None
        # The gate must match the encoder actually in use; see AbstentionGate.load.
        if gate is not None and gate.calibrated_for_embedder and embedder is not None \
                and gate.calibrated_for_embedder != embedder.name:
            log.warning("Disabling abstention gate: calibrated for '%s', index uses '%s'.",
                        gate.calibrated_for_embedder, embedder.name)
            self.gate = AbstentionGate(enabled=False)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, config: Optional[Config] = None, save: bool = True,
              extra_dirs: Optional[Sequence[str | Path]] = None) -> "RAGPipeline":
        """Chunk, embed and index the corpus."""
        config = config or Config.load()
        set_seed(config.seed)
        for warning in config.warnings():
            log.warning("Config: %s", warning)

        corpus_path = _resolve(config.corpus_path)
        index_dir = _resolve(config.index_dir)
        timings: Dict[str, float] = {}

        documents = dedupe_documents(load_jsonl(corpus_path))
        for extra in extra_dirs or []:
            documents = dedupe_documents(documents + load_text_dir(extra))
        documents = dedupe_documents(documents + load_inbox(ROOT / "data" / "corpus" / "inbox"))
        log.info("Corpus: %d documents from %s", len(documents), corpus_path)

        embedder, cache = build_embedder(config, cache_dir=ROOT / "data" / "cache")

        with timed("chunking", timings):
            chunks = chunk_documents(documents, config)
        stats = chunk_stats(chunks)
        log.info("Chunks: %s", stats)
        if not chunks:
            raise RuntimeError("Chunking produced no chunks; check corpus and chunk_size.")

        # The offline lexical encoder needs corpus statistics for IDF first.
        if isinstance(embedder, LexicalEmbedder):
            embedder.fit([c.embed_text for c in chunks])

        with timed("embedding", timings):
            vectors, embedder = _encode_with_fallback(embedder, [c.embed_text for c in chunks], config, cache)

        store = VectorStore()
        with timed("indexing", timings):
            store.build(chunks, vectors)

        bm25 = None
        with timed("bm25", timings):
            bm25 = BM25Index(config.bm25_k1, config.bm25_b, Tokenizer(config.expand_acronyms))
            bm25.build(chunks)

        reranker = None
        if config.rerank:
            reranker = Reranker(config.rerank_model, config.rerank_candidates)
            reranker.load()

        build_info = {
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "corpus_path": str(corpus_path),
            "n_documents": len(documents),
            "chunk_stats": stats,
            "embedder": embedder.name,
            "embedder_is_semantic": bool(getattr(embedder, "is_semantic", False)),
            "embedding_dim": store.dim,
            "reranker": reranker.name if reranker else None,
            "reranker_is_neural": bool(reranker.is_neural) if reranker else False,
            "timings_ms": {k: round(v, 1) for k, v in timings.items()},
            "config": config.snapshot(),
        }

        pipeline = cls(config, documents, chunks, embedder, store, bm25, reranker,
                       AbstentionGate.load(config, embedder.name), build_info)
        if save:
            pipeline.save(index_dir)
        return pipeline

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, index_dir: Optional[str | Path] = None) -> Dict[str, Any]:
        index_dir = _resolve(index_dir or self.config.index_dir)
        ensure_dir(index_dir)
        meta = dict(self.build_info)
        meta.update(self.store.save(index_dir))
        if self.bm25 is not None:
            meta["bm25"] = self.bm25.save(index_dir)
        write_json(index_dir / "index_meta.json", meta)
        self.build_info = meta
        log.info("Index saved -> %s", index_dir)
        return meta

    @classmethod
    def load(cls, config: Optional[Config] = None,
             allow_embedder_mismatch: bool = False) -> "RAGPipeline":
        """Restore a previously built index."""
        config = config or Config.load()
        index_dir = _resolve(config.index_dir)
        meta_path = index_dir / "index_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No index at {index_dir}. Run `python scripts/build_index.py` first.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        documents = load_jsonl(_resolve(config.corpus_path))

        # Adopt the embedder the index was actually built with. Query vectors must
        # live in the same space as the stored vectors, or every score is
        # meaningless - and making the user remember which flags they built with is
        # a needless failure mode. It is still an error if the environment cannot
        # provide that embedder at all.
        recorded = meta.get("embedder")
        if recorded and not allow_embedder_mismatch:
            if recorded == "lexical":
                if config.embed_backend != "lexical":
                    log.info("Index was built with the offline lexical encoder; adopting it.")
                    config = config.merge({"embed_backend": "lexical"})
            elif recorded.startswith("openai:"):
                if config.embed_backend != "openai":
                    log.info("Index was built with OpenAI embeddings; adopting them.")
                    config = config.merge({"embed_backend": "openai"})
            elif recorded != config.embed_model:
                log.info("Index was built with '%s'; adopting it instead of '%s'.",
                         recorded, config.embed_model)
                config = config.merge({"embed_backend": "sentence_transformers",
                                       "embed_model": recorded})

        embedder, cache = build_embedder(config, cache_dir=ROOT / "data" / "cache")

        if recorded and recorded != embedder.name and not allow_embedder_mismatch:
            raise RuntimeError(
                f"This index was built with embedder '{recorded}' but this environment "
                f"resolves to '{embedder.name}'. Query vectors would not be comparable to "
                "the stored vectors, so every score would be meaningless. Rebuild the "
                "index, or pass allow_embedder_mismatch=True if you are certain.")

        store = VectorStore.load(index_dir)
        if isinstance(embedder, LexicalEmbedder):
            embedder.fit([c.embed_text for c in store.chunks])

        bm25 = None
        if (index_dir / "bm25_meta.json").exists():
            bm25 = BM25Index.load(index_dir, store.chunks, config.bm25_k1, config.bm25_b,
                                  Tokenizer(config.expand_acronyms))

        reranker = None
        if config.rerank:
            reranker = Reranker(config.rerank_model, config.rerank_candidates)
            reranker.load()

        log.info("Index loaded: %d chunks (embedder=%s, reranker=%s)",
                 len(store.chunks), meta.get("embedder"),
                 getattr(reranker, "name", None))
        return cls(config, documents, store.chunks, embedder, store, bm25, reranker,
                   AbstentionGate.load(config, embedder.name), meta)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    @property
    def llm(self):
        if self._llm is None:
            self._llm = build_llm(self.config)
        return self._llm

    def retrieve(self, question: str, top_k: Optional[int] = None,
                 mode: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[Hit]:
        """Retrieval only - no generation, no abstention."""
        query = clean_question(question)
        allowed: Optional[Set[str]] = None
        if filters:
            allowed = compile_filter(self.chunks, **filters)
            if allowed is not None and not allowed:
                log.warning("Filter matched no chunks; returning nothing.")
                return []
        return self.retriever.retrieve(query, top_k=top_k, allowed=allowed, mode=mode)

    def ask(self, question: str, top_k: Optional[int] = None, mode: Optional[str] = None,
            filters: Optional[Dict[str, Any]] = None, extra: Optional[str] = None) -> Answer:
        """Full RAG pass: retrieve -> abstain? -> prompt -> generate -> cite."""
        timings: Dict[str, float] = {}
        with timed("retrieval", timings):
            hits = self.retrieve(question, top_k=top_k, mode=mode, filters=filters)

        signals = confidence_signals(hits)

        # ---- abstain before prompting ------------------------------------
        refuse, confidence, reason = self.gate.decide(hits)
        if refuse:
            log.info("Abstaining on %r: %s", question[:60], reason)
            return Answer(
                question=question, text=self.config.refusal_token, hits=hits,
                refused=True, refusal_reason=f"abstention gate: {reason}",
                backend="abstention_gate", confidence=round(confidence, 4),
                latency_ms={k: round(v, 1) for k, v in timings.items()},
                config={**self.config.snapshot(), "signals": signals})

        with timed("generation", timings):
            try:
                raw = self.llm.generate(question, hits)
            except Exception as exc:
                log.error("LLM backend failed (%s); refusing rather than guessing.", exc)
                raw = self.config.refusal_token

        if refusal_detected(raw, self.config.refusal_token):
            return Answer(
                question=question, text=raw or self.config.refusal_token, hits=hits,
                refused=True, refusal_reason=refusal_reason(raw) if raw else "no passages retrieved",
                backend=self.llm.name, confidence=round(confidence, 4),
                latency_ms={k: round(v, 1) for k, v in timings.items()},
                config={**self.config.snapshot(), "signals": signals})

        citations, fabricated = resolve_citations(raw, hits)
        answer = Answer(
            question=question, text=raw.strip(), citations=citations, hits=hits,
            refused=False, backend=self.llm.name, confidence=round(confidence, 4),
            latency_ms={k: round(v, 1) for k, v in timings.items()},
            config={**self.config.snapshot(), "signals": signals, "fabricated_markers": fabricated})
        answer.config["audit"] = audit(answer)
        return answer

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Everything needed to know how this index was built and what ran."""
        return {
            "n_documents": len(self.documents),
            "n_chunks": len(self.chunks),
            "retriever": self.retriever.describe(),
            "llm": f"{self.llm.name} ({'generative' if self.llm.is_generative else 'EXTRACTIVE-FALLBACK'})",
            "abstention": ("active, threshold=%.3f" % self.gate.threshold) if self.gate.active else "inactive",
            "vector_store": VectorStore.kind,
            "embedding_dim": self.store.dim,
            "bm25_terms": len(self.bm25.df) if self.bm25 else 0,
            "config": self.config.snapshot(),
            "built_at": self.build_info.get("built_at"),
            "chunk_stats": self.build_info.get("chunk_stats"),
            "timings_ms": self.build_info.get("timings_ms"),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _encode_with_fallback(embedder, texts: Sequence[str], config: Config, cache):
    """Encode the corpus, degrading batch size then model on memory pressure.

    Base-size encoders OOM *during* the forward pass, not at load time, so the
    failure only appears once encoding starts. Rather than crashing the build we
    step down: halve the batch size, then batch 1, then a smaller model. Whichever
    encoder actually succeeded is returned, because ``index_meta.json`` must
    record the truth - a benchmark attributed to the wrong encoder is worse than
    no benchmark.
    """
    batch = max(1, config.embed_batch_size)
    attempts = [(embedder, batch), (embedder, max(1, batch // 2)), (embedder, 1)]

    lighter = "all-MiniLM-L6-v2"
    if getattr(embedder, "name", "") != lighter and not isinstance(embedder, LexicalEmbedder):
        try:
            from .embeddings import SentenceTransformerEmbedder
            attempts.append((SentenceTransformerEmbedder(config, model_name=lighter), max(1, batch // 2)))
        except Exception as exc:
            log.warning("Could not prepare fallback encoder '%s': %s", lighter, exc)

    last: Optional[BaseException] = None
    for candidate, candidate_batch in attempts:
        try:
            log.info("Encoding %d chunks with %s (batch=%d) ...", len(texts), candidate.name, candidate_batch)
            saved, candidate.config.embed_batch_size = candidate.config.embed_batch_size, candidate_batch
            try:
                return candidate.encode_cached(list(texts), cache=cache), candidate
            finally:
                candidate.config.embed_batch_size = saved
        except (MemoryError, RuntimeError) as exc:
            if not _is_oom(exc):
                raise
            last = exc
            log.warning("Memory failure with %s (batch=%d): %s", candidate.name, candidate_batch, str(exc)[:140])

    log.warning("All neural encoders failed on memory; using the offline lexical encoder.")
    lexical = LexicalEmbedder(config)
    lexical.fit(texts)
    try:
        return lexical.encode_cached(list(texts), cache=cache), lexical
    except Exception as exc:
        raise RuntimeError(f"Could not embed the corpus with any backend: {last or exc}") from (last or exc)
