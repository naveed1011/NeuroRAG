"""Retrieval: BM25 + dense vectors, fused by Reciprocal Rank Fusion, then reranked.

Why hybrid at all
-----------------
A bi-encoder compresses a passage into one vector. That is excellent for
paraphrase and terrible for **rare exact identifiers** - dataset names
("OASIS-1", "ADNI"), architectures ("Swin UNETR"), clinical acronyms ("CDR",
"T1gd"), metrics ("nDCG@10"). Those tokens carry most of the information in a
biomedical query and are precisely what averaging into a single vector washes
out. BM25 matches them exactly but cannot bridge synonymy. Fusing the two covers
both failure modes.

Why RRF rather than score weighting
-----------------------------------
BM25 scores are unbounded and corpus-dependent; cosine similarities live in
[-1, 1]. Combining them linearly requires per-query normalisation that is fragile
in practice. **Reciprocal Rank Fusion** (Cormack, Clarke & Buettcher, SIGIR 2009)

    score(d) = Σ_channels  1 / (k + rank_channel(d))

fuses *ranks* instead of scores, discarding magnitudes entirely. It needs no
tuning beyond ``k`` (60 is the published default) and is robust across
heterogeneous channels. ``mode: dense`` and ``mode: sparse`` are kept so the
evaluation script can measure what the hybrid actually buys.

Two-stage retrieve-then-rerank
------------------------------
The cross-encoder feeds ``[CLS] query [SEP] passage [SEP]`` through a transformer
jointly, so it models fine-grained term interaction that a single dot product
cannot. It is far more accurate and far too slow to run over the whole corpus, so
it only rescores the ~30 fused candidates. Accuracy of the expensive stage,
recall of the cheap stage.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import Config
from .embeddings import BaseEmbedder, tokenize
from .schema import Chunk, Hit
from .utils import ensure_dir, get_logger, read_jsonl, stable_hash, write_json, write_jsonl

log = get_logger("neurorag.retriever")


# ---------------------------------------------------------------------------
# Domain vocabulary
# ---------------------------------------------------------------------------
# Bidirectional acronym map. Expansion is applied at BOTH index and query time so
# that a paper written with "AD" and a user asking about "Alzheimer's disease"
# meet in the same inverted list.
ACRONYMS: Dict[str, Tuple[str, ...]] = {
    # clinical
    "ad": ("alzheimer", "disease"),
    "mci": ("mild", "cognitive", "impairment"),
    "cdr": ("clinical", "dementia", "rating"),
    "mmse": ("mini", "mental", "state", "examination"),
    "ftd": ("frontotemporal", "dementia"),
    "dlb": ("dementia", "lewy", "body"),
    "abeta": ("amyloid", "beta"),
    "apoe": ("apolipoprotein", "e"),
    # imaging
    "mri": ("magnetic", "resonance", "imaging"),
    "smri": ("structural", "magnetic", "resonance", "imaging"),
    "fmri": ("functional", "magnetic", "resonance", "imaging"),
    "pet": ("positron", "emission", "tomography"),
    "ct": ("computed", "tomography"),
    "t1w": ("t1", "weighted"),
    "t2w": ("t2", "weighted"),
    "t1gd": ("t1", "gadolinium", "enhanced"),
    "flair": ("fluid", "attenuated", "inversion", "recovery"),
    "dti": ("diffusion", "tensor", "imaging"),
    "vbm": ("voxel", "based", "morphometry"),
    "roi": ("region", "interest"),
    # datasets
    "adni": ("alzheimer", "disease", "neuroimaging", "initiative"),
    "oasis": ("open", "access", "series", "imaging", "studies"),
    "brats": ("brain", "tumor", "segmentation"),
    # methods
    "cnn": ("convolutional", "neural", "network"),
    "vit": ("vision", "transformer"),
    "gan": ("generative", "adversarial", "network"),
    "xai": ("explainable", "artificial", "intelligence"),
    "gradcam": ("gradient", "class", "activation", "mapping"),
    "auc": ("area", "under", "curve"),
    "roc": ("receiver", "operating", "characteristic"),
    "cv": ("cross", "validation"),
    "dl": ("deep", "learning"),
    # rag
    "rag": ("retrieval", "augmented", "generation"),
    "llm": ("large", "language", "model"),
    "llms": ("large", "language", "models"),
    "dpr": ("dense", "passage", "retrieval"),
    "rrf": ("reciprocal", "rank", "fusion"),
    "qa": ("question", "answering"),
}

# Kept short on purpose. In biomedical text words like "cell", "score" and
# "model" carry real signal, so an aggressive general-English stopword list
# actively hurts recall.
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "were", "was", "are",
    "been", "have", "has", "had", "their", "these", "those", "which", "while",
    "into", "than", "then", "there", "they", "each", "both", "such", "also",
    "over", "under", "using", "used", "our", "its", "can", "may", "but", "not",
    "all", "any", "what", "why", "how", "does", "is", "of", "in", "on", "to",
    "a", "an", "be", "it", "as", "at", "by", "we", "you", "your",
}


class Tokenizer:
    """Tokenizer with acronym expansion, compound splitting and light stopwords."""

    def __init__(self, expand: bool = True):
        self.expand_acronyms = expand

    def __call__(self, text: str) -> List[str]:
        base = tokenize(text)
        enriched: List[str] = []
        for tok in base:
            enriched.append(tok)
            if "-" in tok or "_" in tok:
                enriched.extend(p for p in re.split(r"[-_]", tok) if p)
        if self.expand_acronyms:
            for tok in list(enriched):
                enriched.extend(ACRONYMS.get(tok, ()))
        return [t for t in enriched if t not in STOPWORDS and len(t) >= 2]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class BM25Index:
    """Okapi BM25 with ``k1`` (term-frequency saturation) and ``b`` (length
    normalisation). Only documents sharing a query token are ever scored, via the
    inverted index, which is what keeps the search sub-linear."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer: Optional[Tokenizer] = None):
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer or Tokenizer()
        self.chunks: List[Chunk] = []
        self.freqs: List[Counter] = []
        self.lengths: List[int] = []
        self.df: Counter = Counter()
        self.inverted: Dict[str, List[int]] = {}
        self.avgdl = 0.0
        self.n_docs = 0

    def build(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.freqs, self.lengths = [], []
        self.df, self.inverted = Counter(), {}
        for i, chunk in enumerate(self.chunks):
            counts = Counter(self.tokenizer(chunk.embed_text))
            self.freqs.append(counts)
            self.lengths.append(sum(counts.values()))
            for token in counts:
                self.df[token] += 1
                self.inverted.setdefault(token, []).append(i)
        self.n_docs = len(self.chunks)
        self.avgdl = (sum(self.lengths) / self.n_docs) if self.n_docs else 0.0
        log.info("BM25 built: %d chunks, %d terms, avgdl=%.1f", self.n_docs, len(self.df), self.avgdl)

    def idf(self, token: str) -> float:
        """Robertson-Sparck Jones IDF, floored at 0 so an unseen term cannot
        contribute a negative score."""
        df = self.df.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10,
               allowed: Optional[Set[str]] = None) -> List[Tuple[Chunk, float, List[str]]]:
        """Return ``(chunk, score, matched_terms)`` ranked by BM25 score."""
        if not self.n_docs:
            return []
        counts = Counter(self.tokenizer(query))
        if not counts:
            return []

        candidates: Set[int] = set()
        for token in counts:
            candidates.update(self.inverted.get(token, ()))
        if allowed is not None:
            candidates = {i for i in candidates if self.chunks[i].chunk_id in allowed}
        if not candidates:
            return []

        norm_base = self.k1 * (1.0 - self.b)
        idf_cache: Dict[str, float] = {}
        scored: List[Tuple[float, int, List[str]]] = []
        for idx in candidates:
            freqs = self.freqs[idx]
            length_norm = norm_base + self.k1 * self.b * ((self.lengths[idx] or 1) / (self.avgdl or 1.0))
            score, matched = 0.0, []
            for token, qtf in counts.items():
                tf = freqs.get(token, 0)
                if not tf:
                    continue
                if token not in idf_cache:
                    idf_cache[token] = self.idf(token)
                if idf_cache[token] <= 0:
                    continue
                score += idf_cache[token] * (tf * (self.k1 + 1.0)) / (tf + length_norm)
                matched.append(token)
            if score > 0:
                scored.append((score, idx, matched))

        scored.sort(key=lambda t: (-t[0], t[1]))
        out = []
        for score, idx, matched in scored[:top_k]:
            # Report the most informative matches (highest IDF) rather than an
            # arbitrary slice - that is what makes the explanation useful.
            best = sorted(set(matched), key=lambda t: -self.idf(t))[:8]
            out.append((self.chunks[idx], float(score), best))
        return out

    # -- persistence --
    def save(self, directory: str | Path) -> Dict[str, object]:
        directory = ensure_dir(directory)
        write_jsonl(directory / "bm25_tokens.jsonl",
                    [{"chunk_id": c.chunk_id,
                      "tokens": list(self.tokenizer(c.embed_text))} for c in self.chunks])
        meta = {"n_docs": self.n_docs, "avgdl": self.avgdl, "n_terms": len(self.df),
                "k1": self.k1, "b": self.b,
                "fingerprint": stable_hash(self.k1, self.b, self.n_docs, len(self.df))}
        write_json(directory / "bm25_meta.json", meta)
        return meta

    @classmethod
    def load(cls, directory: str | Path, chunks: Sequence[Chunk],
             k1: float = 1.5, b: float = 0.75, tokenizer: Optional[Tokenizer] = None) -> "BM25Index":
        directory = Path(directory)
        index = cls(k1, b, tokenizer)
        index.chunks = list(chunks)
        token_path = directory / "bm25_tokens.jsonl"
        if token_path.exists():
            by_id = {r["chunk_id"]: r["tokens"] for r in read_jsonl(token_path)}
            token_lists = [by_id.get(c.chunk_id) or index.tokenizer(c.embed_text) for c in index.chunks]
        else:
            log.warning("No persisted BM25 tokens; re-tokenising.")
            token_lists = [index.tokenizer(c.embed_text) for c in index.chunks]
        index.freqs = [Counter(t) for t in token_lists]
        index.lengths = [len(t) for t in token_lists]
        index.df, index.inverted = Counter(), {}
        for i, counts in enumerate(index.freqs):
            for token in counts:
                index.df[token] += 1
                index.inverted.setdefault(token, []).append(i)
        index.n_docs = len(index.chunks)
        index.avgdl = (sum(index.lengths) / index.n_docs) if index.n_docs else 0.0
        log.info("BM25 loaded: %d chunks, %d terms", index.n_docs, len(index.df))
        return index


# ---------------------------------------------------------------------------
# Dense vector store
# ---------------------------------------------------------------------------


class VectorStore:
    """Exact cosine search over L2-normalised vectors.

    Deliberately brute force: at ~1.3k chunks the whole search is one matrix
    multiply that finishes in well under a millisecond, so an ANN index would add
    approximation error and a dependency for no measurable gain. Swap in FAISS if
    the corpus grows past roughly a million chunks.
    """

    kind = "numpy-exact"

    def __init__(self):
        self.chunks: List[Chunk] = []
        self._matrix = None

    def build(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        import numpy as np
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks/vectors mismatch: {len(chunks)} vs {len(vectors)}")
        self.chunks = list(chunks)
        self._matrix = np.asarray(vectors, dtype="float32")
        if self._matrix.ndim == 1:
            self._matrix = self._matrix.reshape(1, -1)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def dim(self) -> Optional[int]:
        return int(self._matrix.shape[1]) if self._matrix is not None else None

    def search(self, query_vector: Sequence[float], top_k: int = 10,
               allowed: Optional[Set[str]] = None) -> List[Tuple[Chunk, float]]:
        import numpy as np
        if not self.chunks:
            return []
        scores = (self._matrix @ np.asarray(query_vector, dtype="float32").reshape(-1, 1)).ravel()
        if allowed is not None:
            mask = np.fromiter((c.chunk_id in allowed for c in self.chunks),
                               dtype=bool, count=len(self.chunks))
            if not mask.any():
                return []
            scores = np.where(mask, scores, -np.inf)
        k = int(min(top_k, len(scores)))
        part = np.argpartition(-scores, k - 1)[:k]
        part = part[np.argsort(-scores[part], kind="stable")]
        return [(self.chunks[i], float(scores[i])) for i in part if np.isfinite(scores[i])]

    def vectors_for(self, chunk_ids: Sequence[str]) -> Dict[str, List[float]]:
        index = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        return {cid: self._matrix[index[cid]].tolist() for cid in chunk_ids if cid in index}

    def save(self, directory: str | Path) -> Dict[str, object]:
        import numpy as np
        directory = ensure_dir(directory)
        np.save(directory / "vectors.npy", self._matrix)
        write_jsonl(directory / "chunks.jsonl", [c.to_dict() for c in self.chunks])
        meta = {"kind": self.kind, "dim": self.dim, "n_chunks": len(self.chunks),
                "n_documents": len({c.doc_id for c in self.chunks})}
        write_json(directory / "vector_store.json", meta)
        log.info("Vector store saved: %d chunks x %d dims", len(self.chunks), self.dim)
        return meta

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        import numpy as np
        directory = Path(directory)
        vec, chunks = directory / "vectors.npy", directory / "chunks.jsonl"
        if not vec.exists() or not chunks.exists():
            raise FileNotFoundError(
                f"No index in {directory}. Run `python scripts/build_index.py` first.")
        store = cls()
        store._matrix = np.load(vec)
        store.chunks = [Chunk.from_dict(r) for r in read_jsonl(chunks)]
        if len(store.chunks) != store._matrix.shape[0]:
            raise ValueError("Corrupt index: chunk count does not match vector count.")
        log.info("Vector store loaded: %d chunks x %d dims", len(store.chunks), store.dim)
        return store


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Tuple[Chunk, float]]], k: int = 60
) -> List[Tuple[Chunk, float, Dict[str, float]]]:
    """Fuse ranked lists by reciprocal rank.

    Returns ``(chunk, fused_score, per_channel_scores)``. Chunks are keyed by id,
    so a passage found by both channels is merged and scores its RRF contribution
    twice - which is exactly the desired consensus boost.
    """
    names = ("dense", "sparse", "extra")
    fused: Dict[str, float] = {}
    by_id: Dict[str, Chunk] = {}
    detail: Dict[str, Dict[str, float]] = {}

    for ci, ranked in enumerate(ranked_lists):
        name = names[ci] if ci < len(names) else f"ch{ci}"
        for rank, (chunk, score) in enumerate(ranked, start=1):
            cid = chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(cid, chunk)
            detail.setdefault(cid, {})[name] = float(score)
            detail[cid][f"{name}_rank"] = float(rank)

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    for cid, score in ordered:
        detail[cid]["rrf"] = float(score)
    return [(by_id[cid], score, detail[cid]) for cid, score in ordered]


def weighted_fusion(
    dense_hits: Sequence[Tuple[Chunk, float]],
    sparse_hits: Sequence[Tuple[Chunk, float]],
    dense_weight: float = 0.6,
) -> List[Tuple[Chunk, float, Dict[str, float]]]:
    """Min-max normalised score fusion.

    Included because it is the obvious alternative to RRF and worth measuring
    against it: normalising per query is exactly the fragile step RRF avoids.

    The detail dict keeps **both** the raw channel score (``dense``, ``sparse``)
    and the per-query normalised one (``dense_norm``, ``sparse_norm``). That
    distinction matters downstream: the abstention gate reads ``dense`` as a
    cosine similarity against a calibrated threshold, so handing it a min-max
    value would silently change the meaning of the signal and destroy its
    discriminative power.
    """
    def norm(pairs):
        if not pairs:
            return {}
        vals = [s for _, s in pairs]
        lo, hi, span = min(vals), max(vals), max(vals) - min(vals)
        if span <= 1e-12:
            return {c.chunk_id: 1.0 for c, _ in pairs}
        return {c.chunk_id: (s - lo) / span for c, s in pairs}

    raw_dense = {c.chunk_id: float(s) for c, s in dense_hits}
    raw_sparse = {c.chunk_id: float(s) for c, s in sparse_hits}
    dense_scores, sparse_scores = norm(dense_hits), norm(sparse_hits)
    by_id: Dict[str, Chunk] = {}
    for chunk, _ in list(dense_hits) + list(sparse_hits):
        by_id.setdefault(chunk.chunk_id, chunk)
    sparse_weight = 1.0 - dense_weight
    fused = {
        cid: dense_weight * dense_scores.get(cid, 0.0) + sparse_weight * sparse_scores.get(cid, 0.0)
        for cid in set(dense_scores) | set(sparse_scores)
    }
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    out = []
    for cid, score in ordered:
        out.append((by_id[cid], score, {
            "weighted": float(score),
            "dense": raw_dense.get(cid, 0.0),
            "sparse": raw_sparse.get(cid, 0.0),
            "dense_norm": dense_scores.get(cid, 0.0),
            "sparse_norm": sparse_scores.get(cid, 0.0),
        }))
    return out


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """Cross-encoder reranker with a pass-through fallback.

    If ``sentence_transformers`` is unavailable the reranker preserves the
    first-stage order and says so (``is_neural = False``), rather than silently
    producing a benchmark attributed to a neural stage that never ran.
    """

    def __init__(self, model_name: str, candidate_k: int = 30):
        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        self.candidate_k = candidate_k
        self.is_neural = False
        self._model = None

    def load(self) -> bool:
        try:
            from sentence_transformers import CrossEncoder
            log.info("Loading cross-encoder '%s' ...", self.model_name)
            self._model = CrossEncoder(self.model_name)
            self.is_neural = True
            log.info("Loaded cross-encoder '%s'", self.model_name)
            return True
        except Exception as exc:
            log.warning("Could not load cross-encoder '%s': %s", self.model_name, str(exc)[:160])
            log.warning("Reranking will preserve first-stage order (reranker=noop).")
            self.is_neural = False
            return False

    def rerank(self, query: str, chunks: Sequence[Chunk], top_n: int) -> List[Tuple[Chunk, float]]:
        if not chunks:
            return []
        pool = list(chunks)[: self.candidate_k]
        if not self.is_neural or self._model is None:
            n = max(len(pool), 1)
            return [(c, 1.0 - i / n) for i, c in enumerate(pool[:top_n])]
        scores = self._model.predict([[query, c.text] for c in pool],
                                     batch_size=4, show_progress_bar=False,
                                     convert_to_numpy=True)
        ranked = sorted(zip(pool, [float(s) for s in scores]), key=lambda t: -t[1])
        return ranked[:top_n]


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------


def cap_per_document(
    fused: Sequence[Tuple[Chunk, float, Dict[str, float]]], max_per_doc: int
) -> List[Tuple[Chunk, float, Dict[str, float]]]:
    """Limit how many chunks any single document may contribute.

    Without this, one long review paper can occupy every slot: the generator then
    sees a single source repeated, answers lose cross-source support, and
    document-level recall drops because distinct documents never reach the top-k.
    Order is preserved, so this only ever substitutes a later passage from a new
    document for a redundant one.
    """
    if max_per_doc <= 0:
        return list(fused)
    counts: Dict[str, int] = {}
    out = []
    for chunk, score, meta in fused:
        if counts.get(chunk.doc_id, 0) >= max_per_doc:
            continue
        counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        out.append((chunk, score, meta))
    return out


def compile_filter(chunks: Sequence[Chunk], strata: Optional[List[str]] = None,
                   year_min: Optional[int] = None, year_max: Optional[int] = None,
                   sections: Optional[List[str]] = None,
                   contains: Optional[List[str]] = None) -> Optional[Set[str]]:
    """Build the allowed chunk-id set, or ``None`` when no filter is active.

    Filtering happens *before* scoring, so both channels stay consistent. It is a
    precision lever, not a UI convenience: asking "what did 2024+ papers report?"
    over an unfiltered index returns whatever is semantically nearest, which is
    often a 2015 methods paper that happens to share vocabulary.
    """
    if not any([strata, year_min, year_max, sections, contains]):
        return None
    allowed: Set[str] = set()
    for chunk in chunks:
        if strata and (chunk.doc_stratum or "") not in strata:
            continue
        if sections and (chunk.section or "") not in sections:
            continue
        if year_min is not None and (chunk.doc_year is None or chunk.doc_year < year_min):
            continue
        if year_max is not None and (chunk.doc_year is None or chunk.doc_year > year_max):
            continue
        if contains:
            haystack = f"{chunk.header or ''} {chunk.text}".lower()
            if not all(term.lower() in haystack for term in contains):
                continue
        allowed.add(chunk.chunk_id)
    log.info("Filter -> %d/%d chunks allowed", len(allowed), len(chunks))
    return allowed


# ---------------------------------------------------------------------------
# The retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Dense + sparse retrieval, RRF fusion, optional cross-encoder reranking."""

    def __init__(self, config: Config, store: VectorStore, bm25: Optional[BM25Index],
                 embedder: BaseEmbedder, reranker: Optional[Reranker] = None):
        self.config = config
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 allowed: Optional[Set[str]] = None,
                 mode: Optional[str] = None) -> List[Hit]:
        top_k = top_k or self.config.top_k
        mode = (mode or self.config.mode or "hybrid").lower()
        depth = max(self.config.candidate_k, top_k * 3)

        dense_hits: List[Tuple[Chunk, float]] = []
        sparse_hits: List[Tuple[Chunk, float, List[str]]] = []
        terms: Dict[str, List[str]] = {}

        if mode in {"hybrid", "weighted", "dense"}:
            qv = self.embedder.encode([query])[0]
            dense_hits = self.store.search(qv, top_k=depth, allowed=allowed)
        if mode in {"hybrid", "weighted", "sparse"} and self.bm25 is not None:
            sparse_hits = self.bm25.search(query, top_k=depth, allowed=allowed)
            terms = {c.chunk_id: t for c, _s, t in sparse_hits}

        if mode == "dense":
            fused = [(c, s, {"dense": s}) for c, s in dense_hits]
        elif mode == "sparse":
            fused = [(c, s, {"sparse": s}) for c, s, _t in sparse_hits]
        else:
            sparse_pairs = [(c, s) for c, s, _t in sparse_hits]
            if mode == "weighted":
                fused = weighted_fusion(dense_hits, sparse_pairs,
                                        dense_weight=self.config.dense_weight)
            else:  # hybrid -> Reciprocal Rank Fusion
                fused = reciprocal_rank_fusion([dense_hits, sparse_pairs], k=self.config.rrf_k)

        if not fused:
            return []

        # ---- diversity cap --------------------------------------------------
        if self.config.max_chunks_per_doc > 0:
            before = len(fused)
            fused = cap_per_document(fused, self.config.max_chunks_per_doc)
            if len(fused) < before:
                log.debug("Per-document cap removed %d redundant passages", before - len(fused))

        # ---- rerank --------------------------------------------------------
        if self.config.rerank and self.reranker is not None:
            pool = fused[: max(self.config.rerank_candidates, top_k)]
            # Keep at least top_k items: the reranker's budget is a *generation*
            # budget, not a measurement budget. Truncating to fewer than top_k
            # would silently cap recall@k for every reranked configuration.
            lookup = {c.chunk_id: (s, m) for c, s, m in fused}
            reranked = self.reranker.rerank(query, [c for c, _s, _m in pool], top_n=max(top_k, self.config.top_k))
            merged = []
            for chunk, rerank_score in reranked:
                _orig, meta = lookup.get(chunk.chunk_id, (0.0, {}))
                merged.append((chunk, float(rerank_score), {**meta, "rerank": float(rerank_score)}))
            fused = merged

        hits = []
        for rank, (chunk, score, meta) in enumerate(fused[:top_k], start=1):
            hits.append(Hit(chunk=chunk, score=float(score), rank=rank,
                            scores={k: float(v) for k, v in meta.items()},
                            matched_terms=terms.get(chunk.chunk_id, [])))
        return hits

    def describe(self) -> str:
        """One-line backend summary, so no result is ever ambiguous about how it
        was produced."""
        semantic = "semantic" if getattr(self.embedder, "is_semantic", False) else "LEXICAL-FALLBACK"
        rr = "off"
        if self.reranker is not None:
            rr = "neural" if self.reranker.is_neural else "noop"
        return (f"mode={self.config.mode} | embedder={self.embedder.name} ({semantic}) | "
                f"bm25={'on' if self.bm25 else 'off'} | reranker={rr} | top_k={self.config.top_k}")
