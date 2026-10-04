"""Configuration.

One flat dataclass loaded from ``config.yaml``. Flat rather than deeply nested
because every field is then overridable from the command line with a single
``--set key=value``, which is how the evaluation comparisons are run.

Nothing here is required: ``Config()`` gives sensible defaults, so the package
imports and runs even with no YAML file present.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    corpus_path: str = "data/corpus/corpus.jsonl"
    index_dir: str = "data/index"
    golden_path: str = "data/golden/golden.jsonl"
    abstain_path: str = "config/abstention.json"

    # ---- chunking ---------------------------------------------------------
    # Chunk size is the highest-leverage knob in a RAG system: it decides what
    # unit of meaning retrieval operates on, and a bad boundary cannot be fixed
    # downstream. 900 chars (~225 tokens) keeps a whole abstract section intact.
    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_chars: int = 120
    contextual_headers: bool = True   # embed "Title > Section" with the body

    # ---- embeddings -------------------------------------------------------
    # `auto` tries the local model, then falls back to the offline lexical
    # encoder so the repo still runs with no ML dependencies installed.
    embed_backend: str = "auto"       # auto | sentence_transformers | openai | lexical
    embed_model: str = "all-MiniLM-L6-v2"
    embed_batch_size: int = 8
    embed_max_seq_length: int = 256
    openai_embed_model: str = "text-embedding-3-small"

    # ---- retrieval --------------------------------------------------------
    mode: str = "weighted"            # weighted | hybrid (RRF) | dense | sparse
    top_k: int = 6
    candidate_k: int = 30             # per-channel depth before fusion
    rrf_k: int = 60                   # Reciprocal Rank Fusion constant
    # Weight given to the dense channel in `weighted` fusion (sparse gets 1-x).
    # 0.0 == BM25 only, 1.0 == dense only. This is tuned from measurement, not
    # assumed: `python scripts/evaluate.py --sweep-weights` produces the curve.
    dense_weight: float = 0.2
    bm25_k1: float = 1.5              # BM25 term-frequency saturation
    bm25_b: float = 0.75              # BM25 length normalisation
    expand_acronyms: bool = True      # AD -> "alzheimer disease" at tokenise time
    # Cap on how many chunks one document may contribute to the result list.
    # Without it a single long review can occupy every slot, so the generator
    # sees one source repeated and document-level recall drops.
    max_chunks_per_doc: int = 2
    rerank: bool = False   # measured to hurt on this corpus; see config.yaml
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 30       # how many fused hits the reranker rescores

    # ---- generation -------------------------------------------------------
    llm: str = "auto"                 # auto | openai | gemini | ollama | extractive
    llm_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-1.5-flash"
    ollama_model: str = "llama3.1:8b"
    temperature: float = 0.0          # 0.0 so answers are reproducible/auditable
    max_tokens: int = 900
    timeout_s: int = 60
    max_context_chars: int = 2200     # per passage, to bound prompt length
    refusal_token: str = "INSUFFICIENT_CONTEXT"
    persona: str = (
        "You are NeuroRAG, a precise research assistant for Alzheimer's disease "
        "and neuroimaging literature. You answer strictly from the supplied "
        "context and cite every claim."
    )

    # ---- abstention -------------------------------------------------------
    # Refusing to answer is a first-class behaviour. A confident wrong answer in
    # a clinical literature assistant is worse than no answer, so the system
    # abstains when retrieval confidence is below a calibrated threshold.
    abstain: bool = True
    abstain_threshold: Optional[float] = None   # None -> load from abstain_path
    abstain_max_false_refusal: float = 0.15     # calibration budget

    # ---- misc -------------------------------------------------------------
    seed: int = 42

    # -- loading / overrides -------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "Config":
        path = Path(path) if path else DEFAULT_PATH
        if not path.exists():
            return cls()
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        allowed = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            # Warn rather than fail: a stale key in a config file should not
            # stop someone from running the system.
            from .utils import get_logger
            get_logger("neurorag.config").warning("Ignoring unknown config keys: %s", sorted(unknown))
        return cls(**{k: v for k, v in raw.items() if k in allowed})

    def merge(self, overrides: Dict[str, Any]) -> "Config":
        """Return a new Config with ``{"top_k": 10}``-style overrides applied."""
        data = self.to_dict()
        data.update({k: v for k, v in overrides.items() if k in data})
        return Config.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def snapshot(self) -> Dict[str, Any]:
        """Compact summary recorded with every result, so a number is always
        traceable to the settings that produced it."""
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "rerank": self.rerank,
            "chunk": f"{self.chunk_size}+{self.chunk_overlap}",
            "embed_model": self.embed_model if self.embed_backend != "lexical" else "lexical",
            "llm": self.llm,
            "abstain": self.abstain,
        }

    def warnings(self) -> list[str]:
        """Sanity checks that catch configs which would silently misbehave."""
        out: list[str] = []
        if self.chunk_overlap >= self.chunk_size:
            out.append("chunk_overlap >= chunk_size: chunks will never advance.")
        if self.min_chunk_chars >= self.chunk_size / 2:
            out.append(f"min_chunk_chars ({self.min_chunk_chars}) is close to chunk_size "
                       f"({self.chunk_size}): most chunks will be treated as fragments and "
                       "merged or discarded.")
        if self.rerank and self.rerank_candidates < self.top_k:
            out.append("rerank_candidates < top_k: reranking cannot fill the requested k.")
        if self.mode not in {"hybrid", "weighted", "dense", "sparse"}:
            out.append(f"unknown retrieval mode '{self.mode}'.")
        if self.temperature > 0:
            out.append("temperature > 0 makes generation non-reproducible run to run.")
        return out


def parse_overrides(pairs: list[str] | None) -> Dict[str, Any]:
    """Parse ``--set key=value`` CLI pairs, coercing to the right scalar type."""
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, raw = (s.strip() for s in pair.split("=", 1))
        value: Any = raw
        if raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        elif raw.lower() in {"none", "null"}:
            value = None
        else:
            for cast in (int, float):
                try:
                    value = cast(raw)
                    break
                except ValueError:
                    continue
        out[key] = value
    return out
