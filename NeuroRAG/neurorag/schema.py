"""Data structures that flow through the pipeline.

    Document  -> one paper (title, authors, abstract/full text, metadata)
    Chunk     -> one retrievable piece of text cut from a Document
    Hit       -> a Chunk plus the retrieval scores that ranked it
    Citation  -> a resolved, human-checkable reference for one Hit
    Answer    -> generated text + citations + provenance + audit flags

Provenance is carried on the objects themselves rather than in side tables, so an
``Answer`` can always be walked back to the exact passage it came from. That is
the point of a RAG system: the claim and its source travel together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """One bibliographic record plus the text we were able to obtain."""

    doc_id: str
    title: str
    text: str
    source: str = "unknown"        # europepmc | arxiv | crossref | pdf | local
    stratum: str = "unspecified"   # clinical | methods | rag
    topic: str = "unspecified"
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    sections: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class Chunk:
    """A retrievable unit of text, with enough metadata to filter and cite it."""

    chunk_id: str
    doc_id: str
    text: str
    index: int
    section: Optional[str] = None
    # `header` is prepended when *embedding* (so a chunk that reads "They report
    # 94.2% accuracy" is not detached from its paper) but is NOT part of `text`,
    # so what the user sees is always verbatim source text.
    header: Optional[str] = None
    doc_title: Optional[str] = None
    doc_year: Optional[int] = None
    doc_stratum: Optional[str] = None
    doc_url: Optional[str] = None
    doc_authors: Optional[List[str]] = None

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}" if self.header else self.text

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class Hit:
    """A ranked chunk with the score from each retrieval channel.

    Keeping every channel's score (not just the final one) is what makes the
    system debuggable: you can see whether a chunk won on lexical match, on
    semantic similarity, or on the reranker.
    """

    chunk: Chunk
    score: float
    rank: int
    scores: Dict[str, float] = field(default_factory=dict)
    matched_terms: List[str] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


@dataclass
class Citation:
    """One resolved citation marker from the answer text."""

    marker: int
    chunk_id: str
    doc_id: str
    title: str
    quote: str
    section: Optional[str] = None
    url: Optional[str] = None
    year: Optional[int] = None
    authors: Optional[List[str]] = None

    def as_reference(self) -> str:
        authors = ", ".join((self.authors or [])[:3])
        if len(self.authors or []) > 3:
            authors += " et al."
        return f"[{self.marker}] {authors or 'Unknown'} ({self.year or 'n.d.'}). {self.title}. {self.url or ''}".strip()


@dataclass
class Answer:
    """Final response plus the audit trail that justifies it."""

    question: str
    text: str
    citations: List[Citation] = field(default_factory=list)
    hits: List[Hit] = field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None
    backend: str = "unknown"
    latency_ms: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None      # abstention-gate score
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "text": self.text,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "backend": self.backend,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "config": self.config,
            "citations": [asdict(c) for c in self.citations],
            "retrieved": [
                {"rank": h.rank, "score": round(h.score, 4), "doc_id": h.doc_id,
                 "title": h.chunk.doc_title, "scores": {k: round(v, 4) for k, v in h.scores.items()}}
                for h in self.hits
            ],
        }
