"""Chunking.

Chunking decides what unit of meaning retrieval operates on. Too small and a
chunk loses the context that makes it interpretable ("They report 94.2%
accuracy" - who, on what?); too large and a retrieved chunk is mostly
irrelevant text that dilutes both the embedding and the prompt.

Two strategies:

``recursive`` (default)
    Packs greedily up to ``chunk_size`` along a delimiter hierarchy - paragraph,
    then sentence, then word - with ``chunk_overlap`` carried across boundaries so
    a claim is never cut in half. Model-agnostic and cheap.

``section``
    Emits one chunk per recovered abstract section (Background / Methods /
    Results / Conclusions), splitting further only when a section overflows.
    Better for structured biomedical abstracts because it keeps a Methods claim
    out of a Results chunk. Falls back to ``recursive`` when the loader could not
    recover any section structure.

Both attach a **contextual header** (``Title > Section``) that is embedded with
the body but excluded from ``Chunk.text``, so retrieval gets the context while
the user is always shown verbatim source text.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from .config import Config
from .schema import Chunk, Document
from .utils import get_logger, normalize_whitespace, split_sentences, truncate

log = get_logger("neurorag.chunking")

_PARAGRAPH = re.compile(r"\n\s*\n")


def _header(doc: Document, section: Optional[str], enabled: bool) -> Optional[str]:
    """Context string embedded with the chunk but not shown to the user."""
    if not enabled:
        return None
    head = truncate(doc.title, 140)
    if section and section.lower() not in {"abstract", "preamble", "full text"}:
        head = f"{head} > {section}"
    return head


def recursive_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Greedily pack sentences into windows of ``chunk_size`` with ``overlap``."""
    text = normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    overlap = max(0, min(overlap, chunk_size - 1))
    units: List[str] = []
    for para in _PARAGRAPH.split(text):
        para = para.strip()
        if not para:
            continue
        units.extend(split_sentences(para) or [para])

    chunks: List[str] = []
    current: List[str] = []
    length = 0

    def flush() -> None:
        nonlocal current, length
        if current:
            chunks.append(" ".join(current).strip())
            current, length = [], 0

    def tail_window(items: List[str]) -> List[str]:
        """The trailing items that fit inside the overlap window."""
        keep, kept = [], 0
        for item in reversed(items):
            if kept + len(item) + 1 > overlap:
                break
            keep.insert(0, item)
            kept += len(item) + 1
        return keep

    for unit in units:
        # A single unit longer than the window is hard-split on words.
        if len(unit) + 1 > chunk_size:
            flush()
            buf, buf_len = [], 0
            for word in unit.split():
                if buf_len + len(word) + 1 > chunk_size and buf:
                    chunks.append(" ".join(buf).strip())
                    buf, buf_len = [], 0
                    for w in tail_window(chunks[-1].split()):
                        buf.append(w)
                        buf_len += len(w) + 1
                buf.append(word)
                buf_len += len(word) + 1
            if buf:
                chunks.append(" ".join(buf).strip())
            continue

        if length + len(unit) + 1 > chunk_size and current:
            carry = tail_window(current)
            flush()
            current, length = carry, sum(len(c) + 1 for c in carry)

        current.append(unit)
        length += len(unit) + 1

    flush()
    return [c for c in chunks if c]


def chunk_document(doc: Document, config: Config, strategy: str = "recursive") -> List[Chunk]:
    """Chunk one document, section by section, stamping citation metadata."""
    pieces: List[Tuple[str, str]] = []
    if doc.sections and (strategy == "section" or len(doc.sections) > 1):
        pieces = [(name, body) for name, body in doc.sections.items()]
    elif doc.text.strip():
        # Papers fetched from Europe PMC / arXiv are abstracts; anything else
        # (a local note, a user PDF) is treated as unstructured full text.
        label = "Abstract" if doc.source in {"europepmc", "arxiv", "crossref"} else "Full text"
        pieces = [(label, doc.text)]

    chunks: List[Chunk] = []
    for section, body in pieces:
        body = normalize_whitespace(body)
        if not body:
            continue
        if strategy == "section" and len(body) <= config.chunk_size:
            bodies = [body]
        else:
            bodies = recursive_split(body, config.chunk_size, config.chunk_overlap)
        for text in bodies:
            # A chunk below min_chunk_chars is a fragment. Dropping it would
            # silently lose source text, so fold it back into the previous chunk
            # when there is room and only discard it if that is impossible.
            if len(text) < config.min_chunk_chars and chunks:
                previous = chunks[-1]
                if previous.doc_id == doc.doc_id and \
                        len(previous.text) + len(text) + 1 <= config.chunk_size * 1.25:
                    previous.text = f"{previous.text} {text}".strip()
                    continue
                if len(text) < config.min_chunk_chars // 2:
                    continue
            idx = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#c{idx}",
                doc_id=doc.doc_id,
                text=text,
                index=idx,
                section=section,
                header=_header(doc, section, config.contextual_headers),
                doc_title=doc.title,
                doc_year=doc.year,
                doc_stratum=doc.stratum,
                doc_url=doc.url,
                doc_authors=doc.authors,
            ))
            if len(chunks) >= 200:
                log.warning("Document %s hit the 200-chunk cap; truncating", doc.doc_id)
                return chunks

    if not chunks and doc.text.strip():
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#c0", doc_id=doc.doc_id,
            text=truncate(normalize_whitespace(doc.text), config.chunk_size * 2),
            index=0, section="Abstract", header=_header(doc, None, config.contextual_headers),
            doc_title=doc.title, doc_year=doc.year, doc_stratum=doc.stratum,
            doc_url=doc.url, doc_authors=doc.authors,
        ))
    return chunks


def chunk_documents(docs: Iterable[Document], config: Config, strategy: str = "recursive") -> List[Chunk]:
    docs = list(docs)
    chunks: List[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, config, strategy))
    log.info("Chunked %d documents into %d chunks (strategy=%s, size=%d+%d)",
             len(docs), len(chunks), strategy, config.chunk_size, config.chunk_overlap)
    return chunks


def chunk_stats(chunks: List[Chunk]) -> dict:
    """Summary reported after every index build."""
    if not chunks:
        return {"n_chunks": 0}
    lengths = sorted(len(c.text) for c in chunks)
    n = len(lengths)
    return {
        "n_chunks": n,
        "n_documents": len({c.doc_id for c in chunks}),
        "mean_chars": round(sum(lengths) / n, 1),
        "median_chars": lengths[n // 2],
        "min_chars": lengths[0],
        "max_chars": lengths[-1],
        "n_sections": len({c.section for c in chunks if c.section}),
    }
