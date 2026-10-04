"""Corpus loading.

Sources (all keyless and free):
  * **Europe PMC** - biomedical literature, abstract text included
  * **arXiv**      - methods / ML literature
  * **Crossref**   - bibliographic metadata by DOI, for paywalled papers whose
                     abstract is not openly deposited
  * **offline**    - the committed JSONL snapshot, local text files, and any PDF
                     you drop into ``data/corpus/inbox/``

Two design points worth noting:

* ``doc_id`` is derived from the source and the external identifier, so the same
  paper always gets the same id and indexes stay stable across rebuilds.
* Structured abstract headings (``Background:`` / ``Methods:`` / ``Results:``)
  are recovered into ``Document.sections``. Europe PMC strips the JATS tags
  without a separator, so headings often arrive *glued* to the body text
  (``"BackgroundAlzheimer's disease is..."``); both forms are handled, because
  section boundaries are what let the chunker keep a Methods claim out of a
  Results chunk.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schema import Document
from .utils import dedupe, get_logger, normalize_whitespace, stable_hash

log = get_logger("neurorag.corpus")

USER_AGENT = "NeuroRAG/1.0 (academic prototype)"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ARXIV = "http://export.arxiv.org/api/query"
CROSSREF = "https://api.crossref.org/works"
DELAY_S = 1.0  # politeness delay: these are free public services


class SourceUnavailable(RuntimeError):
    """A live literature source could not be reached."""


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")

_HEADINGS = (
    "Background|Objective|Objectives|Purpose|Aim|Aims|Introduction|Methods?"
    "|Materials? and Methods?|Methodology|Approach|Results?|Findings"
    "|Discussion|Conclusions?|Implications?|Significance|Highlights"
)
# Heading followed by punctuation: "Methods: we trained ..."
_HEAD_PUNCT = re.compile(rf"(?:^|\n|(?<=\.\s))\s*({_HEADINGS})\s*[:.]\s+", re.IGNORECASE)
# Heading glued to the body: "BackgroundAlzheimer's disease is ..."
# Requires no preceding letter and an uppercase letter immediately after, which
# ordinary in-sentence use of the same word ("the Results were") does not match.
_HEAD_GLUED = re.compile(rf"(?<![A-Za-z])({_HEADINGS})(?=[A-Z0-9(\"])", re.IGNORECASE)


def clean_markup(raw: Optional[str]) -> str:
    """Strip JATS/HTML markup, turning structural tags into newlines first."""
    if not raw:
        return ""
    text = html.unescape(raw)
    text = re.sub(r"(?i)</(p|sec|abstract|title)>", "\n", text)
    text = re.sub(r"(?i)<title>", "\n## ", text)
    return normalize_whitespace(_TAG.sub("", text))


_MD_HEADING = re.compile(r"(?:^|\n)\s*#{1,6}\s+(.+?)\s*(?:\n|$)")


def parse_sections(text: str) -> Dict[str, str]:
    """Recover document sections from plain text.

    Three forms are handled, in priority order:

    1. **Markdown headings** (``## Methods``) - present in author-supplied records
       and some preprint servers.
    2. **Punctuation-delimited headings** (``Methods: we trained ...``).
    3. **Glued headings** (``MethodsWe trained ...``), the common Europe PMC
       artefact where JATS tag removal concatenates heading and body.

    Without step 1 the heading markers stay inside the body text and leak verbatim
    into generated answers, and the whole document collapses to one section.
    """
    if not text or not text.strip():
        return {}

    md = list(_MD_HEADING.finditer(text))
    if md:
        sections: Dict[str, str] = {}
        lead = text[: md[0].start()].strip()
        if lead:
            sections["Preamble"] = normalize_whitespace(lead)
        for i, match in enumerate(md):
            heading = normalize_whitespace(match.group(1)).strip("#").strip().title()
            end = md[i + 1].start() if i + 1 < len(md) else len(text)
            body = normalize_whitespace(text[match.end(): end])
            if not body:
                continue
            sections[heading] = f"{sections[heading]}\n\n{body}" if heading in sections else body
        if sections:
            if list(sections) == ["Preamble"]:
                return {"Abstract": sections["Preamble"]}
            return sections

    matches = list(_HEAD_PUNCT.finditer(text))
    if matches:
        spans = [(m.start(1), m.end()) for m in matches]
    else:
        glued = [
            m for m in _HEAD_GLUED.finditer(text)
            if m.start() == 0 or re.search(r"[.!?:;\)\]]\s*$", text[: m.start()])
        ]
        if not glued:
            return {"Abstract": text.strip()}
        spans = [(m.start(1), m.end(1)) for m in glued]

    sections: Dict[str, str] = {}
    lead = text[: spans[0][0]].strip()
    if lead:
        sections["Preamble"] = lead
    for i, (head_start, body_start) in enumerate(spans):
        heading = text[head_start:body_start].strip().rstrip(":.").title()
        end = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        body = normalize_whitespace(text[body_start:end])
        if not body:
            continue
        sections[heading] = f"{sections[heading]}\n\n{body}" if heading in sections else body

    if list(sections) == ["Preamble"]:
        return {"Abstract": sections["Preamble"]}
    return sections or {"Abstract": text.strip()}


def parse_authors(names: Any) -> List[str]:
    """Normalise the several author shapes the APIs return."""
    if not names:
        return []
    if isinstance(names, str):
        return dedupe([p.strip(" .") for p in re.split(r"\s*,\s*|\s+and\s+", names) if p.strip()])[:24]
    if isinstance(names, dict):
        return parse_authors(names.get("author", names.get("authors")))
    if isinstance(names, list):
        out = []
        for entry in names:
            if isinstance(entry, str):
                out.append(entry.strip())
            elif isinstance(entry, dict):
                name = entry.get("fullName") or entry.get("name") or entry.get("collectiveName")
                if name:
                    out.append(str(name).strip())
        return dedupe(out)[:24]
    return []


def parse_year(value: Any) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(m.group(0)) if m else None


def _get(url: str, timeout: int = 30, retries: int = 2) -> bytes:
    """GET with retry; raises SourceUnavailable if the source is unreachable."""
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(DELAY_S * (attempt + 1))
    raise SourceUnavailable(f"Could not reach {url[:110]}: {last}")


# ---------------------------------------------------------------------------
# Live sources
# ---------------------------------------------------------------------------


def fetch_europepmc(query: str, max_docs: int = 25) -> List[Document]:
    """Search Europe PMC; returns Documents carrying abstract text."""
    params = {"query": query, "format": "json", "resultType": "core",
              "pageSize": str(min(max(max_docs, 1), 100))}
    try:
        payload = json.loads(_get(f"{EUROPEPMC}?{urllib.parse.urlencode(params)}").decode())
    except Exception as exc:
        raise SourceUnavailable(f"Europe PMC unusable response: {exc}") from exc

    docs: List[Document] = []
    for item in (payload.get("resultList") or {}).get("result") or []:
        title = clean_markup(item.get("title")).strip()
        abstract = clean_markup(item.get("abstractText"))
        if not title or not abstract:
            continue
        pmid, pmcid, doi = item.get("pmid"), item.get("pmcid"), item.get("doi")
        external = pmcid or pmid or doi or stable_hash(title)
        journal = ((item.get("journalInfo") or {}).get("journal") or {}).get("title")
        docs.append(Document(
            doc_id=f"epmc:{external}".lower(),
            title=title,
            text=abstract,
            source="europepmc",
            year=parse_year(item.get("pubYear") or item.get("firstPublicationDate")),
            authors=parse_authors(item.get("authorString") or item.get("authorList")),
            venue=journal or item.get("journalTitle"),
            doi=doi,
            url=f"https://europepmc.org/article/MED/{pmid}" if pmid
                else (f"https://europepmc.org/article/{pmcid}" if pmcid else None),
            sections=parse_sections(abstract),
            note="Abstract text retrieved from Europe PMC.",
        ))
        if len(docs) >= max_docs:
            break
    log.info("Europe PMC '%s' -> %d docs", query[:55], len(docs))
    return docs


_ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _arxiv_entry_to_doc(entry) -> Optional[Document]:
    title_el = entry.find("a:title", _ATOM)
    summary_el = entry.find("a:summary", _ATOM)
    if title_el is None or summary_el is None:
        return None
    title = normalize_whitespace(title_el.text or "")
    abstract = normalize_whitespace(summary_el.text or "")
    if not title or not abstract:
        return None
    id_el = entry.find("a:id", _ATOM)
    arxiv_id = re.sub(r"v\d+$", "", (id_el.text or "").strip().rsplit("/", 1)[-1])
    published = entry.find("a:published", _ATOM)
    authors = [
        normalize_whitespace(a.find("a:name", _ATOM).text or "")
        for a in entry.findall("a:author", _ATOM) if a.find("a:name", _ATOM) is not None
    ]
    return Document(
        doc_id=f"arxiv:{arxiv_id}".lower(),
        title=title,
        text=abstract,
        source="arxiv",
        year=parse_year(published.text if published is not None else ""),
        authors=dedupe([a for a in authors if a]),
        venue="arXiv preprint",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        sections=parse_sections(abstract),
        tags=["preprint"],
        note="Abstract text retrieved from the arXiv API.",
    )


def fetch_arxiv(query: str, max_docs: int = 15) -> List[Document]:
    """Relevance search over arXiv."""
    params = {"search_query": f"all:{query}", "start": "0",
              "max_results": str(min(max(max_docs, 1), 100)),
              "sortBy": "relevance", "sortOrder": "descending"}
    try:
        root = ET.fromstring(_get(f"{ARXIV}?{urllib.parse.urlencode(params)}").decode())
    except ET.ParseError as exc:
        raise SourceUnavailable(f"arXiv unparseable XML: {exc}") from exc
    docs = [d for d in (_arxiv_entry_to_doc(e) for e in root.findall("a:entry", _ATOM)) if d]
    log.info("arXiv '%s' -> %d docs", query[:55], len(docs[:max_docs]))
    return docs[:max_docs]


def fetch_arxiv_ids(arxiv_ids: Iterable[str]) -> List[Document]:
    """Resolve explicit arXiv ids.

    Relevance search does not reliably surface canonical primary sources - a
    query for "Swin UNETR" can rank a citing paper above the original - so papers
    that must be present are pinned by identifier instead.
    """
    ids = dedupe([a.strip() for a in arxiv_ids if a.strip()])
    if not ids:
        return []
    url = f"{ARXIV}?{urllib.parse.urlencode({'id_list': ','.join(ids), 'max_results': len(ids)})}"
    try:
        root = ET.fromstring(_get(url).decode())
    except ET.ParseError as exc:
        raise SourceUnavailable(f"arXiv id_list unparseable XML: {exc}") from exc
    docs = [d for d in (_arxiv_entry_to_doc(e) for e in root.findall("a:entry", _ATOM)) if d]
    for d in docs:
        d.tags = sorted(set(d.tags + ["pinned"]))
    log.info("arXiv id_list (%d ids) -> %d docs", len(ids), len(docs))
    return docs


def fetch_doi(doi: str) -> Optional[Document]:
    """Bibliographic metadata from Crossref (abstract only if the publisher
    deposited one, which for Elsevier it usually has not)."""
    url = f"{CROSSREF}/{urllib.parse.quote(doi, safe='')}"
    try:
        payload = json.loads(_get(url).decode())["message"]
    except Exception as exc:
        log.warning("Crossref lookup failed for %s: %s", doi, exc)
        return None

    title = normalize_whitespace((payload.get("title") or [""])[0])
    if not title:
        return None
    authors = []
    for a in payload.get("author", []) or []:
        if a.get("family"):
            authors.append(f"{a.get('given', '').strip()} {a['family'].strip()}".strip())
        elif a.get("name"):
            authors.append(str(a["name"]).strip())
    year = None
    for key in ("published-print", "published-online", "published", "issued", "created"):
        parts = ((payload.get(key) or {}).get("date-parts") or [[]])[0]
        year = parse_year(parts[0] if parts else None)
        if year:
            break
    abstract = clean_markup(payload.get("abstract"))
    return Document(
        doc_id=f"doi:{doi.lower()}",
        title=title,
        text=abstract,
        source="crossref",
        year=year,
        authors=dedupe(authors),
        venue=(payload.get("container-title") or [""])[0] or None,
        doi=doi,
        url=f"https://doi.org/{doi}",
        sections=parse_sections(abstract) if abstract else {},
        tags=[] if abstract else ["metadata-only"],
        note="Bibliographic metadata retrieved from Crossref.",
    )


# ---------------------------------------------------------------------------
# Offline sources
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> List[Document]:
    """Load the committed corpus snapshot."""
    from .utils import read_jsonl

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No corpus at {path}. Run `python scripts/build_corpus.py`, or set corpus_path.")
    docs = [Document.from_dict(row) for row in read_jsonl(path)]
    log.info("Loaded %d documents from %s", len(docs), path)
    return docs


def load_text_dir(directory: str | Path) -> List[Document]:
    """Load .txt/.md files as documents - handy for your own notes or a lit review."""
    root = Path(directory)
    if not root.exists():
        return []
    docs = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".markdown"} or not path.is_file():
            continue
        text = normalize_whitespace(path.read_text(encoding="utf-8", errors="ignore"))
        if len(text) < 80:
            continue
        docs.append(Document(
            doc_id=f"local:{stable_hash(path.name, text[:200])}",
            title=(text.splitlines()[0].lstrip('# ').strip() or path.stem)[:200],
            text=text, source="local", stratum="methods", topic="local_notes",
            url=path.name, sections=parse_sections(text),
            note="Local file supplied by the repository owner.",
        ))
    if docs:
        log.info("Loaded %d local text documents from %s", len(docs), root)
    return docs


def load_pdf(path: str | Path) -> Optional[Document]:
    """Best-effort PDF ingestion for papers you supply yourself."""
    path = Path(path)
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed; skipping %s (pip install pypdf)", path.name)
        return None
    try:
        reader = PdfReader(str(path))
        text = normalize_whitespace("\n\n".join((p.extract_text() or "") for p in reader.pages))
    except Exception as exc:
        log.warning("Could not parse %s: %s", path.name, exc)
        return None
    if len(text) < 200:
        log.warning("PDF %s yielded too little text; skipping", path.name)
        return None
    meta = reader.metadata or {}
    return Document(
        doc_id=f"pdf:{stable_hash(path.name, text[:200])}",
        title=normalize_whitespace(str(meta.get("/Title") or ""))[:300] or path.stem,
        text=text, source="pdf", stratum="methods", topic="user_supplied_pdf",
        authors=parse_authors(str(meta.get("/Author") or "")),
        url=path.name, sections=parse_sections(text),
        note="User-supplied PDF. Check publisher terms before redistributing.",
    )


def load_inbox(directory: str | Path = "data/corpus/inbox") -> List[Document]:
    """Load everything a user dropped into the inbox (PDFs and text files)."""
    root = Path(directory)
    if not root.exists():
        return []
    docs: List[Document] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            doc = load_pdf(path)
            if doc:
                docs.append(doc)
        elif path.suffix.lower() in {".txt", ".md"}:
            docs.extend(load_text_dir(root))
            break
    if docs:
        log.info("Loaded %d documents from inbox %s", len(docs), root)
    return docs


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def dedupe_documents(docs: Iterable[Document]) -> List[Document]:
    """Collapse duplicates by id and by DOI, keeping whichever record has more text.

    The DOI pass matters when the same paper arrives twice under different id
    namespaces - e.g. a Crossref metadata stub and an author-supplied record that
    carries real text. Without it the corpus ends up citing a stub for a paper it
    already has in full.
    """
    by_id: Dict[str, Document] = {}
    for doc in docs:
        existing = by_id.get(doc.doc_id)
        if existing is None:
            by_id[doc.doc_id] = doc
        elif len(doc.text) > len(existing.text):
            doc.tags = sorted(set(doc.tags) | set(existing.tags))
            by_id[doc.doc_id] = doc
        else:
            existing.tags = sorted(set(existing.tags) | set(doc.tags))

    best_by_doi: Dict[str, Document] = {}
    for doc in by_id.values():
        if not doc.doi:
            continue
        key = doc.doi.lower()
        if key not in best_by_doi or len(doc.text) > len(best_by_doi[key].text):
            best_by_doi[key] = doc

    drop: set[str] = set()
    for doi, winner in best_by_doi.items():
        for doc in by_id.values():
            if doc.doi and doc.doi.lower() == doi and doc.doc_id != winner.doc_id:
                winner.tags = sorted(set(winner.tags) | set(doc.tags))
                drop.add(doc.doc_id)
    if drop:
        log.info("Collapsed %d duplicate record(s) by DOI: %s", len(drop), sorted(drop))
    return [d for did, d in by_id.items() if did not in drop]


def synthesise_bibliographic_text(doc: Document) -> Document:
    """Give a metadata-only record a citable body.

    Clearly labelled as a stub, so a retrieved chunk can never masquerade as the
    paper's own prose. It still lets the system answer "who wrote X / where was X
    published" correctly instead of hallucinating.
    """
    lines = [
        "BIBLIOGRAPHIC RECORD (no open-access abstract available)",
        f"Title: {doc.title}",
        f"Authors: {', '.join(doc.authors) if doc.authors else 'not recorded'}",
    ]
    if doc.year:
        lines.append(f"Year: {doc.year}")
    if doc.venue:
        lines.append(f"Published in: {doc.venue}")
    if doc.doi:
        lines.append(f"DOI: {doc.doi}")
    if doc.url:
        lines.append(f"URL: {doc.url}")
    lines.append(
        "Note: only bibliographic metadata for this work is openly deposited, so "
        "NeuroRAG can cite it but cannot quote its findings. Add the publisher PDF "
        "to data/corpus/inbox/ to index the full text."
    )
    doc.text = normalize_whitespace("\n".join(lines))
    doc.sections = {"Bibliographic record": doc.text}
    doc.tags = sorted(set(doc.tags + ["metadata_only_stub"]))
    return doc
