#!/usr/bin/env python3
"""Build the literature corpus from open APIs.

    python scripts/build_corpus.py               # fetch and write the snapshot
    python scripts/build_corpus.py --dry-run     # show the fetch plan only
    python scripts/build_corpus.py --verify      # validate the existing snapshot

Topics come from ``config/corpus_topics.yaml``; canonical primary sources that
keyword search may not surface are pinned by identifier in
``config/pinned_records.yaml``.

The output is a **frozen snapshot** at ``data/corpus/corpus.jsonl``, committed to
the repository so every number in the evaluation report is reproducible by anyone
who clones it, independent of live API state. Re-running this refreshes the corpus
and will change the numbers.

If a source is unreachable the build logs a warning and continues with the rest;
``--strict`` turns that into a hard error.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from neurorag.corpus import (  # noqa: E402
    SourceUnavailable,
    dedupe_documents,
    fetch_arxiv,
    fetch_arxiv_ids,
    fetch_doi,
    fetch_europepmc,
    load_inbox,
    parse_sections,
    synthesise_bibliographic_text,
)
from neurorag.schema import Document  # noqa: E402
from neurorag.utils import ensure_dir, get_logger, normalize_whitespace, read_json, read_jsonl, stable_hash, write_json, write_jsonl  # noqa: E402

log = get_logger("neurorag.build_corpus")

TOPICS = ROOT / "config" / "corpus_topics.yaml"
PINNED = ROOT / "config" / "pinned_records.yaml"
DEFAULT_OUT = ROOT / "data" / "corpus" / "corpus.jsonl"
DELAY_S = 1.2  # politeness delay between live API calls


def load_pinned(path: Path = PINNED) -> list[Document]:
    """Resolve the pinned primary sources: arXiv ids, DOIs, author-supplied records."""
    if not path.exists():
        return []
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    docs: list[Document] = []

    ids, id_meta, id_tags = [], {}, {}
    for group in spec.get("arxiv", []) or []:
        entries = group.get("ids", []) if isinstance(group, dict) else [group]
        meta = {k: group[k] for k in ("stratum", "topic") if isinstance(group, dict) and group.get(k)}
        tags = (group.get("tags") or []) if isinstance(group, dict) else []
        for aid in entries:
            aid = str(aid).strip()
            ids.append(aid)
            id_meta.setdefault(aid, {}).update(meta)
            id_tags.setdefault(aid, []).extend(tags)
    if ids:
        try:
            for doc in fetch_arxiv_ids(ids):
                meta = id_meta.get(doc.doc_id.split(":", 1)[-1], {})
                doc.stratum = meta.get("stratum", doc.stratum)
                doc.topic = meta.get("topic", doc.topic)
                doc.tags = sorted(set(doc.tags + id_tags.get(doc.doc_id.split(":", 1)[-1], [])))
                docs.append(doc)
        except SourceUnavailable as exc:
            log.warning("Could not resolve pinned arXiv ids: %s", exc)

    for entry in spec.get("crossref", []) or []:
        doi = entry.get("doi") if isinstance(entry, dict) else entry
        if not doi:
            continue
        try:
            doc = fetch_doi(doi)
        except SourceUnavailable as exc:
            log.warning("Could not resolve DOI %s: %s", doi, exc)
            doc = None
        if doc is None:
            continue
        if isinstance(entry, dict):
            doc.stratum = entry.get("stratum", doc.stratum)
            doc.topic = entry.get("topic", doc.topic)
            doc.note = entry.get("note", doc.note)
        docs.append(doc)
        time.sleep(DELAY_S)

    for record in spec.get("records", []) or []:
        text = normalize_whitespace(record.get("text", ""))
        if not text:
            continue
        docs.append(Document(
            doc_id=record.get("doc_id") or f"manual:{stable_hash(record.get('title', ''), text[:120])}",
            title=record.get("title", "Untitled"),
            text=text, source=record.get("source", "author_supplied"),
            stratum=record.get("stratum", "clinical"), topic=record.get("topic", "own_publication"),
            year=record.get("year"), authors=record.get("authors") or [],
            venue=record.get("venue"), doi=record.get("doi"),
            url=record.get("url") or (f"https://doi.org/{record['doi']}" if record.get("doi") else None),
            sections=parse_sections(text),
            tags=sorted(set((record.get("tags") or []) + ["author_supplied_summary"])),
            note=record.get("note", "Author-supplied summary; replace with the publisher PDF in data/corpus/inbox/."),
        ))

    log.info("Pinned records contributed %d documents", len(docs))
    return docs


def summarise(docs: list[Document]) -> dict:
    lengths = sorted(len(d.text) for d in docs)
    n = len(lengths) or 1
    years = Counter(d.year for d in docs if d.year)
    topics: dict[str, int] = defaultdict(int)
    for d in docs:
        topics[d.topic] += 1
    return {
        "n_documents": len(docs),
        "by_stratum": dict(Counter(d.stratum for d in docs)),
        "by_source": dict(Counter(d.source for d in docs)),
        "by_topic": dict(sorted(topics.items())),
        "year_range": [min(years) if years else None, max(years) if years else None],
        "median_text_chars": lengths[n // 2] if lengths else 0,
        "total_text_chars": sum(lengths),
        "approx_tokens": round(sum(lengths) / 4),
        "n_with_sections": sum(1 for d in docs if len(d.sections) > 1),
        "n_with_doi": sum(1 for d in docs if d.doi),
        "n_with_authors": sum(1 for d in docs if d.authors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the NeuroRAG literature corpus.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--reparse-sections", action="store_true",
                        help="Re-derive Document.sections from the existing snapshot without "
                             "re-fetching. Keeps every doc_id identical, so the golden set's "
                             "gold labels stay valid.")
    parser.add_argument("--strict", action="store_true", help="Fail on any unreachable source.")
    parser.add_argument("--min-chars", type=int, default=250, help="Minimum text length to keep.")
    parser.add_argument("--no-inbox", action="store_true", help="Skip data/corpus/inbox/.")
    args = parser.parse_args()

    if args.verify:
        docs = [Document.from_dict(r) for r in read_jsonl(args.out)]
        stats = summarise(docs)
        print(f"Snapshot {args.out} is valid: {stats['n_documents']} documents")
        for stratum, count in stats["by_stratum"].items():
            print(f"  {stratum:>10}: {count}")
        print(f"  year range: {stats['year_range'][0]}..{stats['year_range'][1]}")
        print(f"  ~{stats['approx_tokens']:,} tokens total")
        return 0

    if args.reparse_sections:
        from neurorag.corpus import load_jsonl

        existing = load_jsonl(args.out)
        changed = 0
        for doc in existing:
            before = len(doc.sections)
            doc.sections = parse_sections(doc.text)
            if len(doc.sections) != before:
                changed += 1
        write_jsonl(args.out, [d.to_dict() for d in existing])
        manifest_path = args.out.parent / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            manifest["statistics"] = summarise(existing)
            manifest["reparse_sections_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_json(manifest_path, manifest)
        log.info("Re-parsed sections for %d documents (%d changed); doc_ids untouched.",
                 len(existing), changed)
        print(f"\nRe-parsed {len(existing)} documents in place ({changed} gained section structure).")
        print("Next: python scripts/build_index.py")
        return 0

    spec = yaml.safe_load(TOPICS.read_text(encoding="utf-8")) or {}
    plan = [{"name": e["name"], "source": e.get("source", "europepmc"), "query": e["query"],
             "max_docs": e.get("max_docs", 20), "stratum": stratum}
            for stratum, entries in (spec.get("strata") or {}).items() for e in (entries or [])]

    if args.dry_run:
        print(f"Would fetch {len(plan)} topics:")
        for t in plan:
            print(f"  {t['stratum']:<9} {t['name']:<28} {t['source']:<10} max={t['max_docs']:<4} {t['query'][:55]}")
        print(f"Plus pinned records from {PINNED.name}")
        return 0

    documents: list[Document] = []
    audit: list[dict] = []
    fetchers = {"europepmc": fetch_europepmc, "arxiv": fetch_arxiv}

    for i, topic in enumerate(plan, start=1):
        label = f"[{i}/{len(plan)}] {topic['stratum']}/{topic['name']}"
        try:
            fetched = fetchers[topic["source"]](topic["query"], max_docs=topic["max_docs"])
        except SourceUnavailable as exc:
            message = f"{label}: source unavailable -> {exc}"
            if args.strict:
                raise SystemExit(message)
            log.warning(message)
            audit.append({**{k: topic[k] for k in ("name", "stratum", "source")}, "fetched": 0, "error": str(exc)})
            continue
        except Exception as exc:
            message = f"{label}: unexpected error -> {exc}"
            if args.strict:
                raise SystemExit(message)
            log.warning(message)
            audit.append({**{k: topic[k] for k in ("name", "stratum", "source")}, "fetched": 0, "error": str(exc)})
            continue

        for doc in fetched:
            doc.stratum, doc.topic = topic["stratum"], topic["name"]
        documents.extend(fetched)
        audit.append({**{k: topic[k] for k in ("name", "stratum", "source")}, "fetched": len(fetched), "error": None})
        log.info("%s -> %d docs (total %d)", label, len(fetched), len(documents))
        if i < len(plan):
            time.sleep(DELAY_S)

    documents.extend(load_pinned())

    if not args.no_inbox:
        inbox = load_inbox(ROOT / "data" / "corpus" / "inbox")
        if inbox:
            log.info("Added %d user-supplied documents from inbox", len(inbox))
            documents.extend(inbox)

    documents = dedupe_documents(documents)

    # Metadata-only pinned records are worth keeping (they let the system cite a
    # canonical source correctly) but get an explicitly-labelled stub body rather
    # than being dropped for having too little text.
    kept, dropped = [], 0
    for doc in documents:
        if len((doc.text or "").strip()) >= args.min_chars:
            kept.append(doc)
        elif "pinned" in doc.tags or doc.source in {"crossref", "author_supplied"}:
            kept.append(synthesise_bibliographic_text(doc))
        else:
            dropped += 1
    documents = sorted(kept, key=lambda d: (d.stratum, d.topic, d.doc_id))

    ensure_dir(args.out.parent)
    n = write_jsonl(args.out, [d.to_dict() for d in documents])
    stats = summarise(documents)
    write_json(ROOT / "data" / "corpus" / "manifest.json", {
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snapshot_hash": stable_hash(*[d.doc_id for d in documents], length=16),
        "dropped_short_documents": dropped, "min_chars": args.min_chars,
        "statistics": stats, "fetch_audit": audit,
    })

    log.info("=" * 72)
    log.info("Wrote %d documents to %s", n, args.out)
    log.info("Strata : %s", stats["by_stratum"])
    log.info("Sources: %s", stats["by_source"])
    log.info("Years %s..%s | median %d chars | ~%s tokens | %d/%d with section structure",
             stats["year_range"][0], stats["year_range"][1], stats["median_text_chars"],
             f"{stats['approx_tokens']:,}", stats["n_with_sections"], stats["n_documents"])
    log.info("=" * 72)
    print(f"\nNext: python scripts/build_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
