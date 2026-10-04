#!/usr/bin/env python3
"""Ask NeuroRAG a question from the command line.

    python scripts/ask.py "Why does slice-level cross-validation inflate accuracy?"
    python scripts/ask.py "..." --explain            # per-channel retrieval scores
    python scripts/ask.py "..." --show-prompt        # the exact prompt sent to the LLM
    python scripts/ask.py "..." --stratum methods --year-min 2022
    python scripts/ask.py --interactive              # REPL
    python scripts/ask.py "..." --json                # machine-readable Answer

Exit codes: 0 answered, 1 abstained/refused, 2 error.

The retrieval trace (``--explain``) shows the dense, sparse, fused and reranked
score for every passage plus the BM25 terms that matched. That is the single most
useful thing in this script when a retrieval result looks wrong: it tells you
*which* channel was responsible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neurorag.config import Config, parse_overrides  # noqa: E402
from neurorag.generator import build_system_prompt, build_user_prompt, references_block  # noqa: E402
from neurorag.pipeline import RAGPipeline  # noqa: E402
from neurorag.utils import get_logger, truncate  # noqa: E402

log = get_logger("neurorag.ask")

RESET, BOLD, DIM, CYAN, GREEN, RED = "\033[0m", "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[31m"


def c(text: str, code: str, on: bool) -> str:
    return f"{code}{text}{RESET}" if on else text


def render(answer, pipeline: RAGPipeline, args, colour: bool) -> None:
    print()
    if answer.refused:
        print(c("ABSTAINED", RED, colour) + c(f"  {answer.refusal_reason}", DIM, colour))
        print(c("The retrieved passages did not support an answer. This is intended "
                "behaviour: in a clinical literature assistant a confident wrong "
                "answer is worse than no answer.", DIM, colour))
    else:
        print(c("ANSWER", BOLD, colour))
        print(answer.text)

    if answer.citations:
        print()
        print(c("SOURCES", BOLD, colour))
        for citation in sorted(answer.citations, key=lambda x: x.marker):
            authors = ", ".join((citation.authors or [])[:3])
            if len(citation.authors or []) > 3:
                authors += " et al."
            print(c(f"  [{citation.marker}] ", CYAN, colour) + c(f"{authors or 'Unknown'} ({citation.year or 'n.d.'})", BOLD, colour))
            print(f"      {truncate(citation.title, 110)}")
            if citation.section:
                print(c(f"      section: {citation.section}", DIM, colour))
            if citation.url:
                print(c(f"      {citation.url}", DIM, colour))
            print(c(f'      "{truncate(citation.quote, 190)}"', DIM, colour))

    if args.explain and answer.hits:
        print()
        print(c("RETRIEVAL TRACE", BOLD, colour))
        # Show whichever fusion score this mode actually produced, so the column
        # is never a wall of NaN.
        fusion_key = next((k for k in ("weighted", "rrf") if k in answer.hits[0].scores), None)
        fusion_label = fusion_key or "fused"
        print(c(f"  {'#':>2} {'final':>8} {'dense':>8} {'sparse':>8} "
                f"{fusion_label:>8} {'rerank':>8}  document", DIM, colour))
        for hit in answer.hits:
            s = hit.scores
            nan = float("nan")
            print(f"  {hit.rank:>2} {hit.score:>8.4f} {s.get('dense', nan):>8.4f} "
                  f"{s.get('sparse', nan):>8.4f} "
                  f"{(s.get(fusion_key, nan) if fusion_key else nan):>8.4f} "
                  f"{s.get('rerank', nan):>8.4f}  "
                  f"{truncate(hit.chunk.doc_title or hit.doc_id, 50)}")
            if hit.matched_terms:
                print(c(f"      bm25 matched: {', '.join(hit.matched_terms[:8])}", DIM, colour))
        if answer.confidence is not None and pipeline.gate.active:
            print(c(f"  abstention confidence: {answer.confidence:.4f} "
                    f"(threshold {pipeline.gate.threshold:.4f})", DIM, colour))

    audit_info = (answer.config or {}).get("audit")
    if audit_info:
        print()
        print(c("AUDIT", BOLD, colour))
        print(f"  citation coverage : {audit_info['citation_coverage']:.0%} of sentences carry a marker")
        print(f"  unique sources    : {audit_info['unique_sources']} of {audit_info['passages_supplied']} passages supplied")
        fabricated = (answer.config or {}).get("fabricated_markers") or []
        print(f"  fabricated markers: {len(fabricated)}")
    if answer.latency_ms:
        print(c("  latency           : " + ", ".join(f"{k}={v:.0f}ms" for k, v in answer.latency_ms.items()), DIM, colour))
    print(c(f"  backend           : {answer.backend} | {pipeline.retriever.describe()}", DIM, colour))
    print()


def ask_once(pipeline: RAGPipeline, question: str, args, colour: bool):
    filters = {}
    if args.stratum:
        filters["strata"] = args.stratum
    if args.section:
        filters["sections"] = args.section
    if args.year_min is not None:
        filters["year_min"] = args.year_min
    if args.year_max is not None:
        filters["year_max"] = args.year_max
    if args.contains:
        filters["contains"] = args.contains
    filters = filters or None

    if args.show_prompt:
        hits = pipeline.retrieve(question, top_k=args.top_k, filters=filters)
        print(c("=" * 74, DIM, colour))
        print(c("SYSTEM PROMPT", BOLD, colour))
        print(build_system_prompt(pipeline.config))
        print(c("\nUSER PROMPT", BOLD, colour))
        print(build_user_prompt(question, hits, pipeline.config)[:6000])
        print(c("=" * 74, DIM, colour))

    answer = pipeline.ask(question, top_k=args.top_k, filters=filters)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        render(answer, pipeline, args, colour)
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the NeuroRAG literature assistant.")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stratum", action="append", choices=["clinical", "methods", "rag"])
    parser.add_argument("--section", action="append")
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--contains", action="append", help="Require this substring in the passage text.")
    parser.add_argument("--explain", action="store_true", help="Show per-channel retrieval scores.")
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--no-colour", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index instead of loading it.")
    args = parser.parse_args()

    colour = sys.stdout.isatty() and not args.no_colour and not args.json
    config = Config.load(args.config)
    if args.overrides:
        config = config.merge(parse_overrides(args.overrides))

    if args.rebuild:
        pipeline = RAGPipeline.build(config)
    else:
        try:
            pipeline = RAGPipeline.load(config)
        except (FileNotFoundError, RuntimeError) as exc:
            log.warning("Could not load the index (%s); building a fresh one.", exc)
            pipeline = RAGPipeline.build(config)

    if args.interactive:
        print(c("NeuroRAG interactive mode. Type a question, or 'exit' to quit.", BOLD, colour))
        print(c(f"Backends: {pipeline.retriever.describe()}", DIM, colour))
        print(c(f"LLM     : {pipeline.llm.name} "
                f"({'generative' if pipeline.llm.is_generative else 'EXTRACTIVE FALLBACK'})", DIM, colour))
        while True:
            try:
                question = input(c("\nquestion> ", CYAN, colour)).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                return 0
            ask_once(pipeline, question, args, colour)
        return 0

    if not args.question:
        parser.error("provide a question, or use --interactive")

    answer = ask_once(pipeline, " ".join(args.question), args, colour)
    return 1 if answer.refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
