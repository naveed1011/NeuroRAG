#!/usr/bin/env python3
"""Build (or rebuild) the retrieval index from the corpus snapshot.

    python scripts/build_index.py
    python scripts/build_index.py --set chunk_size=512 --set chunk_overlap=96
    python scripts/build_index.py --set embed_model=NeuML/pubmedbert-base-embeddings
    python scripts/build_index.py --lexical      # no ML dependencies needed
    python scripts/build_index.py --no-rerank
    python scripts/build_index.py --json         # machine-readable stats

Writes ``data/index/`` containing the dense vectors, the chunk metadata, the BM25
token lists and ``index_meta.json``. The metadata records which embedder produced
the vectors, and ``RAGPipeline.load`` refuses to load an index with a different
embedder - comparing query vectors against stored vectors from another model
would make every score meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neurorag.config import Config, parse_overrides  # noqa: E402
from neurorag.pipeline import RAGPipeline  # noqa: E402
from neurorag.utils import get_logger, set_seed  # noqa: E402

log = get_logger("neurorag.build_index")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the NeuroRAG retrieval index.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="Override a config value, e.g. --set top_k=10")
    parser.add_argument("--lexical", action="store_true",
                        help="Force the offline lexical encoder (no ML dependencies).")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder.")
    parser.add_argument("--json", action="store_true", help="Print stats as JSON.")
    args = parser.parse_args()

    config = Config.load(args.config)
    overrides = parse_overrides(args.overrides)
    if args.lexical:
        overrides["embed_backend"] = "lexical"
    if args.no_rerank:
        overrides["rerank"] = False
    if overrides:
        config = config.merge(overrides)
        log.info("Overrides applied: %s", overrides)

    set_seed(config.seed)
    for warning in config.warnings():
        log.warning("Config: %s", warning)

    pipeline = RAGPipeline.build(config)
    stats = pipeline.stats()

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
        return 0

    print()
    print("=" * 74)
    print("INDEX BUILT")
    print("=" * 74)
    print(f"  documents    : {stats['n_documents']}")
    print(f"  chunks       : {stats['n_chunks']}")
    cs = stats.get("chunk_stats") or {}
    if cs:
        print(f"  chunk size   : {config.chunk_size}+{config.chunk_overlap} chars "
              f"(mean {cs.get('mean_chars')}, median {cs.get('median_chars')}, "
              f"min {cs.get('min_chars')}, max {cs.get('max_chars')})")
        print(f"  sections     : {cs.get('n_sections')} distinct section labels recovered")
    print(f"  retriever    : {stats['retriever']}")
    print(f"  embedding dim: {stats['embedding_dim']}")
    print(f"  bm25 terms   : {stats['bm25_terms']}")
    print(f"  index dir    : {ROOT / config.index_dir}")
    timings = stats.get("timings_ms") or {}
    if timings:
        print(f"  build time   : " + ", ".join(f"{k}={v / 1000:.1f}s" for k, v in timings.items()))
    print("=" * 74)
    if "LEXICAL-FALLBACK" in stats["retriever"]:
        print("  NOTE: no semantic encoder available - retrieval is lexical only.")
        print("        Install sentence-transformers for real embeddings:")
        print("          pip install sentence-transformers")
    if "reranker=noop" in stats["retriever"]:
        print("  NOTE: cross-encoder unavailable - reranking preserves first-stage order.")
    print()
    print("Next:  python scripts/ask.py \"your question here\"")
    print("   or: python scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
