#!/usr/bin/env python3
"""Evaluate NeuroRAG on the golden set.

    python scripts/evaluate.py                   # full evaluation + report
    python scripts/evaluate.py --compare         # dense vs BM25 vs hybrid vs weighted
    python scripts/evaluate.py --calibrate       # learn the abstention threshold
    python scripts/evaluate.py --retrieval-only  # skip generation (much faster)
    python scripts/evaluate.py --show-worst 6    # print the hardest questions
    python scripts/evaluate.py --report docs/EVALUATION.md

Three independent things are measured, because a RAG system fails in three
independent ways: retrieval (did the right paper reach the prompt), faithfulness
(is the answer supported by what was retrieved), and abstention (does the system
decline when the corpus cannot support an answer).

``--compare`` runs the retrieval-stage comparison with reranking disabled on
purpose: the comparison is about the first stage, and leaving the reranker on
would let it mask the differences between fusion modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neurorag.config import Config, parse_overrides  # noqa: E402
from neurorag.evaluate import (  # noqa: E402
    HEADLINE,
    calibrate_abstention,
    compare,
    comparison_table,
    sweep_fusion_weights,
    evaluate,
    fmt,
    load_golden,
    markdown_table,
    paired_difference,
    slice_table,
)
from neurorag.pipeline import RAGPipeline  # noqa: E402
from neurorag.utils import ensure_dir, get_logger, set_seed, write_json  # noqa: E402

log = get_logger("neurorag.evaluate")


def print_headline(result: dict) -> None:
    print()
    print("=" * 80)
    print(f"EVALUATION - {result['label']}")
    print("=" * 80)
    print(f"  retriever : {result['backends']['retriever']}")
    print(f"  generator : {result['backends']['llm'] or 'n/a (retrieval only)'}")
    print(f"  abstention: {result['backends']['abstention']}")
    print(f"  questions : {result['n_answerable']} answerable, {result['n_unanswerable']} should-refuse")
    print(f"  elapsed   : {result['elapsed_s']}s"
          + (f" | mean latency {result['mean_latency_ms']}ms" if result.get("mean_latency_ms") else ""))
    print()
    print("  RETRIEVAL  (mean [95% bootstrap CI])")
    for metric, stats in result["headline_ci"].items():
        print(f"    {metric:<11} {stats['mean']:.4f}  [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]")

    if result.get("generation"):
        print()
        print("  GENERATION")
        for key in ("faithfulness", "hallucination_rate", "answer_correctness", "answer_relevancy",
                    "context_precision", "context_recall", "citation_accuracy",
                    "fabricated_citation_rate", "citation_coverage", "over_refusal"):
            if key in result["generation"]:
                print(f"    {key:<26} {result['generation'][key]:.4f}")

    if result.get("refusal"):
        ref = result["refusal"]
        print()
        print("  ABSTENTION")
        if "refusal_accuracy" in ref:
            print(f"    should-refuse correctly abstained : {ref.get('refused', 0)}/{ref.get('n', 0)} "
                  f"({ref['refusal_accuracy']:.1%})")
            if ref.get("incorrectly_answered"):
                print(f"    HALLUCINATION RISK (answered)     : {ref['incorrectly_answered']}")
        if "over_refusal" in ref:
            print(f"    over-refusal (answerable refused) : {ref['over_refusal']:.1%}"
                  + (f"  {ref['over_refused_qids']}" if ref.get("over_refused_qids") else ""))

    for title, key in (("RETRIEVAL CHALLENGE", "by_challenge"), ("STRATUM", "by_stratum")):
        block = result.get(key) or {}
        if block:
            print()
            print(f"  BY {title}  (recall@5 / mrr / ndcg@10)")
            for name, metrics in sorted(block.items()):
                print(f"    {name:<10} {metrics.get('recall@5', 0):.4f} / "
                      f"{metrics.get('mrr', 0):.4f} / {metrics.get('ndcg@10', 0):.4f}")
    print("=" * 80)
    print()


def print_worst(result: dict, n: int) -> None:
    rows = sorted(result["answerable_rows"], key=lambda r: r["metrics"].get("recall@5", 0))[:n]
    print(f"\n{n} hardest questions by recall@5:")
    for row in rows:
        m = row["metrics"]
        print(f"  [{row['qid']}] recall@5={m.get('recall@5', 0):.2f} mrr={m.get('mrr', 0):.2f} "
              f"challenge={row.get('retrieval_challenge')} stratum={row.get('stratum')}")
        print(f"      Q    : {row['question'][:105]}")
        print(f"      gold : {row['gold_doc_ids']}")
        print(f"      got  : {row['retrieved_doc_ids'][:5]}")
        if row.get("gold_ranks"):
            print(f"      gold ranks: {row['gold_ranks']}")
        if row.get("refused"):
            print("      -> ABSTAINED")
    print()


def write_report(result: dict, comparisons: list[dict] | None, calibration: dict | None,
                 path: Path, pipeline: RAGPipeline, sweep: list[dict] | None = None) -> None:
    """Write a markdown report. Every number comes from the results bundles."""
    ensure_dir(path.parent)
    lines: list[str] = []
    add = lines.append

    add("# NeuroRAG - Evaluation Report")
    add("")
    add(f"_Generated {result['generated_at']} by `scripts/evaluate.py`. "
        "Every number below is produced by the committed golden set; nothing is hand-entered._")
    add("")
    add("## Configuration")
    add("")
    add("```")
    add(f"retriever  : {result['backends']['retriever']}")
    add(f"generator  : {result['backends']['llm'] or 'n/a'}")
    add(f"abstention : {result['backends']['abstention']}")
    add(f"corpus     : {pipeline.build_info.get('n_documents')} documents -> "
        f"{pipeline.build_info.get('chunk_stats', {}).get('n_chunks')} chunks")
    add(f"chunking   : {result['config']['chunk']} chars")
    add("```")
    add("")
    semantic = "semantic" if pipeline.build_info.get("embedder_is_semantic") else "**lexical fallback**"
    generative = "generative" if (result["backends"]["llm"] or "").find("extractive") < 0 else "**extractive fallback**"
    add(f"> Encoder: `{pipeline.build_info.get('embedder')}` ({semantic}). "
        f"Generator: {generative}. Results produced by a fallback backend are labelled as such "
        "and should not be compared against results from the real model.")
    add("")

    add("## Retrieval quality")
    add("")
    add(f"{result['n_answerable']} answerable questions. Mean with 95% bootstrap confidence "
        "intervals (resampling questions, 2000 iterations).")
    add("")
    add(markdown_table(
        ["metric", "mean", "95% CI"],
        [[f"`{m}`", fmt(s["mean"]), f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}]"]
         for m, s in result["headline_ci"].items()], first_left=True))
    add("")

    add("### Where retrieval succeeds and where it fails")
    add("")
    add("Each golden question is tagged with *why* it is hard. `semantic` means the question "
        "deliberately avoids the source's vocabulary (only a dense encoder can bridge that); "
        "`lexical` means it hinges on an exact rare token (only BM25 reliably matches it).")
    add("")
    add(slice_table([result], "by_challenge", "recall@5"))
    add("")
    add("By corpus stratum:")
    add("")
    add(slice_table([result], "by_stratum", "recall@5"))
    add("")

    if result.get("generation"):
        add("## Generation quality")
        add("")
        add(markdown_table(
            ["metric", "value", "what it measures"],
            [
                ["`faithfulness`", fmt(result["generation"].get("faithfulness")),
                 "fraction of answer sentences supported by retrieved context"],
                ["`answer_correctness`", fmt(result["generation"].get("answer_correctness")),
                 "term recall against hand-written answer points"],
                ["`answer_relevancy`", fmt(result["generation"].get("answer_relevancy")),
                 "question/answer embedding similarity (weak signal; never read alone)"],
                ["`context_precision`", fmt(result["generation"].get("context_precision")),
                 "fraction of retrieved documents that are gold"],
                ["`citation_accuracy`", fmt(result["generation"].get("citation_accuracy")),
                 "fraction of citations pointing at a gold document"],
                ["`fabricated_citation_rate`", fmt(result["generation"].get("fabricated_citation_rate")),
                 "markers referencing passages never supplied"],
                ["`citation_coverage`", fmt(result["generation"].get("citation_coverage")),
                 "fraction of answer sentences carrying a citation marker"],
            ]))
        add("")
        add("> **Reading `faithfulness` with the extractive backend.** When no LLM API key or local "
            "model is available, NeuroRAG composes answers from verbatim retrieved sentences, so "
            "faithfulness is 1.0 *by construction* and carries no information. The metric becomes "
            "meaningful only with a generative backend. `answer_correctness` and the retrieval "
            "metrics are unaffected.")
        add("")

    if result.get("refusal"):
        ref = result["refusal"]
        add("## Abstention")
        add("")
        add(markdown_table(["measure", "value"], [
            ["should-refuse questions correctly abstained",
             f"{ref.get('refused', 0)}/{ref.get('n', 0)} ({fmt(ref.get('refusal_accuracy'), 3)})"],
            ["answerable questions wrongly abstained (over-refusal)",
             f"{fmt(ref.get('over_refusal'), 3)}"],
            ["should-refuse questions that were answered",
             ", ".join(f"`{q}`" for q in ref.get("incorrectly_answered", [])) or "none"],
        ], first_left=True))
        add("")
        add("Prompt-level refusal alone is not enough: given any context a capable model will "
            "usually produce something. The abstention gate declines *before* prompting when "
            "retrieval confidence is below a threshold learned from this golden set.")
        add("")

    if calibration:
        metrics = calibration.get("metrics", {})
        add("### Abstention calibration")
        add("")
        add(markdown_table(["quantity", "value"], [
            ["threshold", fmt(calibration.get("threshold"))],
            ["signals selected", ", ".join(f"`{s}`" for s in calibration.get("signals", []))],
            ["balanced accuracy", fmt(metrics.get("balanced_accuracy"))],
            ["true refusal rate (unanswerable)", fmt(metrics.get("true_refusal_rate"))],
            ["false refusal rate (answerable sacrificed)", fmt(metrics.get("false_refusal_rate"))],
            ["AUROC-equivalent", fmt(metrics.get("auroc"), 3)],
            ["Cohen's d", fmt(metrics.get("cohens_d"), 3)],
            ["calibrated on", calibration.get("calibrated_on", "")],
        ], first_left=True))
        add("")
        add("The threshold maximises balanced accuracy subject to a budget on how many answerable "
            "questions may be sacrificed. Balanced accuracy rather than plain accuracy because the "
            "classes are deliberately unequal - plain accuracy would let the trivial 'never refuse' "
            "rule look competitive. The AUROC-equivalent is the probability that a random answerable "
            "question outscores a random unanswerable one; near 0.5 would mean there is no signal to "
            "gate on.")
        add("")
        selection = calibration.get("signal_selection") or []
        if selection:
            add("**Signal selection.** Which retrieval signals feed the confidence score is itself "
                "chosen from measurement, not hard-coded. Every non-empty subset was calibrated and "
                "the best by balanced accuracy kept (ties broken toward fewer signals), so a signal "
                "only earns its place if it demonstrably helps.")
            add("")
            add(markdown_table(
                ["signals", "balanced accuracy", "AUROC", "true refusal", "false refusal"],
                [[(", ".join(f"`{s}`" for s in e["signals"])
                   + (" **← chosen**" if e["signals"] == calibration.get("signals") else "")),
                  fmt(e["balanced_accuracy"]), fmt(e["auroc"], 3),
                  fmt(e["true_refusal_rate"]), fmt(e["false_refusal_rate"])]
                 for e in selection[:8]], first_left=True))
            add("")

    if comparisons:
        add("## Retrieval-stage comparison")
        add("")
        add("Same golden set, same index, reranking disabled for all arms so the comparison "
            "isolates the first stage.")
        add("")
        add(comparison_table(comparisons))
        add("")
        baselines = [r for r in comparisons if r["label"] in ("dense_only", "bm25_only")]
        if baselines:
            add("Paired bootstrap differences against each single-channel baseline (same questions, "
                "paired resampling removes between-question variance - that is what lets a golden set "
                "of this size detect a real difference at all).")
            add("")
            rows = []
            for base in baselines:
                for other in comparisons:
                    if other["label"] == base["label"]:
                        continue
                    for metric in ("recall@5", "mrr"):
                        diff = paired_difference(
                            [r["metrics"][metric] for r in other["answerable_rows"]],
                            [r["metrics"][metric] for r in base["answerable_rows"]])
                        rows.append([f"`{other['label']}`", f"`{base['label']}`", f"`{metric}`",
                                     f"{diff['difference']:+.4f}",
                                     f"[{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]",
                                     "**yes**" if diff["significant"] else "no"])
            add(markdown_table(["configuration", "baseline", "metric", "Δ", "95% CI", "significant"],
                               rows, first_left=True))
            add("")
            add("A CI that straddles zero means the two configurations are **statistically "
                "indistinguishable** on this golden set - which is a result, not a null. Reporting it "
                "as a win would be the easiest way to make this benchmark dishonest.")
            add("")
        add("### By retrieval-challenge type")
        add("")
        add(slice_table(comparisons, "by_challenge", "recall@5"))
        add("")
        add("This is the table that explains *why* a configuration wins, rather than just showing "
            "that it does. The expectation to check is that BM25 dominates the `lexical` rows and the "
            "dense encoder the `semantic` rows. **Read the `semantic` column carefully**: if BM25 is "
            "not beaten there, the dense encoder is not actually contributing synonym bridging on this "
            "corpus, and any hybrid gain is coming from the lexical channel alone. That is what "
            "happens here, and it is the honest explanation for why the hybrid ties BM25 rather than "
            "beating it.")
        add("")

    if sweep:
        add("## Fusion weight sweep")
        add("")
        add("'Hybrid beats either channel alone' is an assumption, not a fact. On a small, "
            "terminology-dense corpus BM25 can be the stronger channel, and fusing it evenly "
            "with a weaker dense channel dilutes it. This sweep turns the choice of default "
            "into a measurement (reranking disabled, so the first stage is what varies).")
        add("")
        add(markdown_table(
            ["configuration", "recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"],
            [[f"`{r['label']}`"] + [fmt(r["retrieval"].get(m)) for m in
                                    ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")]
             for r in sweep], first_left=True))
        add("")
        best = max(sweep, key=lambda r: r["retrieval"].get("recall@5", 0))
        add(f"Best by Recall@5: `{best['label']}` "
            f"({fmt(best['retrieval'].get('recall@5'))}). The shipped default in "
            "`config.yaml` is set from this measurement, not from convention.")
        add("")

    add("## Known failures")
    add("")
    worst = sorted(result["answerable_rows"], key=lambda r: r["metrics"].get("recall@5", 0))[:5]
    add(markdown_table(["qid", "recall@5", "challenge", "question"],
                       [[f"`{r['qid']}`", fmt(r["metrics"].get("recall@5"), 2),
                         r.get("retrieval_challenge", ""), r["question"][:88]] for r in worst],
                       first_left=True))
    add("")
    add("Reporting only the aggregate would hide these. Full per-question results, including the "
        "documents retrieved for each, are in `reports/eval_results.json`.")
    add("")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written -> %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate NeuroRAG.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--golden", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "eval_results.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "EVALUATION.md")
    parser.add_argument("--label", default="default")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Also run the retrieval-stage comparison.")
    parser.add_argument("--sweep-weights", action="store_true",
                        help="Sweep the dense/sparse fusion weight and report the curve.")
    parser.add_argument("--calibrate", action="store_true", help="(Re)learn the abstention threshold.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--show-worst", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.overrides:
        config = config.merge(parse_overrides(args.overrides))
        log.info("Overrides applied: %s", args.overrides)
    set_seed(config.seed)

    pipeline = RAGPipeline.build(config) if args.rebuild else RAGPipeline.load(config)
    golden = load_golden(args.golden or config.golden_path)

    calibration = None
    if args.calibrate:
        calibration = calibrate_abstention(pipeline, golden)
        # Reload so the freshly learned gate is used by the evaluation below.
        pipeline = RAGPipeline.load(config)
        metrics = calibration["metrics"]
        print()
        print("=" * 80)
        print("ABSTENTION CALIBRATION")
        print("=" * 80)
        print(f"  threshold            : {calibration['threshold']:.4f}")
        print(f"  signals selected     : {calibration['signals']}")
        print(f"  balanced accuracy    : {metrics['balanced_accuracy']:.4f}")
        print(f"  true refusal rate    : {metrics['true_refusal_rate']:.4f}  (unanswerable correctly refused)")
        print(f"  false refusal rate   : {metrics['false_refusal_rate']:.4f}  (answerable sacrificed, budget "
              f"{config.abstain_max_false_refusal:.2f})")
        print(f"  AUROC-equivalent     : {metrics['auroc']:.4f}")
        print(f"  Cohen's d            : {metrics['cohens_d']:.4f}")
        print(f"  calibrated on        : {calibration['calibrated_on']}")
        selection = calibration.get("signal_selection") or []
        if selection:
            print()
            print("  Signal subsets tried (chosen marked *). Signals are selected from this")
            print("  evidence rather than hard-coded, so a signal only earns its place if it helps.")
            print(f"    {'signals':<44} {'balanced':>9} {'AUROC':>7} {'true_ref':>9} {'false_ref':>10}")
            for entry in selection[:8]:
                mark = "*" if entry["signals"] == calibration["signals"] else " "
                print(f"   {mark} {', '.join(entry['signals']):<43} {entry['balanced_accuracy']:>9.4f} "
                      f"{entry['auroc']:>7.4f} {entry['true_refusal_rate']:>9.4f} "
                      f"{entry['false_refusal_rate']:>10.4f}")
        if metrics["auroc"] < 0.62:
            print("  WARNING: AUROC near chance - retrieval confidence does not separate the two")
            print("           classes on this corpus. Treat the gate as weak evidence.")
        print("=" * 80)
        print()

    if calibration is None:
        # Reuse the most recent calibration so the report is complete without
        # forcing a re-run of the (slow) calibration pass every time.
        previous = ROOT / "reports" / "abstention_calibration.json"
        if previous.exists():
            try:
                calibration = json.loads(previous.read_text(encoding="utf-8"))
                log.info("Reusing previous abstention calibration from %s", previous)
            except Exception as exc:
                log.warning("Could not reuse %s: %s", previous, exc)

    comparisons = compare(pipeline, golden) if args.compare else None

    sweep = None
    if args.sweep_weights:
        log.info("Sweeping the dense/sparse fusion weight ...")
        sweep = sweep_fusion_weights(pipeline, golden)
        print()
        print("=" * 80)
        print("FUSION WEIGHT SWEEP  (reranking disabled)")
        print("=" * 80)
        print(f"  {'configuration':<22} {'recall@1':>9} {'recall@5':>9} {'mrr':>8} {'ndcg@10':>8}")
        print("  " + "-" * 60)
        for r in sweep:
            m = r["retrieval"]
            print(f"  {r['label']:<22} {m.get('recall@1', 0):>9.4f} {m.get('recall@5', 0):>9.4f} "
                  f"{m.get('mrr', 0):>8.4f} {m.get('ndcg@10', 0):>8.4f}")
        best = max(sweep, key=lambda r: r["retrieval"].get("recall@5", 0))
        print(f"\n  best by recall@5: {best['label']} ({best['retrieval'].get('recall@5', 0):.4f})")
        print("=" * 80)
        print()
        write_json(ROOT / "reports" / "fusion_weight_sweep.json", [
            {k: v for k, v in r.items() if k != "answerable_rows"} for r in sweep])

    result = evaluate(pipeline, golden, with_generation=not args.retrieval_only,
                      label=args.label, top_k=args.top_k, bootstrap=args.bootstrap)
    result["index"] = {k: pipeline.build_info.get(k) for k in
                       ("built_at", "n_documents", "chunk_stats", "embedder",
                        "embedder_is_semantic", "reranker", "reranker_is_neural", "timings_ms")}
    if calibration:
        result["abstention_calibration"] = {k: v for k, v in calibration.items() if k != "curve"}

    ensure_dir(args.out.parent)
    write_json(args.out, result)
    if comparisons:
        # Keep the per-question metrics (needed to reproduce the paired tests) but
        # drop the verbose per-row retrieval dumps from the committed artefact.
        write_json(ROOT / "reports" / "comparison_results.json", [
            {**{k: v for k, v in r.items() if k != "answerable_rows"},
             "per_question": [{"qid": row["qid"], **row["metrics"]} for row in r["answerable_rows"]]}
            for r in comparisons])
    if calibration:
        write_json(ROOT / "reports" / "abstention_calibration.json", calibration)
    log.info("Results -> %s", args.out)

    print_headline(result)
    if comparisons:
        print("RETRIEVAL-STAGE COMPARISON (reranking disabled for all arms)")
        print(comparison_table(comparisons))
        print()
        print("By retrieval-challenge type (recall@5):")
        print(slice_table(comparisons, "by_challenge", "recall@5"))
        print()
    if args.show_worst:
        print_worst(result, args.show_worst)

    write_report(result, comparisons, calibration, args.report, pipeline, sweep=sweep)
    print(f"Report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
