"""Evaluation: metrics, abstention calibration, configuration comparison.

Three things are measured, because a RAG system fails in three independent ways:

1. **Retrieval** - did the right paper reach the prompt at all? Nothing
   downstream can recover from a miss here.
2. **Faithfulness** - is what was said actually supported by what was retrieved?
3. **Abstention** - does the system decline when the corpus cannot support an
   answer? In a clinical literature assistant this matters more than fluency: a
   confident wrong answer is worse than no answer.

Gold labels are **document-level**. The golden set records which *papers* answer a
question, not which chunks, because a paper's relevant claim can legitimately sit
in any of its chunks and pinning a chunk id would make the benchmark brittle to
exactly the variable under study (chunk size). Retrieved chunks are therefore
collapsed to one entry per document before any rank-sensitive metric is computed;
without that, DCG accumulates the same document's gain several times while IDCG
counts it once, and nDCG can exceed 1.0.

No LLM judge is used. A judge model would add nondeterminism, its own biases and a
second API dependency to what should be a reproducible benchmark. The cost is that
faithfulness is measured with an embedding/lexical-entailment proxy, which is
weaker than a proper NLI or LLM judge - it can credit a sentence that reuses
context vocabulary while asserting something different. That limitation is stated
in the report rather than hidden.
"""

from __future__ import annotations

import itertools
import math
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import Config
from .generator import AbstentionGate, audit, confidence_signals, parse_markers, split_sentences
from .schema import Answer, Hit
from .utils import get_logger, normalize_whitespace, read_jsonl, write_json

log = get_logger("neurorag.evaluate")

ROOT = Path(__file__).resolve().parent.parent
HEADLINE = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10", "map")


# ===========================================================================
# Retrieval metrics
# ===========================================================================


def dedupe_by_document(hits: Sequence[Hit]) -> List[Hit]:
    """Collapse hits to one per document, keeping the best rank, then renumber."""
    seen, out = set(), []
    for hit in sorted(hits, key=lambda h: h.rank):
        if hit.doc_id in seen:
            continue
        seen.add(hit.doc_id)
        out.append(hit)
    return [Hit(chunk=h.chunk, score=h.score, rank=i, scores=h.scores,
                matched_terms=h.matched_terms) for i, h in enumerate(out, start=1)]


def recall_at_k(hits: Sequence[Hit], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    if not gold:
        return 0.0
    found = {h.doc_id for h in hits if h.rank <= k and h.doc_id in gold}
    return len(found) / len(gold)


def precision_at_k(hits: Sequence[Hit], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    top = [h for h in hits if h.rank <= k]
    return (sum(1 for h in top if h.doc_id in gold) / len(top)) if top else 0.0


def hit_at_k(hits: Sequence[Hit], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    return 1.0 if any(h.rank <= k and h.doc_id in gold for h in hits) else 0.0


def reciprocal_rank(hits: Sequence[Hit], gold: Iterable[str]) -> float:
    gold = set(gold)
    for hit in sorted(hits, key=lambda h: h.rank):
        if hit.doc_id in gold:
            return 1.0 / hit.rank
    return 0.0


def ndcg_at_k(hits: Sequence[Hit], gold: Iterable[str], k: int) -> float:
    """Normalised DCG. Each gold document contributes once, at its best rank."""
    gold = set(gold)
    if not gold:
        return 0.0
    best: Dict[str, int] = {}
    for hit in sorted(hits, key=lambda h: h.rank):
        if hit.rank > k:
            continue
        if hit.doc_id in gold and hit.doc_id not in best:
            best[hit.doc_id] = hit.rank
    dcg = sum(1.0 / math.log2(rank + 1) for rank in best.values())
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return min(1.0, dcg / idcg) if idcg > 0 else 0.0


def average_precision(hits: Sequence[Hit], gold: Iterable[str]) -> float:
    gold = set(gold)
    if not gold:
        return 0.0
    found, total = 0, 0.0
    for hit in sorted(hits, key=lambda h: h.rank):
        if hit.doc_id in gold:
            found += 1
            total += found / hit.rank
    return total / len(gold)


def retrieval_metrics(hits: Sequence[Hit], gold_doc_ids: Iterable[str],
                        ks: Sequence[int] = (1, 3, 5, 10), ndcg_cutoff: int = 10) -> Dict[str, float]:
    """All retrieval metrics for one question, on the document-collapsed list."""
    gold = list(gold_doc_ids)
    ranked = dedupe_by_document(hits)
    ranks = [h.rank for h in ranked if h.doc_id in set(gold)]
    out = {
        "mrr": reciprocal_rank(ranked, gold),
        "map": average_precision(ranked, gold),
        f"ndcg@{ndcg_cutoff}": ndcg_at_k(ranked, gold, ndcg_cutoff),
        "first_relevant_rank": float(min(ranks)) if ranks else 0.0,
        "n_documents_retrieved": float(len(ranked)),
    }
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(ranked, gold, k)
        out[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        out[f"precision@{k}"] = precision_at_k(ranked, gold, k)
    return out


# ===========================================================================
# Generation metrics
# ===========================================================================

_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is",
         "are", "was", "were", "be", "been", "that", "this", "these", "those", "it",
         "its", "as", "at", "by", "from", "we", "our", "they", "their", "which",
         "can", "may", "also", "than", "then", "there", "here", "not", "but"}


def content_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9\-']+", (text or "").lower())
            if t not in _STOP and len(t) > 2}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def faithfulness(answer_text: str, hits: Sequence[Hit],
                 encode_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
                 threshold: float = 0.55, lexical_floor: float = 0.34) -> Dict[str, Any]:
    """Fraction of answer sentences supported by the retrieved context.

    A sentence counts as supported if **either** test passes:

    * semantic - max cosine similarity to any context sentence >= ``threshold``
    * lexical  - content-token containment in the context >= ``lexical_floor``,
      which catches verbatim extractive output (the case the embedding test
      handles least distinctly) and keeps the metric usable with no encoder.
    """
    sentences = [s for s in split_sentences(answer_text) if len(s) > 25]
    if not sentences:
        return {"faithfulness": 1.0, "n_sentences": 0, "n_supported": 0, "unsupported": []}

    context_text = "\n".join(h.chunk.text for h in hits)
    context_sentences = [s for s in split_sentences(context_text) if len(s) > 25]
    context_tokens = content_tokens(context_text)

    context_vectors: List[List[float]] = []
    answer_vectors: List[List[float]] = []
    if encode_fn is not None and context_sentences:
        try:
            context_vectors = encode_fn(context_sentences)
            answer_vectors = encode_fn(sentences)
        except Exception as exc:
            log.warning("Embedding-based faithfulness unavailable (%s); lexical test only.", exc)

    supported, unsupported = 0, []
    for i, sentence in enumerate(sentences):
        best_sim = 0.0
        if context_vectors and i < len(answer_vectors):
            best_sim = max(_cosine(answer_vectors[i], cv) for cv in context_vectors)
        tokens = content_tokens(sentence)
        containment = len(tokens & context_tokens) / len(tokens) if tokens else 0.0
        if containment >= lexical_floor or best_sim >= threshold:
            supported += 1
        else:
            unsupported.append(sentence[:160])

    return {"faithfulness": supported / len(sentences), "n_sentences": len(sentences),
            "n_supported": supported, "unsupported": unsupported}


def answer_correctness(answer_text: str, answer_points: Sequence[str]) -> Dict[str, Any]:
    """Term recall over hand-written answer points.

    Deliberately a *recall* measure: an answer covering every required point plus
    extra grounded detail is good, and punishing the extra detail would reward
    terseness over completeness. Unlike faithfulness this is anchored in a human
    label, so it catches the failure faithfulness cannot - an answer perfectly
    faithful to an irrelevant passage.
    """
    if not answer_points:
        return {"answer_correctness": 0.0, "points_matched": 0, "missing": []}
    lowered = normalize_whitespace(answer_text).lower()
    tokens = content_tokens(answer_text)
    matched, missing = 0, []
    for point in answer_points:
        needle = point.strip().lower()
        if not needle:
            continue
        if needle in lowered or (content_tokens(needle) and content_tokens(needle) <= tokens):
            matched += 1
        else:
            missing.append(point)
    return {"answer_correctness": matched / len(answer_points),
            "points_matched": matched, "missing": missing}


def context_precision(hits: Sequence[Hit], gold: Iterable[str]) -> float:
    """Fraction of distinct retrieved documents that are gold: how much noise
    reached the prompt."""
    gold = set(gold)
    docs = {h.doc_id for h in hits}
    return (len(docs & gold) / len(docs)) if docs else 0.0


def citation_metrics(answer: Answer, gold: Iterable[str]) -> Dict[str, float]:
    gold = set(gold)
    if not answer.citations:
        return {"citation_accuracy": 0.0, "n_citations": 0, "fabricated_citation_rate": 0.0,
                "citation_coverage": 0.0, "gold_documents_cited": 0.0}
    fabricated = (answer.config or {}).get("fabricated_markers", [])
    total = len(answer.citations) + len(fabricated)
    return {
        "citation_accuracy": sum(1 for c in answer.citations if c.doc_id in gold) / len(answer.citations),
        "n_citations": float(len(answer.citations)),
        "fabricated_citation_rate": (len(fabricated) / total) if total else 0.0,
        "citation_coverage": answer.config.get("audit", {}).get("citation_coverage", 0.0),
        "gold_documents_cited": float(len({c.doc_id for c in answer.citations} & gold)),
    }


def answer_relevancy(question: str, answer_text: str,
                     encode_fn: Optional[Callable[[List[str]], List[List[float]]]] = None) -> float:
    """Cosine similarity between question and answer.

    A weak-but-cheap signal: it catches answers that drift off-topic and refusals
    (which score low). It must never be read alone - a fluent wrong answer scores
    highly.
    """
    if not answer_text.strip():
        return 0.0
    if encode_fn is not None:
        try:
            qv, av = encode_fn([question, answer_text])
            return _cosine(qv, av)
        except Exception:
            pass
    q, a = content_tokens(question), content_tokens(answer_text)
    return len(q & a) / len(q | a) if (q | a) else 0.0


# ===========================================================================
# Aggregation and uncertainty
# ===========================================================================


def aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {k: statistics.fmean(r.get(k, 0.0) for r in rows) for k in rows[0]}


def group_by(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    """Aggregate question results by a diagnostic tag."""
    groups: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        groups.setdefault(row.get(key) or "unspecified", []).append(row["metrics"])
    return {label: aggregate(items) for label, items in sorted(groups.items())}


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42) -> Tuple[float, float, float]:
    """Percentile bootstrap CI, resampling *questions* with replacement.

    The question is the correct resampling unit: chunks within one question are
    not independent. With a golden set of this size a few points of difference can
    easily be noise, so reporting a point estimate alone would be misleading.
    """
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    point = statistics.fmean(values)
    means = sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return point, means[int((alpha / 2) * n_boot)], means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]


def paired_difference(a: Sequence[float], b: Sequence[float], n_boot: int = 2000,
                      alpha: float = 0.05, seed: int = 42) -> Dict[str, Any]:
    """Bootstrap CI for ``mean(a) - mean(b)`` on the same questions.

    Paired resampling removes between-question variance, which is what lets a
    small golden set still detect a real difference between two configurations.
    """
    if len(a) != len(b) or not a:
        raise ValueError("Paired comparison needs equal-length sequences.")
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    low = means[int((alpha / 2) * n_boot)]
    high = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    base = statistics.fmean(b)
    return {"difference": round(statistics.fmean(diffs), 4), "ci_low": round(low, 4),
            "ci_high": round(high, 4), "significant": bool(low > 0 or high < 0),
            "pct_change": round(100.0 * statistics.fmean(diffs) / base, 2) if base else 0.0}


# ===========================================================================
# Running an evaluation
# ===========================================================================


def load_golden(path: str | Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path if Path(path).is_absolute() else ROOT / path)
    if not rows:
        raise ValueError(f"Golden set {path} is empty.")
    return rows


def split_golden(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate answerable questions from those that should be refused."""
    answerable = [r for r in rows if r.get("expected", "answer") == "answer"]
    unanswerable = [r for r in rows if r.get("expected", "answer") == "refuse"]
    return answerable, unanswerable


def evaluate(pipeline, golden: Sequence[Dict[str, Any]], with_generation: bool = True,
             label: str = "default", mode: Optional[str] = None,
             top_k: Optional[int] = None, bootstrap: int = 2000,
             progress_every: int = 10) -> Dict[str, Any]:
    """Evaluate one configuration; returns a JSON-serialisable bundle."""
    started = time.perf_counter()
    config: Config = pipeline.config
    encode_fn = (lambda texts: pipeline.embedder.encode(list(texts))) if pipeline.embedder else None
    answerable, unanswerable = split_golden(golden)
    measure_k = max(top_k or config.top_k, config.top_k, 10)

    answer_rows: List[Dict[str, Any]] = []
    for i, item in enumerate(answerable, start=1):
        question, gold = item["question"], item.get("gold_doc_ids", [])
        hits = pipeline.retrieve(question, top_k=measure_k, mode=mode)
        row: Dict[str, Any] = {
            "qid": item.get("qid"), "question": question, "gold_doc_ids": gold,
            "stratum": item.get("stratum"), "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "retrieval_challenge": item.get("retrieval_challenge", "none"),
            "metrics": retrieval_metrics(hits, gold),
            "retrieved_doc_ids": [h.doc_id for h in hits[: measure_k]],
            "gold_ranks": [h.rank for h in hits if h.doc_id in set(gold)],
        }
        if with_generation:
            answer = pipeline.ask(question, top_k=top_k, mode=mode)
            text = "" if answer.refused else answer.text
            faith = faithfulness(text, answer.hits, encode_fn=encode_fn)
            row.update({
                "refused": bool(answer.refused),
                "answer_text": answer.text,
                "backend": answer.backend,
                "confidence": answer.confidence,
                "latency_ms": answer.latency_ms,
                "cited_doc_ids": [c.doc_id for c in answer.citations],
                "generation": {
                    "faithfulness": faith["faithfulness"],
                    "hallucination_rate": 1.0 - faith["faithfulness"],
                    "answer_correctness": answer_correctness(text, item.get("answer_points", []))["answer_correctness"],
                    "answer_relevancy": answer_relevancy(question, text, encode_fn),
                    "context_precision": context_precision(answer.hits, gold),
                    "context_recall": recall_at_k(dedupe_by_document(answer.hits), gold, 10 ** 6),
                    **citation_metrics(answer, gold),
                    "refused": 1.0 if answer.refused else 0.0,
                    "over_refusal": 1.0 if answer.refused else 0.0,
                },
                "unsupported_sentences": faith["unsupported"],
            })
        answer_rows.append(row)
        if progress_every and i % progress_every == 0:
            log.info("  [%s] %d/%d answerable", label, i, len(answerable))

    refuse_rows: List[Dict[str, Any]] = []
    for item in unanswerable:
        question = item["question"]
        hits = pipeline.retrieve(question, top_k=top_k or config.top_k, mode=mode)
        row = {"qid": item.get("qid"), "question": question, "stratum": item.get("stratum"),
               "note": item.get("note"), "signals": confidence_signals(hits),
               "top_docs": [h.doc_id for h in hits[:3]]}
        if with_generation:
            answer = pipeline.ask(question, top_k=top_k, mode=mode)
            row.update({"refused": bool(answer.refused), "answer_text": answer.text,
                        "refusal_reason": answer.refusal_reason, "confidence": answer.confidence})
        refuse_rows.append(row)

    retrieval_agg = aggregate([r["metrics"] for r in answer_rows])
    generation_agg = aggregate([r["generation"] for r in answer_rows]) if with_generation else {}
    headline = {}
    for metric in HEADLINE:
        point, low, high = bootstrap_ci([r["metrics"].get(metric, 0.0) for r in answer_rows],
                                        n_boot=bootstrap, seed=config.seed)
        headline[metric] = {"mean": round(point, 4), "ci_low": round(low, 4), "ci_high": round(high, 4)}

    refusal: Dict[str, Any] = {}
    if refuse_rows and with_generation:
        refused = sum(1 for r in refuse_rows if r.get("refused"))
        refusal = {
            "n": len(refuse_rows), "refused": refused,
            "refusal_accuracy": round(refused / len(refuse_rows), 4),
            "incorrectly_answered": [r["qid"] for r in refuse_rows if not r.get("refused")],
        }
    if with_generation and answer_rows:
        over = [r["qid"] for r in answer_rows if r.get("refused")]
        refusal["over_refusal"] = round(len(over) / len(answer_rows), 4)
        refusal["over_refused_qids"] = over

    latencies = [sum(r["latency_ms"].values()) for r in answer_rows if r.get("latency_ms")]
    result = {
        "label": label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.snapshot(),
        "overrides": {k: v for k, v in {"mode": mode, "top_k": top_k}.items() if v is not None},
        "backends": {
            "retriever": pipeline.retriever.describe(),
            "llm": f"{pipeline.llm.name} ({'generative' if pipeline.llm.is_generative else 'extractive-fallback'})"
                   if with_generation else None,
            "abstention": ("active@%.3f" % pipeline.gate.threshold) if pipeline.gate.active else "inactive",
        },
        "n_answerable": len(answer_rows), "n_unanswerable": len(refuse_rows),
        "with_generation": with_generation,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "mean_latency_ms": round(statistics.fmean(latencies), 1) if latencies else None,
        "retrieval": {k: round(v, 4) for k, v in retrieval_agg.items()},
        "headline_ci": headline,
        "generation": {k: round(v, 4) for k, v in generation_agg.items()},
        "refusal": refusal,
        "by_stratum": group_by(answer_rows, "stratum"),
        "by_category": group_by(answer_rows, "category"),
        "by_challenge": group_by(answer_rows, "retrieval_challenge"),
        "by_difficulty": group_by(answer_rows, "difficulty"),
        "answerable_rows": answer_rows,
        "unanswerable_rows": refuse_rows,
    }
    log.info("[%s] %.1fs - recall@5=%.4f mrr=%.4f ndcg@10=%.4f refusal=%s",
             label, result["elapsed_s"], retrieval_agg.get("recall@5", 0),
             retrieval_agg.get("mrr", 0), retrieval_agg.get("ndcg@10", 0),
             refusal.get("refusal_accuracy", "n/a"))
    return result


# ===========================================================================
# Abstention calibration
# ===========================================================================

SIGNALS = ("top_dense", "top_rerank", "mean_top3", "top_sparse")


def _best_threshold(positive: Sequence[float], negative: Sequence[float],
                    max_false_refusal: float, n_candidates: int = 400) -> Dict[str, Any]:
    """Threshold maximising balanced accuracy under a false-refusal budget.

    Balanced accuracy rather than plain accuracy because the two classes are
    deliberately unequal (33 answerable vs 12 unanswerable here); plain accuracy
    would let the trivial "never refuse" rule look competitive.
    """
    lo, hi = min(min(positive), min(negative)), max(max(positive), max(negative))
    if hi <= lo:
        hi = lo + 1e-6
    best = None
    curve = []
    for i in range(n_candidates):
        threshold = lo + (hi - lo) * i / (n_candidates - 1)
        true_refusal = sum(1 for s in negative if s < threshold) / len(negative)
        false_refusal = sum(1 for s in positive if s < threshold) / len(positive)
        balanced = (true_refusal + (1.0 - false_refusal)) / 2.0
        curve.append({"threshold": round(threshold, 5), "true_refusal_rate": round(true_refusal, 4),
                      "false_refusal_rate": round(false_refusal, 4), "balanced_accuracy": round(balanced, 4)})
        if false_refusal > max_false_refusal + 1e-9:
            continue
        if best is None or (balanced, threshold) > (best["balanced_accuracy"], best["threshold"]):
            best = curve[-1]
    if best is None:
        log.warning("No threshold satisfies max_false_refusal=%.2f; the score distributions "
                    "overlap too much for a reliable gate.", max_false_refusal)
        best = {"threshold": round(lo - 1.0, 5), "true_refusal_rate": 0.0,
                "false_refusal_rate": 0.0, "balanced_accuracy": 0.5, "degenerate": True}
    return {"threshold": best["threshold"], "balanced_accuracy": best["balanced_accuracy"],
            "true_refusal_rate": best["true_refusal_rate"],
            "false_refusal_rate": best["false_refusal_rate"], "curve": curve}


def calibrate_abstention(pipeline, golden: Sequence[Dict[str, Any]],
                         max_false_refusal: Optional[float] = None,
                         out_path: Optional[str | Path] = None,
                         write: bool = True) -> Dict[str, Any]:
    """Learn the abstention threshold from the golden sets.

    Retrieval-only (no generation), so it is cheap. Signal means/stds are fitted
    on the pooled sample so they can be z-scored and averaged into one confidence
    score; signals with no variance are dropped rather than allowed to contribute
    a constant offset.
    """
    budget = max_false_refusal if max_false_refusal is not None else pipeline.config.abstain_max_false_refusal
    answerable, unanswerable = split_golden(golden)
    if not answerable or not unanswerable:
        raise ValueError("Calibration needs both answerable and unanswerable golden questions.")

    def signals_for(items):
        out = []
        for item in items:
            hits = pipeline.retrieve(item["question"], top_k=pipeline.config.top_k)
            out.append({"qid": item.get("qid"), **confidence_signals(hits)})
        return out

    log.info("Collecting retrieval confidence: %d answerable, %d unanswerable ...",
             len(answerable), len(unanswerable))
    positive = signals_for(answerable)
    negative = signals_for(unanswerable)

    pooled = positive + negative
    stats: Dict[str, Dict[str, float]] = {}
    for name in SIGNALS:
        values = [r[name] for r in pooled]
        std = statistics.pstdev(values)
        if std > 1e-6:
            stats[name] = {"mean": statistics.fmean(values), "std": std}
        else:
            log.warning("Excluding zero-variance signal '%s' (inactive in this configuration).", name)
    if not stats:
        raise RuntimeError("No usable abstention signal; nothing to calibrate.")

    # ---- select the signal subset from evidence, not by hand ---------------
    # Enumerating subsets is cheap (2^n - 1 for n <= 4 signals) and avoids two
    # failure modes: hard-coding a signal that turns out to hurt, and silently
    # averaging in noise. The winner is chosen by balanced accuracy, tie-broken
    # toward fewer signals and then higher AUROC, so a signal only earns its
    # place if it demonstrably helps.
    available = [k for k in SIGNALS if k in stats]
    default_weights = AbstentionGate().weights
    per_signal_table: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for size in range(1, len(available) + 1):
        for subset in itertools.combinations(available, size):
            weights = {name: default_weights.get(name, 1.0) for name in subset}
            gate = AbstentionGate(enabled=True, weights=weights, stats=stats)
            pos_scores = [gate.score_from_signals(r) for r in positive]
            neg_scores = [gate.score_from_signals(r) for r in negative]
            outcome = _best_threshold(pos_scores, neg_scores, budget)
            sep = _separation(pos_scores, neg_scores)
            entry = {
                "signals": list(subset),
                "threshold": outcome["threshold"],
                "balanced_accuracy": outcome["balanced_accuracy"],
                "true_refusal_rate": outcome["true_refusal_rate"],
                "false_refusal_rate": outcome["false_refusal_rate"],
                "auroc": sep["auroc"],
                "cohens_d": sep["cohens_d"],
                "curve": outcome["curve"],
                "pos_scores": pos_scores,
                "neg_scores": neg_scores,
                "weights": weights,
            }
            per_signal_table.append({k: v for k, v in entry.items() if k not in ("curve", "pos_scores", "neg_scores")})
            key = (outcome["balanced_accuracy"], -len(subset), sep["auroc"])
            if best is None or key > (best["balanced_accuracy"], -len(best["signals"]), best["auroc"]):
                best = entry

    if best is None:
        raise RuntimeError("No signal subset could be calibrated.")
    result = {k: best[k] for k in ("threshold", "balanced_accuracy", "true_refusal_rate", "false_refusal_rate")}
    result["curve"] = best["curve"]
    sep = {"auroc": best["auroc"], "cohens_d": best["cohens_d"],
           "mean_answerable": statistics.fmean(best["pos_scores"]),
           "mean_unanswerable": statistics.fmean(best["neg_scores"])}
    weights, stats_used = best["weights"], {k: stats[k] for k in best["signals"]}
    pos_scores, neg_scores = best["pos_scores"], best["neg_scores"]
    log.info("Selected abstention signals %s (balanced accuracy %.4f, AUROC %.4f) out of %d subsets tried.",
             best["signals"], best["balanced_accuracy"], best["auroc"], len(per_signal_table))

    payload = {
        "threshold": result["threshold"],
        "signals": best["signals"],
        "weights": weights,
        "stats": stats_used,
        "signal_selection": sorted(per_signal_table,
                                   key=lambda e: (-e["balanced_accuracy"], len(e["signals"]))),
        "calibrated_on": f"{len(positive)} answerable / {len(negative)} unanswerable golden questions",
        "calibrated_for_embedder": pipeline.embedder.name if pipeline.embedder else "",
        "retriever_mode": pipeline.retriever.describe(),
        "max_false_refusal_budget": budget,
        "metrics": {
            "balanced_accuracy": result["balanced_accuracy"],
            "true_refusal_rate": result["true_refusal_rate"],
            "false_refusal_rate": result["false_refusal_rate"],
            **sep,
        },
        "curve": result["curve"],
        "answerable_scores": [{"qid": r["qid"], "score": round(s, 5)} for r, s in zip(positive, pos_scores)],
        "unanswerable_scores": [{"qid": r["qid"], "score": round(s, 5)} for r, s in zip(negative, neg_scores)],
        "raw_signals": {"answerable": positive, "unanswerable": negative},
    }

    if write:
        target = Path(out_path or pipeline.config.abstain_path)
        if not target.is_absolute():
            target = ROOT / target
        # The gate file is *configuration*: it must stay small and contain only
        # what AbstentionGate.load needs. The audit trail (calibration curve,
        # every subset tried, per-question raw signals) belongs in reports/, which
        # the caller writes.
        write_json(target, {
            "threshold": payload["threshold"],
            "signals": payload["signals"],
            "weights": payload["weights"],
            "stats": payload["stats"],
            "calibrated_on": payload["calibrated_on"],
            "calibrated_for_embedder": payload["calibrated_for_embedder"],
            "retriever_mode": payload["retriever_mode"],
            "metrics": payload["metrics"],
            "max_false_refusal_budget": payload["max_false_refusal_budget"],
        })
        log.info("Abstention gate written -> %s (threshold=%.4f, balanced accuracy=%.3f, AUROC=%.3f)",
                 target, payload["threshold"], result["balanced_accuracy"], sep["auroc"])
    return payload


def _separation(positive: Sequence[float], negative: Sequence[float]) -> Dict[str, float]:
    """How well the two score distributions separate.

    Reports Cohen's d and an AUROC-equivalent (Mann-Whitney U / n1n2): the
    probability that a random answerable question outscores a random unanswerable
    one. An AUROC near 0.5 means there is no signal to gate on, and the honest
    response is to say so rather than ship a threshold that only looks calibrated.
    """
    mp, sp = statistics.fmean(positive), statistics.pstdev(positive)
    mn, sn = statistics.fmean(negative), statistics.pstdev(negative)
    pooled = ((sp ** 2 + sn ** 2) / 2) ** 0.5
    wins = ties = 0
    for p in positive:
        for n in negative:
            wins += p > n
            ties += p == n
    denom = len(positive) * len(negative)
    return {"mean_answerable": round(mp, 4), "mean_unanswerable": round(mn, 4),
            "cohens_d": round((mp - mn) / pooled, 4) if pooled else 0.0,
            "auroc": round((wins + 0.5 * ties) / denom, 4) if denom else 0.0}


# ===========================================================================
# Configuration comparison
# ===========================================================================

COMPARISONS: List[Dict[str, Any]] = [
    {"label": "dense_only", "mode": "dense",
     "why": "The baseline every RAG tutorial ships: vector similarity alone."},
    {"label": "bm25_only", "mode": "sparse",
     "why": "Exact lexical matching alone: strong on terminology, blind to paraphrase."},
    {"label": "hybrid_rrf", "mode": "hybrid",
     "why": "BM25 + dense fused by reciprocal rank (equal rank weight)."},
    {"label": "hybrid_weighted@0.2", "mode": "weighted", "dense_weight": 0.2,
     "why": "Normalised score weighting at the measured-optimal dense weight."},
    {"label": "hybrid_weighted@0.5", "mode": "weighted", "dense_weight": 0.5,
     "why": "Even score weighting - shows what over-weighting the weaker channel costs."},
]


def compare(pipeline, golden: Sequence[Dict[str, Any]],
            variants: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Run the same golden set through several retrieval configurations.

    Reranking is disabled for these arms on purpose: the comparison is about the
    *first stage*, and leaving the reranker on would let it mask the differences
    between fusion modes.
    """
    variants = variants or COMPARISONS
    saved_flag = pipeline.config.rerank
    saved_reranker = pipeline.retriever.reranker
    saved_weight = pipeline.config.dense_weight
    results = []
    try:
        pipeline.config.rerank = False
        pipeline.retriever.reranker = None
        for variant in variants:
            if "dense_weight" in variant:
                pipeline.config.dense_weight = variant["dense_weight"]
            log.info("Comparison arm: %s", variant["label"])
            results.append(evaluate(pipeline, golden, with_generation=False,
                                    label=variant["label"], mode=variant.get("mode")))
            results[-1]["why"] = variant.get("why", "")
            results[-1]["dense_weight"] = variant.get("dense_weight", pipeline.config.dense_weight)
    finally:
        # Restore BOTH the flag and the object. Leaving the reranker detached
        # silently changes every later result - including the abstention gate's
        # confidence signal, which reads the cross-encoder score.
        pipeline.config.rerank = saved_flag
        pipeline.retriever.reranker = saved_reranker
        pipeline.config.dense_weight = saved_weight
    return results


def sweep_fusion_weights(pipeline, golden: Sequence[Dict[str, Any]],
                         weights: Sequence[float] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
                         include_rrf: bool = True) -> List[Dict[str, Any]]:
    """Measure retrieval quality as a function of the dense/sparse fusion weight.

    This exists because "hybrid beats either channel alone" is an assumption, not
    a fact. On a small, terminology-dense corpus BM25 can be the stronger channel,
    and fusing it 50/50 with a weaker dense channel *dilutes* it. Sweeping the
    weight turns the choice of default into a measurement.

    Reranking is disabled for the sweep so the first stage is what varies.
    """
    saved_flag, saved_reranker = pipeline.config.rerank, pipeline.retriever.reranker
    saved_weight, saved_mode = pipeline.config.dense_weight, pipeline.config.mode
    rows: List[Dict[str, Any]] = []
    try:
        pipeline.config.rerank = False
        pipeline.retriever.reranker = None
        pipeline.config.mode = "weighted"
        for weight in weights:
            pipeline.config.dense_weight = weight
            label = {0.0: "bm25_only", 1.0: "dense_only"}.get(weight, f"dense_weight={weight:.1f}")
            result = evaluate(pipeline, golden, with_generation=False, label=label,
                              progress_every=0)
            result["dense_weight"] = weight
            rows.append(result)
        if include_rrf:
            pipeline.config.mode = "hybrid"
            result = evaluate(pipeline, golden, with_generation=False, label="rrf", progress_every=0)
            result["dense_weight"] = None
            rows.append(result)
    finally:
        pipeline.config.rerank = saved_flag
        pipeline.retriever.reranker = saved_reranker
        pipeline.config.dense_weight = saved_weight
        pipeline.config.mode = saved_mode
    return rows


# ===========================================================================
# Reporting
# ===========================================================================


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
                   first_left: bool = True) -> str:
    aligns = ["left"] if first_left else ["right"]
    aligns += ["right"] * (len(headers) - len(aligns))
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join(":---" if a == "left" else "---:" for a in aligns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if value != value else f"{value:.{digits}f}"
    return str(value)


def comparison_table(results: Sequence[Dict[str, Any]],
                     metrics: Sequence[str] = ("recall@1", "recall@5", "recall@10", "mrr",
                                                "ndcg@10", "map")) -> str:
    rows = [[f"`{r['label']}`"] + [fmt(r["retrieval"].get(m)) for m in metrics] for r in results]
    return markdown_table(["configuration", *metrics], rows)


def slice_table(results: Sequence[Dict[str, Any]], key: str, metric: str) -> str:
    labels = sorted({name for r in results for name in (r.get(key) or {})})
    if not labels:
        return "_(no slices recorded)_"
    rows = [[f"`{r['label']}`"] + [fmt((r.get(key) or {}).get(n, {}).get(metric)) for n in labels]
            for r in results]
    return markdown_table(["configuration", *labels], rows)
