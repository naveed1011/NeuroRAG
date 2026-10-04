#!/usr/bin/env python3
"""Build and validate the golden evaluation set.

Why this script exists
----------------------
A RAG benchmark is only as trustworthy as its gold labels. Auto-generating a
question *from* a document's own abstract and then scoring whether that document
is retrieved measures nothing but vocabulary overlap. So every question here is
hand-written, and each gold label points at a document whose text was read before
the label was assigned.

Each item carries ``expected``:
  * ``answer``  - the corpus supports an answer; ``gold_doc_ids`` names the papers
  * ``refuse``  - the corpus does NOT support an answer; the correct behaviour is
                  to abstain

Diagnostic tags
---------------
``retrieval_challenge`` records *why* a question is hard, so the report can
attribute a hybrid-retrieval gain to a mechanism instead of just reporting a
number:
  * ``semantic`` - the question deliberately avoids the source's vocabulary
                   ("brain shrinkage" for a paper that says "atrophy"). Only a
                   dense encoder can bridge that gap.
  * ``lexical``  - the question hinges on an exact rare token (a dataset name, an
                   acronym, a metric). Only BM25 reliably matches it.
  * ``both``     - needs terminology and conceptual bridging.

Validation
----------
1. every ``gold_doc_id`` must exist in the corpus snapshot
2. questions must be unique and must not appear verbatim in the corpus (a
   verbatim question measures nothing)
3. ``refuse`` items must be provably unanswerable, via either
   ``absent_terms``  - the string occurs nowhere in the corpus, or
   ``must_not_cooccur`` - no single document contains all the listed terms
   The second form is necessary for numeric questions: "75.5" can occur
   coincidentally as an unrelated cohort percentage while the specific claim
   ("model X reported 75.5% accuracy") is still absent.

Run it with no arguments; it refuses to write if validation fails.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neurorag.utils import get_logger, read_jsonl, write_jsonl  # noqa: E402

log = get_logger("neurorag.golden")

CATEGORIES = {"factual", "methodological", "comparative", "numeric", "multi_hop",
              "bibliographic", "definitional", "own_work"}
CHALLENGES = {"semantic", "lexical", "both", "none"}
STRATA = {"clinical", "methods", "rag"}
DIFFICULTY = {"easy", "medium", "hard"}

# ===========================================================================
# ANSWERABLE QUESTIONS
# ===========================================================================
ANSWERABLE = [
    # ---- RAG mechanisms ---------------------------------------------------
    dict(qid="rag-001",
         question="What two kinds of memory does a retrieval-augmented generation model combine?",
         gold_doc_ids=["arxiv:2005.11401"],
         answer_points=["parametric", "non-parametric", "memory"],
         category="definitional", stratum="rag", difficulty="easy", retrieval_challenge="semantic",
         note="Primary RAG source. Avoids the phrase 'parametric memory' so a lexical-only retriever has less to grip."),
    dict(qid="rag-002",
         question="By how much did dense passage retrieval outperform BM25 on top-20 passage retrieval accuracy?",
         gold_doc_ids=["arxiv:2004.04906"],
         answer_points=["9", "19", "absolute", "top-20", "BM25"],
         category="numeric", stratum="rag", difficulty="medium", retrieval_challenge="lexical",
         note="Tests that numeric results survive chunking and that exact terms are preserved."),
    dict(qid="rag-003",
         question="Which metrics does the RAGAS framework propose for evaluating retrieval-augmented generation without ground-truth annotations?",
         gold_doc_ids=["arxiv:2309.15217"],
         answer_points=["faithfulness", "answer relevance", "context", "reference-free"],
         category="methodological", stratum="rag", difficulty="medium", retrieval_challenge="lexical",
         note="Directly relevant to this repository's own evaluation design."),
    dict(qid="rag-004",
         question="How does HyDE perform zero-shot dense retrieval when no relevance labels are available?",
         gold_doc_ids=["arxiv:2212.10496"],
         answer_points=["hypothetical document", "language model", "encoder", "embedding"],
         category="methodological", stratum="rag", difficulty="hard", retrieval_challenge="both",
         note="Uses only the acronym; the source spells out 'Hypothetical Document Embeddings'."),
    dict(qid="rag-005",
         question="Where in a long input context do language models use relevant information least reliably?",
         gold_doc_ids=["arxiv:2307.03172"],
         answer_points=["middle", "beginning", "end", "position", "degrade"],
         category="factual", stratum="rag", difficulty="medium", retrieval_challenge="semantic",
         note="Motivates keeping top_k small and putting the strongest passages at the edges."),
    dict(qid="rag-006",
         question="How does SelfCheckGPT detect hallucinations without access to an external knowledge base?",
         gold_doc_ids=["arxiv:2303.08896"],
         answer_points=["sampling", "consistency", "black-box", "zero-resource"],
         category="methodological", stratum="rag", difficulty="hard", retrieval_challenge="semantic",
         note="'without an external knowledge base' paraphrases 'zero-resource'."),
    dict(qid="rag-007",
         question="Why is a bi-encoder with cosine similarity preferred over cross-encoder BERT for searching a large passage collection?",
         gold_doc_ids=["arxiv:1908.10084"],
         answer_points=["siamese", "cosine", "embeddings", "computational"],
         category="comparative", stratum="rag", difficulty="medium", retrieval_challenge="both",
         note="The bi-encoder/cross-encoder trade-off that justifies retrieve-then-rerank."),
    dict(qid="rag-008",
         question="What is reciprocal rank fusion and which paper introduced it?",
         gold_doc_ids=["doi:10.1145/1571941.1572114"],
         answer_points=["Cormack", "Reciprocal rank fusion", "rank"],
         category="bibliographic", stratum="rag", difficulty="easy", retrieval_challenge="lexical",
         note="This record is a metadata-only stub, so it tests bibliographic retrieval."),
    dict(qid="rag-009",
         question="How does ColBERT differ from a single-vector dense retriever?",
         gold_doc_ids=["arxiv:2004.12832"],
         answer_points=["late interaction", "contextualized", "token"],
         category="comparative", stratum="rag", difficulty="hard", retrieval_challenge="lexical",
         note="Architecture-name query; only exact matching finds it."),
    dict(qid="rag-010",
         question="What does the Fusion-in-Decoder architecture do with multiple retrieved passages?",
         gold_doc_ids=["arxiv:2007.01282"],
         answer_points=["passages", "encoder", "decoder", "combine"],
         category="methodological", stratum="rag", difficulty="medium", retrieval_challenge="both",
         note="Known-hard: the paper's abstract never uses the phrase 'Fusion-in-Decoder'."),

    # ---- Methods / architectures ------------------------------------------
    dict(qid="mth-001",
         question="How does a Vision Transformer process an image without using convolutions?",
         gold_doc_ids=["arxiv:2010.11929"],
         answer_points=["patches", "sequence", "transformer"],
         category="methodological", stratum="methods", difficulty="easy", retrieval_challenge="semantic",
         note="'without using convolutions' paraphrases 'this reliance on CNNs is not necessary'."),
    dict(qid="mth-002",
         question="What is Swin UNETR and what kind of pre-training data was it demonstrated on?",
         gold_doc_ids=["arxiv:2111.14791"],
         answer_points=["Swin UNETR", "self-supervised", "3D", "computed tomography", "hierarchical encoder"],
         category="methodological", stratum="methods", difficulty="medium", retrieval_challenge="lexical",
         note="Primary architecture reference for the author's segmentation work."),
    dict(qid="mth-003",
         question="Why does MONAI exist as a separate framework rather than using plain PyTorch for medical imaging?",
         gold_doc_ids=["arxiv:2211.02701"],
         answer_points=["medical", "PyTorch", "geometry", "physiology", "imaging"],
         category="methodological", stratum="methods", difficulty="medium", retrieval_challenge="semantic",
         note="Tests retrieval of rationale statements, not just name matches."),
    dict(qid="mth-004",
         question="How does the 3D U-Net learn dense volumetric segmentation from only a few annotated slices?",
         gold_doc_ids=["arxiv:1606.06650"],
         answer_points=["sparse", "annotation", "volumetric", "elastic deformation"],
         category="methodological", stratum="methods", difficulty="medium", retrieval_challenge="both"),
    dict(qid="mth-005",
         question="What does nnU-Net self-configure, and on which challenge was it evaluated?",
         gold_doc_ids=["arxiv:1809.10486"],
         answer_points=["preprocessing", "architecture", "training", "Medical Segmentation Decathlon"],
         category="methodological", stratum="methods", difficulty="medium", retrieval_challenge="lexical",
         note="Dataset-name query; exact-match retrieval matters."),
    dict(qid="mth-006",
         question="How does Grad-CAM produce a visual explanation from a convolutional network?",
         gold_doc_ids=["arxiv:1610.02391"],
         answer_points=["gradients", "final convolutional layer", "localization map"],
         category="methodological", stratum="methods", difficulty="easy", retrieval_challenge="semantic"),
    dict(qid="mth-007",
         question="What problem do shifted windows solve in the Swin Transformer?",
         gold_doc_ids=["arxiv:2103.14030"],
         answer_points=["window", "shifted", "hierarchical", "cross-window"],
         category="methodological", stratum="methods", difficulty="hard", retrieval_challenge="both"),
    dict(qid="mth-008",
         question="Why is subject-level data splitting important when evaluating deep learning on brain MRI?",
         gold_doc_ids=["epmc:pmc13085064", "manual:rahim2025-mci-ad-multiview"],
         answer_points=["subject", "leakage", "splitting", "longitudinal"],
         category="methodological", stratum="methods", difficulty="hard", retrieval_challenge="semantic",
         note="Multi-source: supported by a methods paper and the author's own publication. "
              "Mirrors the GroupKFold subject-level validation used in AD-TriFuseViT."),
    dict(qid="mth-009",
         question="Which datasets were used to train a streamlined CNN pipeline for distinguishing Alzheimer's, MCI and healthy controls?",
         gold_doc_ids=["epmc:42400646"],
         answer_points=["OASIS", "ADNI", "CNN", "mild cognitive impairment"],
         category="factual", stratum="methods", difficulty="easy", retrieval_challenge="lexical"),
    dict(qid="mth-010",
         question="What volumetric features were extracted with the CAT12 toolbox for time-distributed Alzheimer's classification?",
         gold_doc_ids=["epmc:42240570"],
         answer_points=["cortical thickness", "white matter", "grey matter", "cerebrospinal fluid", "CAT12"],
         category="numeric", stratum="methods", difficulty="hard", retrieval_challenge="lexical",
         note="Tool-name query that a semantic encoder alone is unlikely to surface."),
    dict(qid="mth-011",
         question="How does a multi-modal fusion framework combine several MRI sequences for brain tumour segmentation?",
         gold_doc_ids=["epmc:10.21203/rs.3.rs-7112498/v1"],
         answer_points=["fusion", "multi-modal", "3D", "segmentation"],
         category="methodological", stratum="methods", difficulty="medium", retrieval_challenge="semantic",
         note="Conceptually parallel to the author's Modality Attention Fusion module."),

    # ---- Clinical ---------------------------------------------------------
    dict(qid="cli-001",
         question="What structural brain change is described as a key sign of Alzheimer's disease in MRI-based classification work?",
         gold_doc_ids=["epmc:41830915"],
         answer_points=["hippocampus", "shrinkage", "frontal lobe", "magnetic resonance"],
         category="factual", stratum="clinical", difficulty="easy", retrieval_challenge="semantic",
         note="Question says 'structural brain change'; the source says 'shrinkage of the hippocampus'."),
    dict(qid="cli-002",
         question="How does the number of afflicted biomarker classes relate to Clinical Dementia Rating sum-of-boxes scores?",
         gold_doc_ids=["epmc:pmc13392201"],
         answer_points=["CDR-SB", "biomarker", "affliction", "ADNI"],
         category="factual", stratum="clinical", difficulty="hard", retrieval_challenge="lexical",
         note="'Clinical Dementia Rating sum-of-boxes' maps to 'CDR-SB' only via acronym expansion."),
    dict(qid="cli-003",
         question="What is the NIA-AA ATN framework used for in Alzheimer's research cohorts?",
         gold_doc_ids=["epmc:pmc13281289"],
         answer_points=["ATN", "amyloid", "tau", "neurodegeneration", "biological classification"],
         category="definitional", stratum="clinical", difficulty="medium", retrieval_challenge="lexical"),
    dict(qid="cli-004",
         question="Why is late fusion of independently extracted per-view features considered a limitation in multi-view MRI diagnosis?",
         gold_doc_ids=["epmc:41805501"],
         answer_points=["late fusion", "spatial information", "anatomical correspondence", "axial", "coronal"],
         category="methodological", stratum="clinical", difficulty="hard", retrieval_challenge="both",
         note="The exact argument motivating learned three-branch fusion in AD-TriFuseViT."),
    dict(qid="cli-005",
         question="Once feature-selection leakage is controlled, what does the multi-compartment benchmark say about predicting MCI-to-AD conversion?",
         gold_doc_ids=["epmc:10.21203/rs.3.rs-10580655/v1"],
         answer_points=["leakage", "plasma", "cerebrospinal fluid", "ADNI", "conversion"],
         category="methodological", stratum="clinical", difficulty="hard", retrieval_challenge="both"),
    dict(qid="cli-006",
         question="How does the mesial temporal atrophy visual rating scale relate to hippocampal volumetry?",
         gold_doc_ids=["epmc:10.21203/rs.3.rs-7953365/v1"],
         answer_points=["mesial temporal atrophy", "hippocampal volumetry", "correlation"],
         category="factual", stratum="clinical", difficulty="hard", retrieval_challenge="lexical",
         note="Rare anatomical terminology; dense embeddings alone tend to blur these."),
    dict(qid="cli-007",
         question="What did the author's own published work on multi-view MRI aim to detect, and in what care setting?",
         gold_doc_ids=["manual:rahim2025-mci-ad-multiview"],
         answer_points=["mild cognitive impairment", "Alzheimer", "multi-view", "assisted living", "progression"],
         category="own_work", stratum="clinical", difficulty="easy", retrieval_challenge="none",
         note="Proves the corpus can answer questions about the repository owner's own Q1 publication."),
    dict(qid="cli-008",
         question="Which three orthogonal MRI acquisition planes are used for multi-view analysis?",
         gold_doc_ids=["epmc:41805501", "manual:rahim2025-mci-ad-multiview"],
         answer_points=["axial", "coronal", "sagittal"],
         category="factual", stratum="clinical", difficulty="easy", retrieval_challenge="lexical",
         note="Short factual query; tests precision at rank 1."),

    # ---- Cross-stratum multi-hop ------------------------------------------
    dict(qid="hop-001",
         question="What evaluation risk arises when the same patient contributes multiple slices to both training and validation sets?",
         gold_doc_ids=["epmc:pmc13085064", "epmc:10.21203/rs.3.rs-10580655/v1",
                       "manual:rahim2025-mci-ad-multiview"],
         answer_points=["leakage", "subject-level", "splitting"],
         category="multi_hop", stratum="methods", difficulty="hard", retrieval_challenge="semantic",
         note="Never says 'leakage'; it describes the mechanism. Three supporting documents."),
    dict(qid="hop-002",
         question="How can a retrieval-augmented system reduce hallucinated claims in generated medical answers?",
         gold_doc_ids=["arxiv:2309.15217", "arxiv:2303.08896", "arxiv:2005.11401"],
         answer_points=["context", "faithfulness", "retrieval", "grounding"],
         category="multi_hop", stratum="rag", difficulty="hard", retrieval_challenge="semantic",
         note="Requires synthesising across the RAG, evaluation and hallucination literatures."),
    dict(qid="hop-003",
         question="Why might a hybrid sparse-plus-dense retriever beat either channel alone on biomedical literature search?",
         gold_doc_ids=["arxiv:2004.04906", "doi:10.1145/1571941.1572114"],
         answer_points=["BM25", "dense", "rank", "fusion", "sparse"],
         category="comparative", stratum="rag", difficulty="hard", retrieval_challenge="both",
         note="The central design claim of this repository, checked against primary sources."),
    dict(qid="hop-004",
         question="What makes transformer-based encoders suitable for both 3D medical segmentation and passage retrieval?",
         gold_doc_ids=["arxiv:2111.14791", "arxiv:1706.03762"],
         answer_points=["attention", "hierarchical", "representation"],
         category="comparative", stratum="methods", difficulty="hard", retrieval_challenge="semantic",
         note="Deliberately cross-domain; tests bridging the clinical and RAG strata."),
]

# ===========================================================================
# QUESTIONS THAT MUST BE REFUSED
#
# The correct behaviour for every one of these is abstention.
# ===========================================================================
UNANSWERABLE = [
    dict(qid="ref-001",
         question="What patient-level accuracy does AD-TriFuseViT report on the OASIS-1 dataset for 4-class CDR staging?",
         absent_terms=["AD-TriFuseViT achieves"],
         must_not_cooccur=[["AD-TriFuseViT", "accuracy"], ["AD-TriFuseViT", "OASIS-1"]],
         stratum="clinical", category="numeric",
         note="The hardest case: the framework IS mentioned in the author-supplied record, but no "
              "accuracy figure exists anywhere in the corpus. Near-miss context invites a confident "
              "fabricated number."),
    dict(qid="ref-002",
         question="What Dice score did the TransUNet-3D model achieve on the BraTS 2024 leaderboard?",
         absent_terms=["TransUNet-3D", "BraTS 2024 leaderboard"],
         stratum="methods", category="numeric",
         note="Plausible model name and benchmark, neither present."),
    dict(qid="ref-003",
         question="Which liquid-cooling configuration was used in the datacentre that trained Llama-4?",
         absent_terms=["Llama-4", "liquid cooling", "datacentre"],
         stratum="rag", category="factual",
         note="Out of domain. An earlier draft asked about GPT-5 and the validator rejected it: GPT-5 "
              "genuinely appears in two corpus papers, so that question was not safely unanswerable."),
    dict(qid="ref-004",
         question="What is the recommended paediatric dose of lecanemab in milligrams per kilogram?",
         absent_terms=["paediatric dose", "pediatric dose", "mg/kg"],
         stratum="clinical", category="numeric",
         note="Clinically dangerous if hallucinated - exactly the case abstention exists for."),
    dict(qid="ref-005",
         question="Who won the 2024 Kaggle Alzheimer's MRI classification competition and what was their solution?",
         absent_terms=["Kaggle competition"],
         stratum="clinical", category="factual",
         note="Out of scope despite matching several corpus keywords."),
    dict(qid="ref-006",
         question="What is the exact p-value reported by Kim et al. (2019) for hippocampal subfield CA1 atrophy?",
         absent_terms=["Kim et al. (2019)", "subfield CA1"],
         stratum="clinical", category="numeric",
         note="Fabricated citation; a good system must not invent a p-value."),
    dict(qid="ref-007",
         question="How many parameters does the NeuroRAG-2 embedding model have?",
         absent_terms=["NeuroRAG-2"],
         stratum="rag", category="factual",
         note="Refers to a successor system that does not exist."),
    dict(qid="ref-008",
         question="What is the capital city of Mongolia?",
         absent_terms=["Mongolia"],
         stratum="rag", category="factual",
         note="Control case: trivially out of domain, should always be refused."),
    dict(qid="ref-009",
         question="Which hyperparameter values did the Swin UNETR authors use for BraTS whole-tumour class weighting?",
         absent_terms=["class weighting"],
         must_not_cooccur=[["Swin UNETR", "class weighting"]],
         stratum="methods", category="numeric",
         note="Near-miss: Swin UNETR is in the corpus but this implementation detail is not."),
    dict(qid="ref-010",
         question="What was the acceptance rate of the SIGIR 2009 reciprocal rank fusion paper?",
         absent_terms=["acceptance rate"],
         stratum="rag", category="factual",
         note="The paper IS in the corpus (as a metadata stub), but acceptance rates are not. Tests that "
              "the presence of a document does not license invention about it."),
    dict(qid="ref-011",
         question="Which FDA approval date applies to the NeuroRAG software as a medical device?",
         absent_terms=["FDA approval"],
         stratum="rag", category="factual",
         note="A regulatory claim about this repository itself; must be refused."),
    dict(qid="ref-012",
         question="What sensitivity and specificity does the MIRIAD dataset report for the ConvNeXt-7T benchmark?",
         absent_terms=["MIRIAD", "ConvNeXt-7T"],
         stratum="clinical", category="numeric",
         note="A real-sounding dataset and a plausible model name, neither present. An earlier draft used "
              "OASIS-3; the validator rejected it because OASIS-3 appears in four corpus documents."),
]


# ===========================================================================
# Validation
# ===========================================================================


def validate(items: list[dict], corpus_rows: list[dict]) -> list[str]:
    errors: list[str] = []
    corpus_ids = {r["doc_id"] for r in corpus_rows}
    docs_lower = [f"{r.get('title','')} {r.get('text','')}".lower() for r in corpus_rows]
    corpus_text = "\n".join(docs_lower)

    seen_qids: set[str] = set()
    seen_questions: set[str] = set()

    for item in items:
        qid = item.get("qid", "<no qid>")
        if qid in seen_qids:
            errors.append(f"{qid}: duplicate qid")
        seen_qids.add(qid)

        question = (item.get("question") or "").strip()
        if not question:
            errors.append(f"{qid}: empty question")
        if question.lower() in seen_questions:
            errors.append(f"{qid}: duplicate question text")
        seen_questions.add(question.lower())

        if item.get("stratum") not in STRATA:
            errors.append(f"{qid}: invalid stratum {item.get('stratum')!r}")
        if item.get("category") not in CATEGORIES:
            errors.append(f"{qid}: invalid category {item.get('category')!r}")

        if question.lower() in corpus_text:
            errors.append(f"{qid}: question appears verbatim in the corpus (leaky - measures nothing)")

        if item["expected"] == "answer":
            gold = item.get("gold_doc_ids") or []
            if not gold:
                errors.append(f"{qid}: no gold_doc_ids")
            for doc_id in gold:
                if doc_id not in corpus_ids:
                    errors.append(f"{qid}: gold_doc_id not in corpus -> {doc_id}")
            if not item.get("answer_points"):
                errors.append(f"{qid}: no answer_points (needed for answer-correctness scoring)")
            if item.get("retrieval_challenge") not in CHALLENGES:
                errors.append(f"{qid}: invalid retrieval_challenge {item.get('retrieval_challenge')!r}")
            if item.get("difficulty") not in DIFFICULTY:
                errors.append(f"{qid}: invalid difficulty {item.get('difficulty')!r}")
        else:
            absent = item.get("absent_terms") or []
            cooccur = item.get("must_not_cooccur") or []
            if not absent and not cooccur:
                errors.append(f"{qid}: needs absent_terms or must_not_cooccur to prove unanswerability")
            for term in absent:
                if term.lower() in corpus_text:
                    errors.append(f"{qid}: absent_term {term!r} IS present in the corpus - may be answerable")
            for group in cooccur:
                lowered = [g.lower() for g in group]
                hits = [i for i, doc in enumerate(docs_lower) if all(g in doc for g in lowered)]
                if hits:
                    errors.append(f"{qid}: {group} co-occur in {len(hits)} doc(s) "
                                  f"(e.g. {corpus_rows[hits[0]]['doc_id']}) - may be answerable")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/validate the golden evaluation set.")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus" / "corpus.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "golden" / "golden.jsonl")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    corpus_rows = read_jsonl(args.corpus)
    items = ([dict(i, expected="answer") for i in ANSWERABLE]
             + [dict(i, expected="refuse", difficulty=None, retrieval_challenge=None) for i in UNANSWERABLE])

    if not args.skip_validation:
        errors = validate(items, corpus_rows)
        if errors:
            print("GOLDEN SET VALIDATION FAILED:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        log.info("Validation passed: %d answerable + %d refuse items; all gold ids exist, "
                 "no verbatim leaks, all refuse items proven unanswerable.",
                 len(ANSWERABLE), len(UNANSWERABLE))

    n = write_jsonl(args.out, items)
    print(f"\nWrote {n} golden items -> {args.out}\n")
    print("Composition:")
    print("  expected  : " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(i['expected'] for i in items).items())))
    print("  stratum   : " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(i['stratum'] for i in items).items())))
    print("  category  : " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(i['category'] for i in items).items())))
    print("  challenge : " + ", ".join(f"{k}={v}" for k, v in sorted(
        Counter(i['retrieval_challenge'] for i in items if i.get('retrieval_challenge')).items())))
    print("  difficulty: " + ", ".join(f"{k}={v}" for k, v in sorted(
        Counter(i['difficulty'] for i in items if i.get('difficulty')).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
