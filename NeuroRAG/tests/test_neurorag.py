"""Test suite for NeuroRAG.

Runs entirely offline: it uses the lexical encoder and the extractive answer
backend, so ``pytest`` works on a machine with no ML dependencies, no GPU and no
API key. That is deliberate - a test suite that needs a model download is a test
suite that does not get run.

Two of these are regression tests for bugs found while building the evaluation:

* ``test_ndcg_never_exceeds_one`` - when several chunks from one gold document
  land in the top-k, DCG accumulated that document's gain repeatedly while IDCG
  counted it once, so nDCG exceeded 1.0.
* ``test_weighted_fusion_preserves_raw_dense_score`` - the fusion detail dict
  stored the per-query min-max value under ``dense``, which the abstention gate
  then compared against a threshold calibrated on cosine similarity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neurorag.chunking import chunk_documents, chunk_stats, recursive_split  # noqa: E402
from neurorag.config import Config, parse_overrides  # noqa: E402
from neurorag.corpus import clean_markup, parse_sections  # noqa: E402
from neurorag.evaluate import (  # noqa: E402
    answer_correctness,
    average_precision,
    dedupe_by_document,
    faithfulness,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
)
from neurorag.generator import (  # noqa: E402
    AbstentionGate,
    ExtractiveLLM,
    build_system_prompt,
    build_user_prompt,
    confidence_signals,
    parse_markers,
    refusal_detected,
    resolve_citations,
)
from neurorag.retriever import (  # noqa: E402
    BM25Index,
    Tokenizer,
    cap_per_document,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from neurorag.schema import Chunk, Document, Hit  # noqa: E402
from neurorag.utils import normalize_whitespace, split_sentences, truncate  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(chunk_id: str, text: str, doc_id: str = "doc1", title: str = "T",
               section: str = "Abstract", year: int = 2024) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, index=0, section=section,
                 header=f"{title} > {section}", doc_title=title, doc_year=year,
                 doc_stratum="methods", doc_url="http://x", doc_authors=["A B"])


def make_hit(rank: int, chunk: Chunk, score: float = 1.0, **scores) -> Hit:
    return Hit(chunk=chunk, score=score, rank=rank, scores=scores or {"dense": score})


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def test_normalize_whitespace_collapses_but_keeps_paragraphs():
    assert normalize_whitespace("a   b\n\n\n\nc") == "a b\n\nc"


def test_split_sentences_protects_abbreviations_and_decimals():
    text = "We used e.g. FreeSurfer. Hippocampal volume fell (p < 0.05). See Fig. 3 next."
    parts = split_sentences(text)
    assert len(parts) == 3
    assert "e.g." in parts[0]
    assert "0.05" in parts[1]
    assert "Fig. 3" in parts[2]


def test_truncate_marks_elision():
    assert truncate("abcdefghij", 9) == "abc [...]"
    assert truncate("abc", 60) == "abc"
    assert len(truncate("x" * 500, 100)) <= 100


# ---------------------------------------------------------------------------
# Corpus parsing
# ---------------------------------------------------------------------------


def test_parse_sections_handles_punctuated_headings():
    text = "Background: AD is progressive. Methods: We trained a CNN. Results: AUC 0.91."
    sections = parse_sections(text)
    assert set(sections) == {"Background", "Methods", "Results"}
    assert "0.91" in sections["Results"]


def test_parse_sections_handles_glued_headings():
    """Europe PMC strips JATS tags without a separator, gluing heading to body."""
    text = "BackgroundAlzheimer's disease is progressive.MethodsWe trained a model."
    sections = parse_sections(text)
    assert "Background" in sections
    assert sections["Background"].startswith("Alzheimer's disease")


def test_parse_sections_does_not_split_on_in_sentence_words():
    text = "The results were strong and the methods are described below."
    assert parse_sections(text) == {"Abstract": text}


def test_parse_sections_handles_markdown_headings():
    """Regression: an author-supplied record used `## Scope` headings. They were
    not recognised, so the record became one chunk and the raw `##` markers
    appeared verbatim in generated answers."""
    text = ("## Scope\n\nThis work addresses early detection of MCI to AD.\n\n"
            "## Approach\n\nThe central idea is multi-view analysis of structural MRI.\n\n"
            "## Position in the literature\n\nIt sits in the conversion prediction literature.")
    sections = parse_sections(text)
    assert set(sections) == {"Scope", "Approach", "Position In The Literature"}
    assert "##" not in " ".join(sections.values())
    assert sections["Approach"].startswith("The central idea")


def test_markdown_headings_do_not_leak_into_chunks():
    text = ("## Scope\n\nThis work addresses early detection of progression.\n\n"
            "## Approach\n\nThe central idea is multi-view analysis of structural MRI.")
    doc = Document(doc_id="md1", title="T", text=text, source="author_supplied",
                   sections=parse_sections(text))
    chunks = chunk_documents([doc], Config(min_chunk_chars=20))
    assert len(chunks) >= 2
    assert {c.section for c in chunks} == {"Scope", "Approach"}
    assert not any("##" in c.text for c in chunks)


def test_clean_markup_strips_tags():
    assert clean_markup("<p>Hello <b>world</b></p>") == "Hello world"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_recursive_split_respects_size_and_preserves_content():
    text = " ".join(f"Sentence number {i} about hippocampal volume." for i in range(60))
    chunks = recursive_split(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 + 60 for c in chunks)   # small slack for a long final sentence
    # Every original sentence must survive somewhere in the output.
    joined = " ".join(chunks)
    for i in range(60):
        assert f"Sentence number {i} about hippocampal volume." in joined


def test_recursive_split_short_text_is_one_chunk():
    assert recursive_split("Short abstract.", 900, 150) == ["Short abstract."]


def test_chunk_document_attaches_contextual_header_outside_text():
    doc = Document(doc_id="d1", title="Multi-view MRI for AD", text="We fuse three planes.",
                   source="arxiv", sections={"Abstract": "We fuse three planes."})
    chunks = chunk_documents([doc], Config(contextual_headers=True))
    assert len(chunks) == 1
    assert "Multi-view MRI" in chunks[0].header
    # The header must NOT leak into the text shown to the user.
    assert "Multi-view MRI" not in chunks[0].text


def test_chunk_document_splits_by_section():
    doc = Document(doc_id="d2", title="T", text="x",
                   sections={"Background": "A" * 200, "Methods": "B" * 200})
    chunks = chunk_documents([doc], Config(min_chunk_chars=50))
    assert {c.section for c in chunks} == {"Background", "Methods"}


def test_chunk_stats_reports_shape():
    text = " ".join(f"Sentence {i} about hippocampal volume in AD." for i in range(40))
    doc = Document(doc_id="d3", title="T", text=text, sections={"Abstract": text})
    stats = chunk_stats(chunk_documents([doc], Config(chunk_size=200, chunk_overlap=40)))
    assert stats["n_chunks"] > 1 and stats["n_documents"] == 1


def test_short_chunks_are_merged_not_silently_dropped():
    """A fragment below min_chunk_chars must not make source text disappear."""
    text = " ".join(f"Sentence {i} about atrophy." for i in range(30))
    doc = Document(doc_id="d4", title="T", text=text, sections={"Abstract": text})
    chunks = chunk_documents([doc], Config(chunk_size=200, chunk_overlap=40))
    recovered = " ".join(c.text for c in chunks)
    for i in range(30):
        assert f"Sentence {i} about atrophy." in recovered


def test_config_warns_when_min_chunk_chars_approaches_chunk_size():
    warnings = Config(chunk_size=120, min_chunk_chars=120).warnings()
    assert any("min_chunk_chars" in w for w in warnings)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenizer_expands_clinical_acronyms():
    tokens = Tokenizer(expand=True)("AD progression on MRI")
    assert "ad" in tokens and "alzheimer" in tokens and "disease" in tokens
    assert "magnetic" in tokens and "resonance" in tokens


def test_tokenizer_can_disable_expansion():
    tokens = Tokenizer(expand=False)("AD progression on MRI")
    assert "alzheimer" not in tokens


def test_tokenizer_keeps_hyphenated_terms_whole_and_split():
    tokens = Tokenizer()("Swin-UNETR architecture")
    assert "swin-unetr" in tokens and "swin" in tokens and "unetr" in tokens


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_bm25_prefers_exact_term_match():
    chunks = [
        make_chunk("a", "The Swin UNETR model segments brain tumours in multimodal MRI."),
        make_chunk("b", "A generic deep learning method for image analysis tasks."),
    ]
    index = BM25Index()
    index.build(chunks)
    results = index.search("Swin UNETR", top_k=2)
    assert results[0][0].chunk_id == "a"
    assert "swin" in results[0][2] or "unetr" in results[0][2]


def test_bm25_idf_is_zero_for_unseen_term():
    index = BM25Index()
    index.build([make_chunk("a", "hippocampal atrophy")])
    assert index.idf("nonexistent") == 0.0


def test_bm25_returns_nothing_for_empty_query():
    index = BM25Index()
    index.build([make_chunk("a", "some text here")])
    assert index.search("", top_k=5) == []


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_rrf_boosts_chunks_found_by_both_channels():
    shared = make_chunk("shared", "x")
    dense_only_chunk = make_chunk("dense", "y")
    sparse_only_chunk = make_chunk("sparse", "z")
    fused = reciprocal_rank_fusion([
        [(shared, 0.9), (dense_only_chunk, 0.5)],
        [(shared, 12.0), (sparse_only_chunk, 3.0)],
    ])
    assert fused[0][0].chunk_id == "shared"
    assert fused[0][2]["dense"] == 0.9 and fused[0][2]["sparse"] == 12.0
    assert fused[0][2]["rrf"] == pytest.approx(1 / 61 + 1 / 61)


def test_weighted_fusion_preserves_raw_dense_score():
    """Regression: the abstention gate compares `dense` against a threshold
    calibrated on cosine similarity, so it must be the raw value, not a
    per-query min-max normalised one."""
    chunks = [make_chunk("a", "x"), make_chunk("b", "y")]
    fused = weighted_fusion([(chunks[0], 0.62), (chunks[1], 0.31)],
                            [(chunks[1], 9.0), (chunks[0], 4.0)], dense_weight=0.2)
    by_id = {c.chunk_id: meta for c, _s, meta in fused}
    assert by_id["a"]["dense"] == pytest.approx(0.62)   # raw cosine preserved
    assert by_id["a"]["dense_norm"] == pytest.approx(1.0)  # normalised kept separately
    assert 0.0 <= by_id["a"]["dense_norm"] <= 1.0


def test_cap_per_document_limits_redundancy_but_preserves_order():
    items = [(make_chunk(f"c{i}", "t", doc_id="docA" if i < 4 else "docB"), 1.0 - i * 0.1, {})
             for i in range(6)]
    capped = cap_per_document(items, max_per_doc=2)
    doc_a = [c for c, _s, _m in capped if c.doc_id == "docA"]
    assert len(doc_a) == 2
    # Relative order must be untouched.
    scores = [s for _c, s, _m in capped]
    assert scores == sorted(scores, reverse=True)


def test_cap_per_document_disabled_by_zero():
    items = [(make_chunk(f"c{i}", "t", doc_id="docA"), 1.0, {}) for i in range(5)]
    assert len(cap_per_document(items, 0)) == 5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_ndcg_never_exceeds_one():
    """Regression: several chunks from one gold document used to inflate DCG."""
    gold_chunk = make_chunk("g1", "x", doc_id="gold")
    gold_chunk_2 = make_chunk("g2", "y", doc_id="gold")
    hits = [make_hit(1, gold_chunk), make_hit(2, gold_chunk_2),
            make_hit(3, make_chunk("n1", "z", doc_id="other"))]
    assert ndcg_at_k(hits, ["gold"], 10) <= 1.0
    assert ndcg_at_k(hits, ["gold"], 10) == pytest.approx(1.0)  # single gold doc, found at rank 1


def test_dedupe_by_document_collapses_and_renumbers():
    hits = [make_hit(1, make_chunk("a", "x", doc_id="d1")),
            make_hit(2, make_chunk("b", "y", doc_id="d1")),
            make_hit(3, make_chunk("c", "z", doc_id="d2"))]
    deduped = dedupe_by_document(hits)
    assert [h.doc_id for h in deduped] == ["d1", "d2"]
    assert [h.rank for h in deduped] == [1, 2]


def test_recall_and_precision_on_hand_built_example():
    hits = [make_hit(1, make_chunk("a", "x", doc_id="d1")),
            make_hit(2, make_chunk("b", "y", doc_id="d2")),
            make_hit(3, make_chunk("c", "z", doc_id="d3"))]
    gold = ["d1", "d3", "d9"]
    assert recall_at_k(hits, gold, 3) == pytest.approx(2 / 3)
    assert precision_at_k(hits, gold, 3) == pytest.approx(2 / 3)
    assert reciprocal_rank(hits, gold) == pytest.approx(1.0)


def test_mrr_is_zero_when_nothing_relevant():
    hits = [make_hit(1, make_chunk("a", "x", doc_id="d1"))]
    assert reciprocal_rank(hits, ["nope"]) == 0.0


def test_average_precision_rewards_early_relevance():
    early = [make_hit(1, make_chunk("a", "x", doc_id="gold")),
             make_hit(2, make_chunk("b", "y", doc_id="other"))]
    late = [make_hit(1, make_chunk("b", "y", doc_id="other")),
            make_hit(2, make_chunk("a", "x", doc_id="gold"))]
    assert average_precision(early, ["gold"]) > average_precision(late, ["gold"])


def test_retrieval_metrics_returns_all_expected_keys():
    hits = [make_hit(1, make_chunk("a", "x", doc_id="gold"))]
    metrics = retrieval_metrics(hits, ["gold"])
    for key in ("mrr", "map", "ndcg@10", "recall@1", "recall@5", "precision@5", "hit@5"):
        assert key in metrics


def test_retrieval_metrics_handles_no_gold_hit_without_crashing():
    """Regression: min() of an empty sequence used to raise."""
    metrics = retrieval_metrics([make_hit(1, make_chunk("a", "x", doc_id="other"))], ["gold"])
    assert metrics["first_relevant_rank"] == 0.0
    assert metrics["recall@5"] == 0.0


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------


def test_faithfulness_credits_verbatim_context():
    hits = [make_hit(1, make_chunk("a", "Hippocampal atrophy predicts conversion to dementia."))]
    result = faithfulness("Hippocampal atrophy predicts conversion to dementia.", hits)
    assert result["faithfulness"] == 1.0


def test_faithfulness_flags_unsupported_claims():
    hits = [make_hit(1, make_chunk("a", "The study used the ADNI dataset with 200 subjects."))]
    result = faithfulness(
        "The model achieved 99.7% accuracy on a private cohort of ten thousand patients.", hits)
    assert result["faithfulness"] < 1.0
    assert result["unsupported"]


def test_answer_correctness_counts_point_recall():
    result = answer_correctness("We used OASIS and ADNI with a CNN.", ["OASIS", "ADNI", "CNN", "Swin UNETR"])
    assert result["answer_correctness"] == pytest.approx(0.75)
    assert "Swin UNETR" in result["missing"]


# ---------------------------------------------------------------------------
# Prompting and citations
# ---------------------------------------------------------------------------


def test_system_prompt_states_refusal_token_and_citation_rule():
    prompt = build_system_prompt(Config())
    assert "INSUFFICIENT_CONTEXT" in prompt
    assert "[1]" in prompt or "bracketed markers" in prompt


def test_user_prompt_numbers_passages_and_includes_verbatim_text():
    hits = [make_hit(1, make_chunk("a", "Unique sentinel phrase here."))]
    prompt = build_user_prompt("Q?", hits, Config())
    assert "[1]" in prompt
    assert "Unique sentinel phrase here." in prompt
    assert "QUESTION:" in prompt and "CONTEXT:" in prompt


def test_parse_markers_handles_lists_and_duplicates():
    assert parse_markers("A claim [1]. Another [2, 4]. Repeated [1].") == [1, 2, 4]


def test_resolve_citations_flags_fabricated_markers():
    hits = [make_hit(1, make_chunk("a", "Some source text about atrophy."))]
    citations, fabricated = resolve_citations("Claim [1] and invented [7].", hits)
    assert [c.marker for c in citations] == [1]
    assert fabricated == [7]
    assert citations[0].quote  # a verbatim supporting span was lifted


def test_refusal_detected_tolerates_decorated_sentinel():
    assert refusal_detected("INSUFFICIENT_CONTEXT")
    assert refusal_detected("I'm sorry, but INSUFFICIENT_CONTEXT.")
    assert not refusal_detected("Hippocampal atrophy is a marker [1].")


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------


def test_confidence_signals_reads_raw_dense_score():
    hits = [make_hit(1, make_chunk("a", "x"), dense=0.71, sparse=12.0),
            make_hit(2, make_chunk("b", "y"), dense=0.55, sparse=9.0)]
    signals = confidence_signals(hits)
    assert signals["top_dense"] == pytest.approx(0.71)
    assert signals["mean_top3"] == pytest.approx((0.71 + 0.55) / 2)


def test_gate_inactive_without_calibration():
    gate = AbstentionGate(enabled=True)   # no stats
    assert not gate.active
    assert gate.decide([make_hit(1, make_chunk("a", "x"))])[0] is False


def test_gate_refuses_low_confidence_and_answers_high_confidence():
    gate = AbstentionGate(enabled=True, threshold=0.0,
                          stats={"top_dense": {"mean": 0.5, "std": 0.1}},
                          weights={"top_dense": 1.0})
    assert gate.active
    confident = [make_hit(1, make_chunk("a", "x"), dense=0.75)]
    weak = [make_hit(1, make_chunk("a", "x"), dense=0.30)]
    assert gate.decide(confident)[0] is False
    assert gate.decide(weak)[0] is True


def test_gate_refuses_when_nothing_retrieved():
    gate = AbstentionGate(enabled=True, threshold=-1.0,
                          stats={"top_dense": {"mean": 0.5, "std": 0.1}})
    assert gate.decide([])[0] is True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_merge_returns_new_object():
    base = Config()
    merged = base.merge({"top_k": 12, "mode": "dense"})
    assert merged.top_k == 12 and merged.mode == "dense"
    assert base.top_k != 12          # original untouched


def test_config_ignores_unknown_keys_without_failing():
    assert Config.from_dict({"top_k": 5, "nonsense_key": 1}).top_k == 5


def test_config_warns_on_degenerate_overlap():
    assert any("chunk_overlap" in w for w in Config(chunk_size=100, chunk_overlap=100).warnings())


def test_parse_overrides_coerces_types():
    parsed = parse_overrides(["top_k=10", "rerank=false", "dense_weight=0.25", "abstain_threshold=null"])
    assert parsed == {"top_k": 10, "rerank": False, "dense_weight": 0.25, "abstain_threshold": None}


# ---------------------------------------------------------------------------
# Extractive backend
# ---------------------------------------------------------------------------


def test_extractive_backend_cites_its_source_passages():
    config = Config()
    llm = ExtractiveLLM(config)
    hits = [make_hit(1, make_chunk("a", "Hippocampal atrophy is a strong predictor of conversion to dementia.")),
            make_hit(2, make_chunk("b", "The weather in Mardan is warm in June."))]
    out = llm.generate("Does hippocampal atrophy predict conversion?", hits)
    assert "[1]" in out
    assert "hippocampal" in out.lower()


def test_extractive_backend_refuses_when_nothing_matches():
    llm = ExtractiveLLM(Config())
    out = llm.generate("What is the capital of Mongolia?",
                       [make_hit(1, make_chunk("a", "Hippocampal atrophy predicts dementia."))])
    assert out == Config().refusal_token


# ---------------------------------------------------------------------------
# End-to-end (offline)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def offline_pipeline():
    """A real pipeline over the committed corpus, with no ML dependencies."""
    from neurorag.pipeline import RAGPipeline

    config = Config.load(ROOT / "config.yaml").merge({
        "embed_backend": "lexical",   # offline encoder, no model download
        "rerank": False,
        "llm": "extractive",
        "abstain": False,             # gate is calibrated for the semantic encoder
    })
    if not (ROOT / config.corpus_path).exists():
        pytest.skip("corpus snapshot not present; run scripts/build_corpus.py")
    return RAGPipeline.build(config, save=False)


def test_end_to_end_answer_carries_citations_and_provenance(offline_pipeline):
    answer = offline_pipeline.ask(
        "Why is subject-level data splitting important for brain MRI classification?", top_k=4)
    assert answer.text
    assert answer.citations, "a grounded answer must cite at least one passage"
    assert all(c.url or c.title for c in answer.citations)
    assert answer.config.get("fabricated_markers", ["x"]) == []
    assert answer.config.get("audit", {}).get("citation_coverage", 0) > 0


def test_end_to_end_retrieval_is_document_diverse(offline_pipeline):
    hits = offline_pipeline.retrieve("Alzheimer disease MRI deep learning classification", top_k=6)
    assert hits
    doc_counts: dict[str, int] = {}
    for hit in hits:
        doc_counts[hit.doc_id] = doc_counts.get(hit.doc_id, 0) + 1
    assert max(doc_counts.values()) <= offline_pipeline.config.max_chunks_per_doc


def test_end_to_end_metadata_filter_restricts_results(offline_pipeline):
    hits = offline_pipeline.retrieve("MRI classification", top_k=6,
                                     filters={"strata": ["rag"]})
    assert all(h.chunk.doc_stratum == "rag" for h in hits)


def test_end_to_end_abstains_on_empty_filter(offline_pipeline):
    hits = offline_pipeline.retrieve("anything", top_k=5, filters={"strata": ["nonexistent"]})
    assert hits == []


def test_gate_refuses_to_activate_for_a_different_embedder(tmp_path):
    """Regression: the threshold is fitted to one encoder's score scale. Applying
    a semantic-calibrated threshold to lexical-encoder scores produced wrong
    abstention decisions with no error anywhere."""
    import json

    gate_file = tmp_path / "abstention.json"
    gate_file.write_text(json.dumps({
        "threshold": -0.3, "weights": {"top_dense": 1.0},
        "stats": {"top_dense": {"mean": 0.5, "std": 0.1}},
        "calibrated_on": "test", "calibrated_for_embedder": "all-MiniLM-L6-v2",
    }), encoding="utf-8")

    config = Config(abstain=True, abstain_path=str(gate_file))
    assert AbstentionGate.load(config, embedder_name="all-MiniLM-L6-v2").active
    assert not AbstentionGate.load(config, embedder_name="lexical").active
    assert not AbstentionGate.load(Config(abstain=False, abstain_path=str(gate_file))).active
