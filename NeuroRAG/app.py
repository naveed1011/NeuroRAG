"""NeuroRAG - Streamlit demo.

    streamlit run app.py

The UI is deliberately built around *evidence*, not just answers: alongside the
response it shows the retrieval trace (per-channel scores), the exact prompt sent
to the model, and the citation audit. That is what makes a RAG system reviewable
rather than a black box that happens to sound confident.

If no index exists yet it offers to build one in-place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from neurorag.config import Config                                    # noqa: E402
from neurorag.evaluate import faithfulness                            # noqa: E402
from neurorag.generator import build_system_prompt, build_user_prompt  # noqa: E402
from neurorag.pipeline import RAGPipeline                             # noqa: E402
from neurorag.utils import truncate                                   # noqa: E402

st.set_page_config(page_title="NeuroRAG", page_icon="🧠", layout="wide")

EXAMPLES = [
    "Why is subject-level cross-validation important for MRI classification of Alzheimer's disease?",
    "What two kinds of memory does a retrieval-augmented generation model combine?",
    "Which three orthogonal MRI planes are used for multi-view analysis?",
    "Why is late fusion of per-view features considered a limitation in multi-view MRI diagnosis?",
    "How does Grad-CAM produce a visual explanation from a convolutional network?",
    # Should be refused: the framework is mentioned in the corpus but no accuracy
    # figure exists anywhere in it. A confident number here would be a hallucination.
    "What patient-level accuracy does AD-TriFuseViT report on OASIS-1?",
]


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the retrieval index ...")
def get_pipeline(config_overrides: tuple) -> RAGPipeline:
    """Load (or build) the pipeline. Cached across reruns.

    Overrides arrive as a hashable tuple because Streamlit cache keys must be.
    """
    config = Config.load(ROOT / "config.yaml")
    if config_overrides:
        config = config.merge(dict(config_overrides))
    try:
        return RAGPipeline.load(config)
    except (FileNotFoundError, RuntimeError):
        return RAGPipeline.build(config)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧠 NeuroRAG")
    st.caption("Citation-grounded RAG over Alzheimer's disease & neuroimaging literature")

    st.subheader("Retrieval")
    mode = st.selectbox("Fusion mode", ["weighted", "hybrid", "dense", "sparse"], index=0,
                        help="weighted/dense/sparse/RRF-hybrid. The default is the measured optimum.")
    top_k = st.slider("Passages to the generator (top_k)", 1, 12, 6)
    rerank = st.toggle("Cross-encoder reranking", value=False,
                       help="Off by default: measured to hurt on this corpus, because "
                            "ms-marco-MiniLM is trained on web passages. See docs/EVALUATION.md.")
    abstain = st.toggle("Abstention gate", value=True,
                        help="Refuse before prompting when retrieval confidence is below a "
                             "calibrated threshold.")

    st.subheader("Filters")
    strata = st.multiselect("Stratum", ["clinical", "methods", "rag"], default=[],
                            help="Restrict retrieval before scoring.")
    year_min = st.number_input("From year", min_value=1990, max_value=2030, value=2015, step=1)
    year_max = st.number_input("To year", min_value=1990, max_value=2030, value=2030, step=1)
    use_years = st.toggle("Apply year filter", value=False)

    st.divider()
    if st.button("Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding ..."):
            get_pipeline.clear()
            RAGPipeline.build(Config.load(ROOT / "config.yaml").merge(
                {"mode": mode, "top_k": top_k, "rerank": rerank, "abstain": abstain}))
        st.success("Index rebuilt.")
        st.rerun()

pipeline = get_pipeline((("mode", mode), ("top_k", top_k), ("rerank", rerank), ("abstain", abstain)))

with st.sidebar:
    st.divider()
    st.subheader("Backends")
    st.code(pipeline.retriever.describe(), language=None)
    llm = pipeline.llm
    st.markdown(f"**Generator:** `{llm.name}`")
    if not llm.is_generative:
        st.warning("No LLM API key or local Ollama found, so answers are composed from "
                   "verbatim retrieved sentences. Retrieval, citation and abstention "
                   "behaviour are genuine; fluency and cross-source synthesis are not.",
                   icon="⚠️")
    else:
        st.success("Generative backend active.")
    semantic = getattr(pipeline.embedder, "is_semantic", False)
    if not semantic:
        st.warning("Using the offline lexical encoder - retrieval is keyword-based, not semantic.",
                   icon="⚠️")
    st.caption(f"{len(pipeline.documents)} documents · {len(pipeline.chunks)} chunks")

    with st.expander("Full configuration"):
        st.json(pipeline.config.snapshot())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Ask the literature")
st.caption("Every claim is cited back to a specific passage of a specific paper. "
           "When the corpus cannot support an answer, the system says so instead of guessing.")

col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input("Question", placeholder="e.g. Why does slice-level cross-validation inflate accuracy?")
with col2:
    st.write("")
    st.write("")
    run = st.button("Ask", type="primary", use_container_width=True)

st.markdown("**Try one of these:**")
example_cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    with example_cols[i % 2]:
        if st.button(truncate(example, 78), key=f"ex{i}", use_container_width=True):
            st.session_state["question"] = example
            st.rerun()

question = st.session_state.pop("question", question)
if question and st.session_state.get("last_question") != question:
    run = True

if not (run and question):
    st.info("Enter a question above, or pick one of the examples.")
    st.stop()

st.session_state["last_question"] = question

filters: dict = {}
if strata:
    filters["strata"] = strata
if use_years:
    filters["year_min"], filters["year_max"] = int(year_min), int(year_max)
filters = filters or None

with st.spinner("Retrieving and generating ..."):
    answer = pipeline.ask(question, top_k=top_k, filters=filters)

# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

if answer.refused:
    st.error("### ⛔ Abstained")
    st.markdown(
        f"**Why:** {answer.refusal_reason}\n\n"
        "The retrieved passages did not support an answer. In a clinical literature "
        "assistant this is the correct behaviour - a confident wrong answer is worse "
        "than no answer."
    )
    if answer.confidence is not None and pipeline.gate.active:
        st.metric("Retrieval confidence", f"{answer.confidence:.3f}",
                  help=f"Calibrated abstention threshold: {pipeline.gate.threshold:.3f}")
else:
    st.success("### Answer")
    st.markdown(answer.text)

    metrics = st.columns(4)
    audit_info = answer.config.get("audit", {})
    metrics[0].metric("Sources cited", len(answer.citations))
    metrics[1].metric("Citation coverage", f"{audit_info.get('citation_coverage', 0):.0%}",
                      help="Fraction of answer sentences carrying a citation marker")
    metrics[2].metric("Fabricated markers", audit_info.get("fabricated_markers", 0)
                      if isinstance(audit_info.get("fabricated_markers"), int) else 0,
                      help="Citations pointing at passages never supplied")
    metrics[3].metric("Latency", f"{sum(answer.latency_ms.values()):.0f} ms")

# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

if answer.citations:
    st.subheader("Sources")
    for citation in sorted(answer.citations, key=lambda c: c.marker):
        authors = ", ".join((citation.authors or [])[:4])
        if len(citation.authors or []) > 4:
            authors += " et al."
        with st.container(border=True):
            st.markdown(f"**\\[{citation.marker}]** {authors or 'Unknown'} "
                        f"({citation.year or 'n.d.'}) — *{citation.title}*")
            meta = []
            if citation.section:
                meta.append(f"section: `{citation.section}`")
            if citation.url:
                meta.append(f"[link]({citation.url})")
            if meta:
                st.caption(" · ".join(meta))
            st.info(f'"{citation.quote}"', icon="📄")

# ---------------------------------------------------------------------------
# Retrieval trace
# ---------------------------------------------------------------------------

with st.expander("🔍 Retrieval trace (what the retriever actually did)"):
    if answer.hits:
        rows = []
        for hit in answer.hits:
            scores = hit.scores
            rows.append({
                "#": hit.rank,
                "final": round(hit.score, 4),
                "dense (cosine)": round(scores.get("dense", 0.0), 4),
                "sparse (BM25)": round(scores.get("sparse", 0.0), 3),
                "fused": round(scores.get("weighted", scores.get("rrf", 0.0)), 4),
                "rerank": round(scores["rerank"], 3) if "rerank" in scores else None,
                "document": truncate(hit.chunk.doc_title or hit.doc_id, 60),
                "section": hit.chunk.section or "",
                "BM25 matched": ", ".join(hit.matched_terms[:6]),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("`dense` is the raw cosine similarity, `sparse` the BM25 score. "
                   "Both are kept so you can see which channel produced a hit - "
                   "that is usually the fastest way to diagnose a bad retrieval.")
        if answer.confidence is not None and pipeline.gate.active:
            st.markdown(f"**Abstention confidence:** `{answer.confidence:.4f}` "
                        f"(threshold `{pipeline.gate.threshold:.4f}`) — "
                        f"signals: `{answer.config.get('signals')}`")
    else:
        st.write("No passages were retrieved.")

    st.markdown("**Retrieved passages**")
    for hit in answer.hits:
        with st.container(border=True):
            st.markdown(f"`[{hit.rank}]` **{truncate(hit.chunk.doc_title or hit.doc_id, 90)}** "
                        f"· {hit.chunk.section or ''}")
            st.write(hit.chunk.text)

# ---------------------------------------------------------------------------
# Prompt inspection
# ---------------------------------------------------------------------------

with st.expander("📝 The exact prompt sent to the model"):
    st.markdown("**System**")
    st.code(build_system_prompt(pipeline.config), language=None)
    st.markdown("**User**")
    st.code(build_user_prompt(question, answer.hits, pipeline.config), language=None)
    st.caption("Passages are inserted verbatim. Nothing is paraphrased before prompting, "
               "because any rewriting at that stage would insert an unattributed claim "
               "between the source and the answer.")

# ---------------------------------------------------------------------------
# Faithfulness probe
# ---------------------------------------------------------------------------

if not answer.refused:
    with st.expander("🧪 Faithfulness probe (answer vs retrieved context)"):
        encode_fn = lambda texts: pipeline.embedder.encode(list(texts))  # noqa: E731
        result = faithfulness(answer.text, answer.hits, encode_fn=encode_fn)
        st.metric("Faithfulness", f"{result['faithfulness']:.0%}",
                  help="Fraction of answer sentences supported by the retrieved context.")
        st.caption(f"{result['n_supported']}/{result['n_sentences']} sentences supported. "
                   "Measured by embedding similarity to context sentences plus a lexical "
                   "containment test - a deterministic proxy, not an LLM judge.")
        if result["unsupported"]:
            st.warning("Sentences the context does not clearly support:")
            for sentence in result["unsupported"]:
                st.markdown(f"- {sentence}")
        else:
            st.success("Every answer sentence is supported by a retrieved passage.")

st.divider()
st.caption("NeuroRAG · answers are grounded in a committed snapshot of 436 papers from "
           "Europe PMC, arXiv and Crossref. See `docs/EVALUATION.md` for measured quality.")
