"""Generation: grounded prompting, LLM backends, citation resolution, abstention.

The prompt is the enforcement mechanism for faithfulness, and three choices
matter more than the wording:

1. **Passages are numbered and self-describing** (title, year, section), so the
   model can cite meaningfully and cannot silently merge two papers' claims.
2. **Refusal is a first-class output.** Without an explicit escape hatch a
   capable model will almost always produce *something* plausible, which in a
   clinical literature assistant is the worst possible failure mode.
3. **The citation protocol is constrained to a parseable form** (``[n]``,
   ``[n, m]``). Free-form attribution is unauditable.

Passages are passed verbatim. Nothing is paraphrased or summarised before
prompting, because any rewriting at that stage inserts an unattributed claim
between the source and the answer.

Abstention
----------
Prompt-level refusal is necessary but not sufficient: given *any* context a model
will usually find something to say, and near-miss context is the dangerous case.
So the system also abstains *before* prompting, on a signal the generator cannot
talk its way out of - how confident the retriever is that it found anything
relevant. ``calibrate_abstention`` in :mod:`neurorag.evaluate` learns that
threshold from the golden set instead of guessing a constant.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Config
from .schema import Answer, Citation, Hit
from .utils import get_logger, normalize_whitespace, split_sentences, truncate

log = get_logger("neurorag.generator")

# ===========================================================================
# Prompting
# ===========================================================================

SYSTEM_PROMPT = """\
{persona}

Rules you MUST follow:
1. Answer ONLY from the numbered CONTEXT passages below. Never use prior
   knowledge, even if you know the answer.
2. Cite every factual claim with the passage number(s) it comes from, using
   bracketed markers such as [1] or [2, 4], placed at the end of the sentence
   they support.
3. If the CONTEXT passages do not contain enough information to answer, respond
   with exactly this token and nothing else: {refusal_token}
4. Never invent or guess a passage number. Only cite passages you actually used.
5. Preserve numbers, dataset names, metric names and units exactly as written in
   the passages. Do not round, restate or "improve" a reported value.
6. If passages disagree with each other, say so explicitly and cite both.
7. Be concise and technical. Do not preface your answer, do not describe what you
   are doing, and do not repeat the question."""


def build_system_prompt(config: Config) -> str:
    return SYSTEM_PROMPT.format(persona=config.persona.strip(),
                                refusal_token=config.refusal_token).strip()


def passage_header(hit: Hit) -> str:
    """Compact provenance line for one context block."""
    chunk = hit.chunk
    parts = []
    if chunk.doc_title:
        parts.append(truncate(chunk.doc_title, 150))
    if chunk.doc_year:
        parts.append(str(chunk.doc_year))
    if chunk.section:
        parts.append(f"section: {chunk.section}")
    return " | ".join(parts)


def format_context(hits: Sequence[Hit], max_chars: int = 2200, show_scores: bool = False) -> str:
    """Render retrieved chunks as numbered context blocks."""
    if not hits:
        return "(no passages retrieved)"
    blocks = []
    for hit in hits:
        score_line = ""
        if show_scores:
            detail = " ".join(f"{k}={v:.4f}" for k, v in hit.scores.items())
            score_line = f"\nretrieval: score={hit.score:.4f} {detail}"
        body = truncate(hit.chunk.text.strip(), max_chars)
        blocks.append(f"[{hit.rank}] {passage_header(hit)}{score_line}\n\"\"\"\n{body}\n\"\"\"")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits: Sequence[Hit], config: Config,
                      extra: Optional[str] = None, show_scores: bool = False) -> str:
    """Build the user turn.

    The generator always sees the *raw* question. Query-side tricks (acronym
    expansion, filters) are retrieval devices; putting them in the prompt would
    only add noise for the model to wade through.
    """
    parts = [f"QUESTION:\n{question.strip()}",
             f"\nCONTEXT:\n{format_context(hits, config.max_context_chars, show_scores)}"]
    if extra:
        parts.append(f"\nADDITIONAL INSTRUCTION:\n{extra.strip()}")
    parts.append("\nANSWER:")
    return "\n".join(parts)


# ===========================================================================
# LLM backends
# ===========================================================================


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    """POST JSON using urllib, so no vendor SDK is required."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise LLMError(f"HTTP {exc.code} from {url}: {exc.read().decode()[:300]}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Could not reach {url}: {exc.reason}") from exc


class LLMError(RuntimeError):
    """A backend could not produce a completion."""


class BaseLLM:
    name = "base"
    is_generative = False

    def __init__(self, config: Config):
        self.config = config

    def complete(self, messages: Sequence[Dict[str, str]]) -> str:
        raise NotImplementedError(f"{type(self).__name__} has no chat endpoint; use generate().")

    def generate(self, question: str, hits: Sequence[Hit]) -> str:
        messages = [
            {"role": "system", "content": build_system_prompt(self.config)},
            {"role": "user", "content": build_user_prompt(question, hits, self.config)},
        ]
        return self.complete(messages)


class OpenAILLM(BaseLLM):
    """OpenAI-compatible chat completions.

    Also works with any endpoint exposing ``/v1/chat/completions`` (vLLM, LM
    Studio, Together, Groq's compatibility layer) via ``OPENAI_BASE_URL``.
    """

    is_generative = True

    def __init__(self, config: Config):
        super().__init__(config)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.name = f"openai:{config.llm_model}"

    def complete(self, messages):
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        data = _post_json(
            f"{self.base_url}/chat/completions",
            {"model": self.config.llm_model, "messages": list(messages),
             "temperature": self.config.temperature, "max_tokens": self.config.max_tokens},
            {"Authorization": f"Bearer {self.api_key}"}, self.config.timeout_s)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenAI response: {str(data)[:300]}") from exc


class GeminiLLM(BaseLLM):
    is_generative = True

    def __init__(self, config: Config):
        super().__init__(config)
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.name = f"gemini:{config.gemini_model}"

    def complete(self, messages):
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY / GOOGLE_API_KEY is not set")
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        payload: Dict[str, Any] = {
            "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                          "parts": [{"text": m["content"]}]}
                         for m in messages if m["role"] != "system"],
            "generationConfig": {"temperature": self.config.temperature,
                                 "maxOutputTokens": self.config.max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.gemini_model}:generateContent"
        data = _post_json(url, payload, {"x-goog-api-key": self.api_key}, self.config.timeout_s)
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected Gemini response: {str(data)[:300]}") from exc


class OllamaLLM(BaseLLM):
    is_generative = True

    def __init__(self, config: Config):
        super().__init__(config)
        self.host = (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.name = f"ollama:{config.ollama_model}"

    def complete(self, messages):
        data = _post_json(f"{self.host}/api/chat", {
            "model": self.config.ollama_model, "messages": list(messages), "stream": False,
            "options": {"temperature": self.config.temperature, "num_predict": self.config.max_tokens},
        }, {}, self.config.timeout_s)
        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Ollama response: {str(data)[:300]}") from exc


class ExtractiveLLM(BaseLLM):
    """Offline, no-model backend: composes an answer from retrieved sentences.

    Why this exists
    ---------------
    A RAG repository that cannot run without an API key cannot be reviewed. This
    backend makes the whole system - retrieval, ranking, citation resolution,
    refusal, evaluation - exercisable on a bare Python install. It selects the
    sentences from the retrieved passages that best match the question (question
    terms weighted by their inverse frequency across the supplied passages, plus a
    positional bonus because the claim in a biomedical abstract is usually in the
    first two sentences) and emits them verbatim with their passage markers.

    The output is therefore faithful *by construction*, which is the point: it
    establishes the ceiling for faithfulness and tests the surrounding machinery.
    It is NOT a substitute for a generative model - it cannot synthesise across
    sources, compare results, or answer a question whose answer requires
    combining two passages. It is labelled ``extractive`` everywhere it appears so
    no number it produces can be mistaken for LLM output.

    It also refuses honestly: if nothing clears ``threshold`` it returns the
    refusal sentinel rather than emitting weakly-related text.
    """

    is_generative = False
    name = "extractive"

    def __init__(self, config: Config, threshold: float = 0.12, max_sentences: int = 4):
        super().__init__(config)
        self.threshold = threshold
        self.max_sentences = max_sentences
        self.last_scores: Dict[str, Any] = {}

    def generate(self, question: str, hits: Sequence[Hit]) -> str:  # type: ignore[override]
        if not hits:
            return self.config.refusal_token

        q_terms = _content_tokens(question)
        if not q_terms:
            return self.config.refusal_token

        # IDF across the supplied passages only: a term present in every passage
        # discriminates nothing for this particular question.
        df: Dict[str, int] = {}
        passage_tokens = []
        for hit in hits:
            toks = set(_content_tokens(hit.chunk.text))
            passage_tokens.append(toks)
            for t in toks:
                df[t] = df.get(t, 0) + 1
        n = max(len(hits), 1)
        idf = {t: math_log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}

        candidates = []
        for hit, toks in zip(hits, passage_tokens):
            sentences = [s for s in split_sentences(hit.chunk.text) if len(s) > 40]
            for pos, sentence in enumerate(sentences):
                overlap = q_terms & set(_content_tokens(sentence))
                if not overlap:
                    continue
                score = sum(idf.get(t, 1.0) for t in overlap) / len(q_terms)
                score += 0.10 / (1 + pos)          # positional bonus
                score += 0.05 / hit.rank           # first-stage rank bonus
                candidates.append((score, sentence, hit.rank, hit.doc_id))

        candidates.sort(key=lambda c: (-c[0], c[2]))
        self.last_scores = {
            "n_candidates": len(candidates),
            "best_score": round(candidates[0][0], 4) if candidates else 0.0,
            "threshold": self.threshold,
        }
        if not candidates or candidates[0][0] < self.threshold:
            log.info("Extractive backend refusing (best score %.4f < %.4f)",
                     candidates[0][0] if candidates else 0.0, self.threshold)
            return self.config.refusal_token

        # At most two sentences per source, for diversity.
        per_doc: Dict[str, int] = {}
        chosen = []
        for score, sentence, rank, doc_id in candidates:
            if per_doc.get(doc_id, 0) >= 2:
                continue
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            chosen.append(f"{normalize_whitespace(sentence)} [{rank}]")
            if len(chosen) >= self.max_sentences:
                break
        return " ".join(chosen)


def math_log(x: float) -> float:
    import math
    return math.log(x)


_STOP = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
         "what", "which", "how", "why", "who", "when", "where", "does", "do", "is",
         "a", "an", "of", "in", "on", "to", "be", "it", "its", "as", "at", "by"}


def _content_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9\-]*", (text or "").lower())
            if t not in _STOP and len(t) > 2}


def _ollama_available(config: Config) -> bool:
    host = (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    try:
        req = urllib.request.Request(f"{host}/api/tags", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            models = {m.get("name", "") for m in json.loads(resp.read().decode()).get("models", [])}
        wanted = config.ollama_model.split(":")[0]
        if any(m.split(":")[0] == wanted for m in models):
            return True
        log.info("Ollama is running but '%s' is not pulled (available: %s)",
                 config.ollama_model, sorted(models)[:5] or "none")
        return False
    except Exception:
        return False


def build_llm(config: Config) -> BaseLLM:
    """Resolve ``llm: auto`` to the best backend actually usable here."""
    choice = (config.llm or "auto").lower()
    if choice == "extractive":
        return ExtractiveLLM(config)
    if choice == "openai":
        return OpenAILLM(config)
    if choice == "gemini":
        return GeminiLLM(config)
    if choice == "ollama":
        return OllamaLLM(config)

    if os.environ.get("OPENAI_API_KEY"):
        log.info("Using OpenAI-compatible backend (%s).", config.llm_model)
        return OpenAILLM(config)
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        log.info("Using Gemini backend (%s).", config.gemini_model)
        return GeminiLLM(config)
    if _ollama_available(config):
        log.info("Using local Ollama backend (%s).", config.ollama_model)
        return OllamaLLM(config)

    log.warning("=" * 70)
    log.warning("No generative LLM available (no OPENAI_API_KEY, no GEMINI_API_KEY,")
    log.warning("no reachable Ollama). Using the OFFLINE EXTRACTIVE backend:")
    log.warning("answers are verbatim sentences selected from retrieved passages.")
    log.warning("Retrieval, citation resolution, abstention and evaluation are")
    log.warning("fully exercised; abstractive synthesis and fluency are NOT.")
    log.warning("Set an API key, or run:  ollama pull %s", config.ollama_model)
    log.warning("=" * 70)
    return ExtractiveLLM(config)


# ===========================================================================
# Citations and auditing
# ===========================================================================

_MARKER = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def parse_markers(text: str) -> List[int]:
    """Every passage number cited in the text, de-duplicated and sorted."""
    found = []
    for group in _MARKER.findall(text or ""):
        found.extend(int(p) for p in re.split(r"[,;]", group) if p.strip().isdigit())
    return sorted(set(found))


def count_markers(text: str) -> int:
    """Total individual marker references (not unique)."""
    return sum(len([p for p in re.split(r"[,;]", g) if p.strip().isdigit()])
               for g in _MARKER.findall(text or ""))


def sentences_by_citation(text: str) -> Tuple[List[str], List[str]]:
    """Return (sentences carrying a marker, sentences without one)."""
    cited, uncited = [], []
    for sentence in split_sentences(text or ""):
        (cited if _MARKER.search(sentence) else uncited).append(sentence)
    return cited, uncited


def _supporting_quote(rank: int, hits: Sequence[Hit], max_chars: int = 260) -> str:
    """Verbatim span from the cited passage.

    The quote is what makes a citation checkable at a glance - a reader should
    never have to trust the model's paraphrase.
    """
    for hit in hits:
        if hit.rank != rank:
            continue
        sentences = [s for s in split_sentences(hit.chunk.text) if len(s) > 30]
        return truncate(normalize_whitespace(sentences[0] if sentences else hit.chunk.text), max_chars)
    return ""


def resolve_citations(text: str, hits: Sequence[Hit]) -> Tuple[List[Citation], List[int]]:
    """Map cited markers onto retrieved passages.

    Returns ``(citations, fabricated)``. A marker is *fabricated* when it points
    at a passage that was never supplied - the model invented a reference number.
    That is a distinct failure from hallucinating content and needs its own metric.
    """
    by_rank = {hit.rank: hit for hit in hits}
    citations, fabricated = [], []
    for marker in parse_markers(text):
        hit = by_rank.get(marker)
        if hit is None:
            fabricated.append(marker)
            continue
        citations.append(Citation(
            marker=marker, chunk_id=hit.chunk.chunk_id, doc_id=hit.doc_id,
            title=hit.chunk.doc_title or "(untitled)",
            quote=_supporting_quote(marker, hits),
            section=hit.chunk.section, url=hit.chunk.doc_url,
            year=hit.chunk.doc_year, authors=hit.chunk.doc_authors,
        ))
    if fabricated:
        log.warning("Model cited %d passage number(s) never supplied: %s", len(fabricated), fabricated)
    return citations, fabricated


_REFUSAL_PHRASES = (
    "insufficient_context", "insufficient context",
    "cannot answer from the provided context", "the context does not contain",
    "the passages do not contain", "not enough information in the context",
    "i cannot answer this from the given context",
)


def refusal_detected(text: str, refusal_token: str = "INSUFFICIENT_CONTEXT") -> bool:
    """Detect a refusal even when the model decorates the sentinel token."""
    if not (text or "").strip():
        return True
    lowered = text.lower().strip()
    if refusal_token.lower() in lowered:
        return True
    return len(lowered) < 220 and any(p in lowered for p in _REFUSAL_PHRASES)


def refusal_reason(text: str) -> str:
    lowered = (text or "").lower()
    if not lowered.strip():
        return "empty response"
    if "insufficient_context" in lowered:
        return "model returned INSUFFICIENT_CONTEXT"
    for phrase in _REFUSAL_PHRASES:
        if phrase in lowered:
            return f"model declined ('{phrase}')"
    return "model declined"


def audit(answer: Answer) -> Dict[str, float]:
    """Citation-level audit metrics for one answer."""
    sentences = split_sentences(answer.text)
    cited, uncited = sentences_by_citation(answer.text)
    markers = parse_markers(answer.text)
    unique_docs = {c.doc_id for c in answer.citations}
    n = max(len(sentences), 1)
    return {
        "n_sentences": len(sentences),
        "citation_coverage": round(len(cited) / n, 3),
        "citation_density": round(count_markers(answer.text) / n, 3),
        "uncited_sentences": len(uncited),
        "unique_sources": len(unique_docs),
        "passages_supplied": len(answer.hits),
        "source_diversity": round(len(unique_docs) / max(len(answer.hits), 1), 3),
    }


def references_block(answer: Answer) -> str:
    if not answer.citations:
        return "_No sources cited._"
    return "\n".join(c.as_reference() for c in sorted(answer.citations, key=lambda c: c.marker))


# ===========================================================================
# Abstention
# ===========================================================================


def confidence_signals(hits: Sequence[Hit]) -> Dict[str, float]:
    """Retrieval-confidence signals used to decide whether to abstain.

    ``top_dense``  cosine similarity of the best dense hit. Interpretable:
                   ~0.75 is a near-paraphrase, ~0.35 is topical noise.
    ``top_rerank`` cross-encoder score for the best passage. The most
                   discriminative signal (query and passage are read jointly) but
                   unbounded, so its threshold does not transfer between models.
    ``mean_top3``  mean dense similarity of the top three - a smoother estimate
                   that is less swayed by one lucky hit.
    ``top_sparse`` best raw BM25 score. Included because dense cosine is
                   systematically *lower* for abstract conceptual questions than
                   for specific factual ones even when retrieval succeeded, so a
                   dense-only gate over-refuses exactly the interesting queries.
                   BM25 is unbounded and query-length dependent, so it is only
                   ever used z-scored against the calibration distribution.
    """
    if not hits:
        return {"top_dense": 0.0, "top_rerank": 0.0, "mean_top3": 0.0, "top_sparse": 0.0}
    dense = [h.scores["dense"] for h in hits if "dense" in h.scores]
    rerank = [h.scores["rerank"] for h in hits if "rerank" in h.scores]
    sparse = [h.scores["sparse"] for h in hits if "sparse" in h.scores]
    ordered = sorted(dense, reverse=True)
    return {
        "top_dense": round(max(dense) if dense else 0.0, 6),
        "top_rerank": round(max(rerank) if rerank else 0.0, 6),
        "mean_top3": round(statistics.fmean(ordered[:3]) if ordered else 0.0, 6),
        "top_sparse": round(max(sparse) if sparse else 0.0, 6),
    }


@dataclass
class AbstentionGate:
    """Decides whether to answer or abstain, from retrieval confidence alone.

    Signals are z-scored using statistics learned during calibration and then
    averaged, because no single signal is reliable across question types. Without
    a calibration file the gate is inactive (fail-open): a fresh clone should
    still answer questions rather than refuse everything.
    """

    enabled: bool = False
    threshold: float = 0.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "top_dense": 1.0, "top_rerank": 1.0, "mean_top3": 0.5, "top_sparse": 0.5})
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    calibrated_on: str = ""
    # The encoder whose score distribution the threshold was fitted to. The
    # threshold is NOT transferable: cosine similarity from a semantic bi-encoder
    # and from the hashed-lexical fallback live on completely different scales, so
    # applying a semantic-calibrated threshold to lexical scores produces wrong
    # abstention decisions with no error anywhere.
    calibrated_for_embedder: str = ""

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.stats)

    def score_from_signals(self, signals: Dict[str, float]) -> float:
        """Z-score each signal with the calibration statistics and average them."""
        total, weight_sum = 0.0, 0.0
        for name, weight in self.weights.items():
            stat = self.stats.get(name)
            if not stat:
                continue
            std = stat.get("std") or 1.0
            z = (signals.get(name, 0.0) - stat.get("mean", 0.0)) / (std if std > 1e-9 else 1.0)
            total += weight * z
            weight_sum += weight
        return total / weight_sum if weight_sum else 0.0

    def score(self, hits: Sequence[Hit]) -> float:
        return self.score_from_signals(confidence_signals(hits))

    def decide(self, hits: Sequence[Hit]) -> Tuple[bool, float, str]:
        """Return ``(should_refuse, score, reason)``."""
        if not self.active:
            return False, 0.0, "gate inactive"
        if not hits:
            return True, 0.0, "no passages retrieved"
        score = self.score(hits)
        if score < self.threshold:
            return True, score, (f"retrieval confidence {score:.3f} below calibrated "
                                 f"threshold {self.threshold:.3f}")
        return False, score, f"retrieval confidence {score:.3f} >= threshold {self.threshold:.3f}"

    @classmethod
    def load(cls, config: Config, embedder_name: Optional[str] = None) -> "AbstentionGate":
        """Load the calibrated gate, refusing to activate it against a different encoder."""
        if not config.abstain:
            return cls(enabled=False)
        path = Path(config.abstain_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        if not path.exists():
            log.warning("No calibrated abstention gate at %s; the gate is INACTIVE and the "
                        "system relies on prompt-level refusal only. Run "
                        "`python scripts/evaluate.py --calibrate` to calibrate it.", path)
            return cls(enabled=False)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            calibrated_for = data.get("calibrated_for_embedder", "")
            if not calibrated_for:
                log.warning("Abstention calibration at %s predates embedder tracking, so its "
                            "provenance is unknown. Re-run `python scripts/evaluate.py --calibrate` "
                            "to make it safe against encoder changes.", path)
            if embedder_name and calibrated_for and calibrated_for != embedder_name:
                log.warning(
                    "Abstention gate was calibrated for embedder '%s' but this index uses '%s'. "
                    "The threshold is not transferable between encoders (their score scales "
                    "differ), so the gate is INACTIVE. Re-calibrate with "
                    "`python scripts/evaluate.py --calibrate`.", calibrated_for, embedder_name)
                return cls(enabled=False)
            gate = cls(
                enabled=True,
                threshold=float(data.get("threshold", 0.0)),
                weights=data.get("weights", cls().weights),
                stats=data.get("stats", {}),
                calibrated_on=data.get("calibrated_on", ""),
                calibrated_for_embedder=calibrated_for,
            )
            log.info("Abstention gate loaded: threshold=%.3f (calibrated on %s, embedder=%s)",
                     gate.threshold, gate.calibrated_on or "unknown", calibrated_for or "unspecified")
            return gate
        except Exception as exc:
            log.warning("Could not read %s: %s; gate inactive.", path, exc)
            return cls(enabled=False)
