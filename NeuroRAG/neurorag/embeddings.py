"""Embedding backends.

Three interchangeable encoders behind one interface, chosen by ``embed_backend``:

``sentence_transformers`` (default)
    A local Hugging Face bi-encoder. ``all-MiniLM-L6-v2`` is the shipped default
    because it runs comfortably on a laptop CPU and inside a Kaggle notebook. For
    clinical vocabulary, ``NeuML/pubmedbert-base-embeddings`` is better but needs
    ~3 GB of RAM.

``openai``
    Hosted embeddings, used when ``OPENAI_API_KEY`` is set and no local model is
    available.

``lexical``
    **Offline, zero-ML-dependency** fallback: IDF-weighted sublinear-TF
    bag-of-words passed through a deterministic signed hashing projection. It is
    *not* semantic - it will not connect "hippocampal atrophy" with "medial
    temporal lobe volume loss" - but it preserves cosine similarity under the
    Johnson-Lindenstrauss lemma, so it behaves like a dense TF-IDF retriever and
    keeps the whole repository runnable on a bare Python install. Every result
    produced with it is labelled ``lexical`` so it is never mistaken for a
    semantic-embedding result.

All encoders return L2-normalised vectors, so cosine similarity is just a dot
product.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .config import Config
from .utils import ensure_dir, get_logger

log = get_logger("neurorag.embeddings")

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    """Tokenizer that keeps hyphenated technical terms intact.

    "Swin-UNETR", "T1w", "p-value" and "MCI-to-AD" must survive as single
    tokens; splitting them destroys exactly the vocabulary a biomedical
    retriever depends on.
    """
    return _TOKEN.findall((text or "").lower())


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Content-addressed cache keyed on ``sha256(model + text)``.

    Re-running an index build, or sweeping chunk sizes that overlap heavily,
    re-encodes the same text repeatedly. Caching makes that free and guarantees
    identical text always maps to an identical vector.
    """

    def __init__(self, directory: str | Path, model_name: str):
        self.dir = ensure_dir(directory)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)[:80]
        self.keys_path = self.dir / f"{safe}.keys"
        self.vec_path = self.dir / f"{safe}.npy"
        self.hits = self.misses = 0
        self._store: Dict[str, List[float]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not (self.keys_path.exists() and self.vec_path.exists()):
            return
        try:
            import numpy as np
            keys = self.keys_path.read_text(encoding="utf-8").splitlines()
            vectors = np.load(self.vec_path)
            if len(keys) == vectors.shape[0]:
                self._store = {k: vectors[i].tolist() for i, k in enumerate(keys)}
                log.info("Loaded %d cached embeddings", len(self._store))
        except Exception as exc:
            log.warning("Ignoring unreadable embedding cache (%s)", exc)
            self._store = {}

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x1f{text}".encode()).hexdigest()[:32]

    def get_many(self, keys: Sequence[str]) -> List[Optional[List[float]]]:
        out = []
        for k in keys:
            vec = self._store.get(k)
            self.hits += vec is not None
            self.misses += vec is None
            out.append(vec)
        return out

    def put(self, key: str, vector: Sequence[float]) -> None:
        self._store[key] = list(vector)
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty or not self._store:
            return
        try:
            import numpy as np
            keys = sorted(self._store)
            matrix = np.asarray([self._store[k] for k in keys], dtype="float32")
            # The temp file must already end in .npy: np.save appends the
            # extension when it is missing, which would break the atomic rename.
            tmp = self.vec_path.with_name(self.vec_path.stem + ".tmp.npy")
            np.save(tmp, matrix)
            tmp.replace(self.vec_path)
            self.keys_path.write_text("\n".join(keys), encoding="utf-8")
            self._dirty = False
            log.info("Cached %d embeddings (%d hits / %d misses this run)",
                     len(keys), self.hits, self.misses)
        except Exception as exc:
            log.warning("Could not persist embedding cache: %s", exc)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def _normalize(vectors) -> List[List[float]]:
    import numpy as np
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).tolist()


class BaseEmbedder:
    name = "base"
    is_semantic = False
    dim: Optional[int] = None

    def __init__(self, config: Config):
        self.config = config

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def encode_cached(self, texts: Sequence[str], cache: Optional[EmbeddingCache] = None) -> List[List[float]]:
        """``encode`` with a content-addressed cache in front of it."""
        if cache is None:
            return self.encode(texts)
        keys = [cache.key(self.name, t) for t in texts]
        vectors = cache.get_many(keys)
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            fresh = self.encode([texts[i] for i in missing])
            for i, vec in zip(missing, fresh):
                vectors[i] = vec
                cache.put(keys[i], vec)
            cache.flush()
        return [v for v in vectors if v is not None]


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local Hugging Face bi-encoder."""

    is_semantic = True

    def __init__(self, config: Config, model_name: Optional[str] = None):
        super().__init__(config)
        self.model_name = model_name or config.embed_model
        self.name = self.model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            device = None
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            log.info("Loading embedding model '%s' on %s ...", self.model_name, device)
            self._model = SentenceTransformer(self.model_name, device=device)
            try:
                self._model.max_seq_length = self.config.embed_max_seq_length
                self.dim = int(self._model.get_sentence_embedding_dimension())
            except Exception:
                pass
            log.info("Loaded '%s' (dim=%s)", self.model_name, self.dim)
        return self._model

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.config.embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _normalize(vectors)


class OpenAIEmbedder(BaseEmbedder):
    """Hosted embeddings via the OpenAI-compatible REST API (no SDK needed)."""

    is_semantic = True

    def __init__(self, config: Config, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(config)
        self.name = f"openai:{config.openai_embed_model}"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        out: List[List[float]] = []
        size = max(1, self.config.embed_batch_size)
        for start in range(0, len(texts), size):
            batch = list(texts[start: start + size])
            body = json.dumps({"model": self.config.openai_embed_model, "input": batch}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/embeddings", data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                payload = json.loads(resp.read().decode())
            ordered = sorted(payload["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in ordered)
        return _normalize(out)


class LexicalEmbedder(BaseEmbedder):
    """Deterministic IDF-weighted hashed bag-of-words encoder (offline fallback).

    Per text: tokenise -> sublinear TF ``1 + log(tf)`` -> multiply by corpus IDF
    -> project into ``dim`` buckets by signed hashing -> L2-normalise.

    The projection is what makes this a *vector* encoder rather than a sparse
    one, and signed feature hashing approximately preserves cosine similarity, so
    it behaves like dense TF-IDF. It is lexical, not semantic, and results
    produced with it are always labelled as such.
    """

    is_semantic = False

    def __init__(self, config: Config, dim: int = 512):
        super().__init__(config)
        self.name = "lexical"
        self.dim = dim
        self.idf: Dict[str, float] = {}

    def fit(self, texts: Iterable[str]) -> "LexicalEmbedder":
        """Compute smoothed IDF over the corpus (sklearn's convention)."""
        texts = list(texts)
        df: Dict[str, int] = {}
        for text in texts:
            for token in set(tokenize(text)):
                df[token] = df.get(token, 0) + 1
        n = max(len(texts), 1)
        self.idf = {tok: math.log((1 + n) / (1 + count)) + 1.0 for tok, count in df.items()}
        log.info("LexicalEmbedder fitted on %d texts (%d unique tokens, dim=%d)",
                 len(texts), len(self.idf), self.dim)
        return self

    def _bucket(self, token: str):
        digest = hashlib.md5(token.encode()).digest()
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return int.from_bytes(digest[:4], "big") % self.dim, sign

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            counts: Dict[str, int] = {}
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, tf in counts.items():
                idx, sign = self._bucket(tok)
                vec[idx] += sign * (1.0 + math.log(tf)) * self.idf.get(tok, 1.0)
            out.append(vec)
        return _normalize(out)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def available_memory_gb() -> Optional[float]:
    """Best-effort available memory, honouring cgroup limits.

    Containers often advertise the host's RAM in ``/proc/meminfo`` while
    enforcing a much smaller cgroup cap. Reading only the former is how an index
    build gets OOM-killed halfway through, so we take the minimum of the two.
    """
    values: List[float] = []
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
            if raw and raw != "max":
                n = int(raw)
                if n < (1 << 62):
                    values.append(n / 1024 ** 3)
        except Exception:
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                values.append(int(line.split()[1]) / 1024 ** 2)
                break
    except Exception:
        pass
    return round(min(values), 2) if values else None


def build_embedder(config: Config, cache_dir: Optional[str | Path] = None):
    """Resolve ``embed_backend: auto`` to the best encoder usable right now.

    Preference: local bi-encoder -> OpenAI -> lexical. Every degradation is
    logged loudly, because *which* encoder produced a number is the difference
    between a meaningful benchmark and a meaningless one.
    """
    backend = (config.embed_backend or "auto").lower()

    def with_cache(emb: BaseEmbedder):
        cache = EmbeddingCache(cache_dir, emb.name) if cache_dir else None
        return emb, cache

    if backend == "lexical":
        return with_cache(LexicalEmbedder(config))

    if backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("embed_backend=openai but OPENAI_API_KEY is not set")
        return with_cache(OpenAIEmbedder(config))

    if backend == "sentence_transformers":
        return with_cache(SentenceTransformerEmbedder(config))

    # ---- auto ----
    # A base-size biomedical encoder needs >3 GB; don't attempt it in a small
    # container, step down to a small model instead.
    model = config.embed_model
    budget = available_memory_gb()
    if budget is not None and budget < 2.0 and "base" in model.lower():
        log.warning("Only %.2f GB available; '%s' will likely OOM. Using all-MiniLM-L6-v2.",
                    budget, model)
        model = "all-MiniLM-L6-v2"

    for candidate in [model, "all-MiniLM-L6-v2"]:
        try:
            emb = SentenceTransformerEmbedder(config, model_name=candidate)
            emb._load()
            if candidate != config.embed_model:
                log.warning("Using '%s' (configured model was '%s').", candidate, config.embed_model)
            return with_cache(emb)
        except Exception as exc:
            log.warning("Could not load '%s': %s", candidate, str(exc)[:160])

    if os.environ.get("OPENAI_API_KEY"):
        try:
            emb = OpenAIEmbedder(config)
            emb.encode(["connectivity probe"])
            log.warning("No local model available; using OpenAI embeddings.")
            return with_cache(emb)
        except Exception as exc:
            log.warning("OpenAI embeddings unavailable: %s", str(exc)[:160])

    log.warning("=" * 70)
    log.warning("No semantic encoder available - falling back to the OFFLINE")
    log.warning("lexical encoder (hashed IDF bag-of-words). Retrieval will be")
    log.warning("lexical, not semantic. Install sentence-transformers for real")
    log.warning("embeddings:  pip install sentence-transformers")
    log.warning("=" * 70)
    return with_cache(LexicalEmbedder(config))
