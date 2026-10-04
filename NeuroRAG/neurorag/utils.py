"""Small shared helpers: logging, JSONL IO, timing, text normalisation, hashing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "neurorag", level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, (level or os.environ.get("NEURORAG_LOG", "INFO")).upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


log = get_logger()


def set_seed(seed: int = 42) -> None:
    """Seed everything that could make a run non-reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


@contextmanager
def timed(label: str, sink: Optional[Dict[str, float]] = None) -> Iterator[None]:
    """Record elapsed milliseconds for a block into ``sink[label]``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        if sink is not None:
            sink[label] = sink.get(label, 0.0) + elapsed


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | os.PathLike, rows: Iterable[Dict[str, Any]]) -> int:
    p = Path(path)
    ensure_dir(p.parent)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | os.PathLike) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist.")
    with p.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: str | os.PathLike, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | os.PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_hash(*parts: Any, length: int = 12) -> str:
    """Deterministic short hash, used for reproducible ids."""
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:length]


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Collapse space runs and excessive newlines without merging paragraphs."""
    text = (text or "").replace("\u00a0", " ").replace("\u2028", "\n").replace("\u2029", "\n")
    return _NL.sub("\n\n", _WS.sub(" ", text)).strip()


def truncate(text: str, max_chars: int, ellipsis: str = " [...]") -> str:
    if len(text or "") <= max_chars:
        return text or ""
    return text[: max_chars - len(ellipsis)].rstrip() + ellipsis


def split_sentences(text: str) -> List[str]:
    """Sentence splitter that survives biomedical text.

    A naive ``split('.')`` shreds "e.g.", "et al.", "Fig. 3" and "p < 0.05", so
    those are protected before splitting and restored afterwards.
    """
    if not text:
        return []
    protected = text
    placeholders: Dict[str, str] = {}
    for i, abbr in enumerate([
        "e.g.", "i.e.", "et al.", "vs.", "cf.", "Fig.", "Figs.", "No.", "Vol.",
        "approx.", "Eq.", "Tab.", "Dr.", "Prof.",
    ]):
        token = f"\x00A{i}\x00"
        placeholders[token] = abbr
        protected = protected.replace(abbr, token)
    protected = re.sub(r"(?<=\d)\.(?=\d)", "\x00D\x00", protected)

    out: List[str] = []
    for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"])", protected):
        part = part.replace("\x00D\x00", ".").strip()
        for token, abbr in placeholders.items():
            part = part.replace(token, abbr)
        if part:
            out.append(part)
    return out


def dedupe(items: Iterable[Any]) -> List[Any]:
    seen, out = set(), []
    for item in items:
        key = item if isinstance(item, (str, int, float)) else repr(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out
