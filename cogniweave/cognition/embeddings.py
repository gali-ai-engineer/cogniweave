"""Dependency-free text embeddings.

``HashingEmbedder`` uses the hashing trick over unigrams + bigrams with signed
buckets and sublinear TF. It is deterministic (blake2b, not Python's salted
``hash``) so memory files are portable across processes. Swap in any object
with an ``embed(text) -> list[float]`` method (e.g. a sentence-transformers
wrapper) for production-grade retrieval.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    (
        "a an the and or of to in on for is are be was were with as by at it this that from into than then "
        "their its can do does not what which who how why when about relevant"
    ).split()
)


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


def _stem(tok: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(tok) > len(suffix) + 3 and tok.endswith(suffix):
            return tok[: -len(suffix)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        toks = tokenize(text)
        feats = Counter(toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)])
        vec = [0.0] * self.dim
        for feat, count in feats.items():
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += (1.0 if h >> 63 else -1.0) * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
