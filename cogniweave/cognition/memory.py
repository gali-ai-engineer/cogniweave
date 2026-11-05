"""Cognitive memory system, following the CoALA taxonomy (Sumers et al., 2023).

* **WorkingMemory** – a per-run, token-budgeted attention buffer. Items compete
  on salience; the least salient are evicted when the budget is exceeded, so
  the prompt always carries the most important material.
* **EpisodicMemory** – timestamped experiences and reflections. Recall scores
  combine relevance, recency and importance, as in Generative Agents
  (Park et al., 2023).
* **SemanticMemory** – chunked knowledge (the research corpus) with vector search.
* **ProceduralMemory** – which *strategy* works for which role, learned online
  with a UCB1 bandit over observed rewards (critic scores / confidence).
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..llm.base import estimate_tokens
from .embeddings import Embedder, HashingEmbedder, cosine


# ------------------------------------------------------------------ working
@dataclass
class WorkingItem:
    content: str
    salience: float
    kind: str
    order: int

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)


class WorkingMemory:
    """Salience-ranked buffer bounded by a token budget (the context window)."""

    def __init__(self, budget_tokens: int = 3000) -> None:
        self.budget_tokens = budget_tokens
        self.items: list[WorkingItem] = []
        self.evicted: list[WorkingItem] = []
        self._order = 0

    def add(self, content: str, *, salience: float = 0.5, kind: str = "note") -> None:
        if not content.strip():
            return
        self._order += 1
        self.items.append(WorkingItem(content, salience, kind, self._order))
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        while self.used_tokens > self.budget_tokens and len(self.items) > 1:
            victim = min(self.items, key=lambda i: (i.salience, -i.order))
            self.items.remove(victim)
            self.evicted.append(victim)
        if self.items and self.used_tokens > self.budget_tokens:  # single oversized item: truncate
            item = self.items[0]
            item.content = item.content[: self.budget_tokens * 4] + "\n…[truncated]"

    @property
    def used_tokens(self) -> int:
        return sum(i.tokens for i in self.items)

    def render(self, kind: str) -> str:
        return "\n".join(i.content for i in sorted(self.items, key=lambda i: i.order) if i.kind == kind)


# ------------------------------------------------------------------ episodic
@dataclass
class Episode:
    agent: str
    task: str
    content: str
    kind: str = "action"  # action | reflection | observation
    importance: float = 0.3
    success: float | None = None
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])


class EpisodicMemory:
    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        half_life_s: float = 7 * 24 * 3600,
        weights: tuple[float, float, float] = (1.0, 0.5, 0.7),
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.half_life_s = half_life_s
        self.w_rel, self.w_rec, self.w_imp = weights
        self.clock = clock
        self.episodes: list[Episode] = []
        self._vecs: dict[str, list[float]] = {}

    def add(self, episode: Episode) -> Episode:
        self.episodes.append(episode)
        self._vecs[episode.id] = self.embedder.embed(f"{episode.task}\n{episode.content}")
        return episode

    def score(self, episode: Episode, query_vec: list[float]) -> float:
        relevance = cosine(query_vec, self._vecs[episode.id])
        age = max(0.0, self.clock() - episode.ts)
        recency = 0.5 ** (age / self.half_life_s)
        return self.w_rel * relevance + self.w_rec * recency + self.w_imp * episode.importance

    def recall(
        self,
        query: str,
        *,
        k: int = 4,
        agent: str | None = None,
        kinds: tuple[str, ...] | None = None,
        min_relevance: float = 0.05,
    ) -> list[tuple[Episode, float]]:
        qv = self.embedder.embed(query)
        pool = [
            e
            for e in self.episodes
            if (agent is None or e.agent == agent)
            and (kinds is None or e.kind in kinds)
            and cosine(qv, self._vecs[e.id]) >= min_relevance
        ]
        scored = sorted(((e, self.score(e, qv)) for e in pool), key=lambda p: p[1], reverse=True)
        return scored[:k]


# ------------------------------------------------------------------ semantic
@dataclass
class Chunk:
    id: str
    source: str
    text: str


class SemanticMemory:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.chunks: list[Chunk] = []
        self._vecs: list[list[float]] = []

    def add_document(self, text: str, source: str) -> int:
        """Split on blank lines, drop markdown headings, index each paragraph."""
        added = 0
        for para in text.split("\n\n"):
            body = " ".join(line for line in para.splitlines() if not line.lstrip().startswith("#")).strip()
            if len(body) < 40:
                continue
            chunk = Chunk(f"{source}#{added}", source, " ".join(body.split()))
            self.chunks.append(chunk)
            self._vecs.append(self.embedder.embed(chunk.text))
            added += 1
        return added

    def load_dir(self, directory: str | Path, pattern: str = "*.md") -> int:
        paths = sorted(Path(directory).glob(pattern))
        return sum(self.add_document(p.read_text(encoding="utf-8"), p.name) for p in paths)

    def search(self, query: str, k: int = 4, min_score: float = 0.05) -> list[tuple[Chunk, float]]:
        qv = self.embedder.embed(query)
        scored = [(c, cosine(qv, v)) for c, v in zip(self.chunks, self._vecs, strict=True)]
        return [p for p in sorted(scored, key=lambda p: p[1], reverse=True)[:k] if p[1] >= min_score]


# ------------------------------------------------------------------ procedural
class ProceduralMemory:
    """Per-role strategy selection with UCB1; rewards are in [0, 1]."""

    def __init__(self, exploration: float = 0.5) -> None:
        self.c = exploration
        self.stats: dict[str, dict[str, list[float]]] = {}  # role -> strategy -> [n, total]

    def select(self, role: str, strategies: list[str]) -> str:
        if not strategies:
            return ""
        table = self.stats.setdefault(role, {})
        for s in strategies:
            if table.get(s, [0, 0])[0] == 0:
                return s
        total = sum(table[s][0] for s in strategies)
        return max(
            strategies,
            key=lambda s: table[s][1] / table[s][0] + self.c * math.sqrt(math.log(total) / table[s][0]),
        )

    def update(self, role: str, strategy: str, reward: float) -> None:
        if not strategy:
            return
        n_total = self.stats.setdefault(role, {}).setdefault(strategy, [0, 0.0])
        n_total[0] += 1
        n_total[1] += max(0.0, min(1.0, reward))

    def mean(self, role: str, strategy: str) -> float | None:
        n, total = self.stats.get(role, {}).get(strategy, [0, 0.0])
        return total / n if n else None


# ------------------------------------------------------------------ facade
class MemorySystem:
    """Shared long-term memory. Working memory is created per agent run."""

    def __init__(self, embedder: Embedder | None = None, clock: Callable[[], float] = time.time) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.episodic = EpisodicMemory(self.embedder, clock=clock)
        self.semantic = SemanticMemory(self.embedder)
        self.procedural = ProceduralMemory()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"episodes": [asdict(e) for e in self.episodic.episodes], "procedural": self.procedural.stats}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        for raw in data.get("episodes", []):
            self.episodic.add(Episode(**raw))
        self.procedural.stats = data.get("procedural", {})
        return True
