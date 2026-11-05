"""Shared blackboard: the team's common workspace.

Every pattern writes its intermediate artefacts here (plans, findings,
drafts, critiques), giving one inspectable record of how an answer was built.
The blackboard *pattern* additionally uses it as its only coordination channel.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

Subscriber = Callable[["Entry"], Awaitable[None] | None]


@dataclass(frozen=True)
class Entry:
    id: int
    kind: str  # goal | question | finding | analysis | draft | critique | plan | result ...
    content: str
    author: str
    confidence: float = 1.0
    refs: tuple[int, ...] = ()
    tags: frozenset[str] = field(default_factory=frozenset)
    ts: float = field(default_factory=time.time)


class Blackboard:
    def __init__(self) -> None:
        self._entries: list[Entry] = []
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()
        self._subs: dict[str, list[Subscriber]] = {}

    async def post(
        self,
        kind: str,
        content: str,
        author: str,
        *,
        confidence: float = 1.0,
        refs: tuple[int, ...] = (),
        tags: tuple[str, ...] = (),
    ) -> Entry:
        async with self._lock:
            entry = Entry(next(self._ids), kind, content, author, confidence, tuple(refs), frozenset(tags))
            self._entries.append(entry)
        for cb in self._subs.get(kind, []) + self._subs.get("*", []):
            result = cb(entry)
            if asyncio.iscoroutine(result):
                await result
        return entry

    def subscribe(self, kind: str, callback: Subscriber) -> None:
        """Register a callback for entries of ``kind`` (``"*"`` for all)."""
        self._subs.setdefault(kind, []).append(callback)

    def query(
        self,
        kind: str | None = None,
        *,
        author: str | None = None,
        ref: int | None = None,
        where: Callable[[Entry], bool] | None = None,
    ) -> list[Entry]:
        return [
            e
            for e in self._entries
            if (kind is None or e.kind == kind)
            and (author is None or e.author == author)
            and (ref is None or ref in e.refs)
            and (where is None or where(e))
        ]

    def latest(self, kind: str) -> Entry | None:
        found = self.query(kind)
        return found[-1] if found else None

    def __len__(self) -> int:
        return len(self._entries)

    def snapshot(self) -> list[dict[str, Any]]:
        return [asdict(e) | {"tags": sorted(e.tags)} for e in self._entries]
