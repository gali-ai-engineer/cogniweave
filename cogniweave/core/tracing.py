"""Structured tracing, token accounting and budget guardrails.

Spans nest correctly across concurrent asyncio tasks via ``contextvars``. A run
can be exported as JSON (for dashboards) or as a Mermaid sequence diagram
(for READMEs and debugging).
"""

from __future__ import annotations

import contextlib
import json
import re
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_current_span: ContextVar[str | None] = ContextVar("cogniweave_span", default=None)


class BudgetExceeded(RuntimeError):
    """Raised when a run exceeds its LLM-call or token budget."""


@dataclass
class Span:
    name: str
    agent: str | None
    start: float
    parent_id: str | None = None
    end: float | None = None
    status: str = "ok"
    attrs: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def duration_ms(self) -> float:
        return round(((self.end or self.start) - self.start) * 1000, 2)


class Tracer:
    def __init__(self, *, max_llm_calls: int | None = None, max_tokens: int | None = None) -> None:
        self.max_llm_calls = max_llm_calls
        self.max_tokens = max_tokens
        self.spans: list[Span] = []
        self.usage: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        self._t0 = time.perf_counter()
        self._listeners: list[Callable[[str, Span], None]] = []

    def now(self) -> float:
        """Seconds since the tracer was created."""
        return time.perf_counter() - self._t0

    _now = now

    def subscribe(self, listener: Callable[[str, Span], None]) -> None:
        """Receive ``("start" | "end" | "event", span)`` as the run happens (used by the live UI)."""
        self._listeners.append(listener)

    def _emit(self, kind: str, span: Span) -> None:
        for listener in self._listeners:
            with contextlib.suppress(Exception):  # an observer must never break the run
                listener(kind, span)

    @asynccontextmanager
    async def span(self, name: str, agent: str | None = None, **attrs: Any) -> AsyncIterator[Span]:
        span = Span(name, agent, self._now(), parent_id=_current_span.get(), attrs=dict(attrs))
        self.spans.append(span)
        token = _current_span.set(span.id)
        self._emit("start", span)
        try:
            yield span
        except BaseException as exc:
            span.status = "error"
            span.attrs["error"] = repr(exc)
            raise
        finally:
            span.end = self._now()
            _current_span.reset(token)
            self._emit("end", span)

    def event(self, name: str, agent: str | None = None, **attrs: Any) -> Span:
        now = self._now()
        span = Span(name, agent, now, parent_id=_current_span.get(), end=now, attrs=dict(attrs))
        self.spans.append(span)
        self._emit("event", span)
        return span

    # --------------------------------------------------------------- budgets
    @property
    def total_calls(self) -> int:
        return sum(u["calls"] for u in self.usage.values())

    @property
    def total_tokens(self) -> int:
        return sum(u["input_tokens"] + u["output_tokens"] for u in self.usage.values())

    def check_budget(self) -> None:
        if self.max_llm_calls is not None and self.total_calls >= self.max_llm_calls:
            raise BudgetExceeded(f"LLM call budget of {self.max_llm_calls} exhausted")
        if self.max_tokens is not None and self.total_tokens >= self.max_tokens:
            raise BudgetExceeded(f"Token budget of {self.max_tokens} exhausted")

    def record_llm(self, agent: str, input_tokens: int, output_tokens: int) -> None:
        u = self.usage[agent]
        u["calls"] += 1
        u["input_tokens"] += input_tokens
        u["output_tokens"] += output_tokens

    # --------------------------------------------------------------- export
    def to_dict(self) -> dict[str, Any]:
        return {
            "spans": [asdict(s) | {"duration_ms": s.duration_ms} for s in self.spans],
            "usage": dict(self.usage),
            "totals": {"llm_calls": self.total_calls, "tokens": self.total_tokens},
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    def usage_table(self) -> str:
        rows = ["| agent | calls | input tokens | output tokens |", "|---|---:|---:|---:|"]
        for agent, u in sorted(self.usage.items()):
            rows.append(f"| {agent} | {u['calls']} | {u['input_tokens']} | {u['output_tokens']} |")
        rows.append(f"| **total** | **{self.total_calls}** | | **{self.total_tokens}** tokens |")
        return "\n".join(rows)

    def to_mermaid(self, max_label: int = 48) -> str:
        """Sequence diagram of agent runs, tool calls, reflections and scheduling."""

        def clean(text: Any) -> str:
            text = re.sub(r"[;#:<>\"`{}\[\]]", " ", str(text))
            text = " ".join(text.split())
            return text[:max_label] + ("…" if len(text) > max_label else "")

        timeline: list[tuple[float, int, str]] = []
        participants: list[str] = ["Orchestrator"]
        for s in self.spans:
            if s.agent and s.agent not in participants and s.attrs.get("kind") in {"agent_run", "reflection", "tool"}:
                participants.append(s.agent)
            kind = s.attrs.get("kind")
            if kind == "agent_run":
                timeline.append((s.start, 0, f"Orchestrator->>{s.agent}: {clean(s.attrs.get('task', s.name))}"))
                conf = s.attrs.get("confidence")
                label = "error" if s.status == "error" else f"done conf {conf}"
                timeline.append((s.end or s.start, 2, f"{s.agent}-->>Orchestrator: {label}"))
            elif kind == "tool":
                timeline.append((s.start, 1, f"{s.agent}->>{s.agent}: tool {clean(s.attrs.get('tool'))}"))
            elif kind == "reflection":
                timeline.append((s.start, 1, f"Note over {s.agent}: reflect - {clean(s.attrs.get('lesson', ''))}"))
            elif kind == "note":
                timeline.append((s.start, 1, f"Note over Orchestrator: {clean(s.attrs.get('text', s.name))}"))
        lines = ["sequenceDiagram"] + [f"    participant {p}" for p in participants]
        lines += [f"    {line}" for _, _, line in sorted(timeline, key=lambda t: (t[0], t[1]))]
        return "\n".join(lines)
