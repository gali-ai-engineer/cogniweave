from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.tracing import Tracer
from .plan import Plan, Review


@dataclass
class RunResult:
    pattern: str
    goal: str
    answer: str
    tracer: Tracer
    review: Review | None = None
    plan: Plan | None = None
    revisions: int = 0
    blackboard: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"pattern={self.pattern}",
            f"llm_calls={self.tracer.total_calls}",
            f"tokens={self.tracer.total_tokens}",
        ]
        if self.review:
            parts.append(f"review_score={self.review.score}")
        parts.append(f"revisions={self.revisions}")
        return " | ".join(parts)
