"""Plans are validated DAGs of role-assigned tasks."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, model_validator


class PlanTask(BaseModel):
    id: str
    description: str
    role: str
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    goal: str
    tasks: list[PlanTask] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dag(self) -> Plan:
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task ids")
        known = set(ids)
        for t in self.tasks:
            missing = set(t.depends_on) - known
            if missing:
                raise ValueError(f"task {t.id} depends on unknown task(s) {sorted(missing)}")
            if t.id in t.depends_on:
                raise ValueError(f"task {t.id} depends on itself")
        self.topo_order()  # raises on cycles
        return self

    def topo_order(self) -> list[PlanTask]:
        """Kahn's algorithm; stable with respect to the declared order."""
        indegree = {t.id: len(t.depends_on) for t in self.tasks}
        by_id = {t.id: t for t in self.tasks}
        ready = [t.id for t in self.tasks if indegree[t.id] == 0]
        order: list[PlanTask] = []
        while ready:
            tid = ready.pop(0)
            order.append(by_id[tid])
            for t in self.tasks:
                if tid in t.depends_on:
                    indegree[t.id] -= 1
                    if indegree[t.id] == 0:
                        ready.append(t.id)
        if len(order) != len(self.tasks):
            stuck = sorted(set(by_id) - {t.id for t in order})
            raise ValueError(f"dependency cycle among tasks {stuck}")
        return order

    def sinks(self) -> list[PlanTask]:
        depended = {d for t in self.tasks for d in t.depends_on}
        return [t for t in self.tasks if t.id not in depended]

    def levels(self) -> list[list[PlanTask]]:
        """Group tasks into waves that can run in parallel."""
        level: dict[str, int] = {}
        for t in self.topo_order():
            level[t.id] = 1 + max((level[d] for d in t.depends_on), default=-1)
        waves: list[list[PlanTask]] = [[] for _ in range(max(level.values()) + 1)]
        for t in self.tasks:
            waves[level[t.id]].append(t)
        return waves

    def to_mermaid(self) -> str:
        lines = ["flowchart LR"]
        for t in self.tasks:
            label = re.sub(r"[\"\[\]()]", "", t.description)[:40]
            lines.append(f'    {t.id}["{t.id} · {t.role}<br/>{label}"]')
        lines += [f"    {d} --> {t.id}" for t in self.tasks for d in t.depends_on]
        return "\n".join(lines)


def extract_json(text: str) -> Any:
    """Pull the first JSON object out of an LLM reply (fenced or bare)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model output")
        candidate = text[start : end + 1]
    return json.loads(candidate)


class Review(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    verdict: str = "revise"
    issues: list[str] = Field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> Review:
        try:
            return cls.model_validate(extract_json(text))
        except Exception:
            return cls(score=0.5, verdict="revise", issues=["Critic output was not valid JSON; treat as unreviewed."])
