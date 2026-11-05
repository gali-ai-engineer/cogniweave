"""Plans are validated DAGs of role-assigned tasks.

Pure-Python validation (no compiled dependencies): LLM output is the place
where malformed data actually appears, so every field is checked explicitly
and errors are written to be fed straight back to the model for self-repair.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


def _require_str(data: dict[str, Any], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, (str, int)) or str(value).strip() == "":
        raise ValueError(f"{where}: field {key!r} is required and must be a non-empty string")
    return str(value)


@dataclass
class PlanTask:
    id: str
    description: str
    role: str
    depends_on: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> PlanTask:
        if not isinstance(data, dict):
            raise ValueError("each task must be a JSON object")
        where = f"task {data.get('id', '?')}"
        deps = data.get("depends_on") or []
        if not isinstance(deps, list):
            raise ValueError(f"{where}: depends_on must be a list of task ids")
        return cls(
            _require_str(data, "id", where),
            _require_str(data, "description", where),
            _require_str(data, "role", where),
            [str(d) for d in deps],
        )


@dataclass
class Plan:
    goal: str
    tasks: list[PlanTask]

    def __post_init__(self) -> None:
        self._validate_dag()

    @classmethod
    def from_dict(cls, data: Any) -> Plan:
        if not isinstance(data, dict):
            raise ValueError("plan must be a JSON object")
        tasks = data.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("plan must contain a non-empty 'tasks' list")
        return cls(str(data.get("goal", "")), [PlanTask.from_dict(t) for t in tasks])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def _validate_dag(self) -> None:
        if not self.tasks:
            raise ValueError("plan must contain at least one task")
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


@dataclass
class Review:
    score: float
    verdict: str = "revise"
    issues: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("score must be a number")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        self.score = float(self.score)

    @classmethod
    def from_dict(cls, data: Any) -> Review:
        if not isinstance(data, dict):
            raise ValueError("review must be a JSON object")
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        return cls(data.get("score"), str(data.get("verdict", "revise")), [str(i) for i in issues])

    @classmethod
    def from_json(cls, text: str) -> Review:
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def parse(cls, text: str) -> Review:
        """Lenient parse of an LLM reply; unparseable critiques count as unreviewed."""
        try:
            return cls.from_dict(extract_json(text))
        except Exception:
            return cls(score=0.5, verdict="revise", issues=["Critic output was not valid JSON; treat as unreviewed."])


def _issue_key(issue: str) -> str:
    """Compare issues across rounds while ignoring counts ("8 of 9" vs "2 of 2")."""
    return re.sub(r"\d+", "#", issue.lower())[:60]


def revision_log(reviews: list[Review]) -> list[dict[str, Any]]:
    """For each review after the first: which earlier issues were fixed, which remain, which are new.

    This is how cogniweave tells a real fix from a rewording: an issue only counts
    as fixed when the critic stops raising it.
    """
    log = []
    for i in range(1, len(reviews)):
        before = {_issue_key(x): x for x in reviews[i - 1].issues}
        after = {_issue_key(x): x for x in reviews[i].issues}
        log.append(
            {
                "revision": i,
                "score_before": reviews[i - 1].score,
                "score_after": reviews[i].score,
                "fixed": [v for k, v in before.items() if k not in after],
                "still_open": [v for k, v in after.items() if k in before],
                "new": [v for k, v in after.items() if k not in before],
            }
        )
    return log


def stalled(log: list[dict[str, Any]]) -> bool:
    """True when the latest revision fixed nothing and did not raise the score."""
    return bool(log) and not log[-1]["fixed"] and log[-1]["score_after"] <= log[-1]["score_before"]
