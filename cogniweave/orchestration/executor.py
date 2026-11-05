"""Concurrent DAG execution with retries and metacognitive follow-ups."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..core.agent import AgentOutput, CognitiveAgent
from ..core.blackboard import Blackboard
from ..core.tracing import BudgetExceeded, Tracer
from .plan import Plan, PlanTask


@dataclass
class TaskResult:
    task: PlanTask
    output: AgentOutput | None
    context: str
    error: str | None = None
    attempts: int = 1
    followed_up: bool = False

    @property
    def ok(self) -> bool:
        return self.output is not None

    @property
    def content(self) -> str:
        return self.output.content if self.output else f"(task failed: {self.error})"


class DAGExecutor:
    """Runs each task as soon as its dependencies finish, bounded by a semaphore.

    Metacognition: when a task in ``follow_up_roles`` returns a confidence
    below ``low_confidence``, the agent is asked to deepen its work once and
    the two outputs are merged.
    """

    def __init__(
        self,
        agents: dict[str, CognitiveAgent],
        *,
        tracer: Tracer,
        blackboard: Blackboard | None = None,
        max_concurrency: int = 4,
        max_retries: int = 1,
        low_confidence: float = 0.5,
        follow_up_roles: tuple[str, ...] = ("researcher",),
    ) -> None:
        self.agents = agents
        self.tracer = tracer
        self.blackboard = blackboard
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.low_confidence = low_confidence
        self.follow_up_roles = follow_up_roles

    @staticmethod
    def _context_for(task: PlanTask, results: dict[str, TaskResult]) -> str:
        parts = []
        for d in task.depends_on:
            r = results.get(d)
            if r is None:
                parts.append(f"### Result of {d}\n(dependency did not complete)")
            else:
                parts.append(f"### Result of {d} ({r.task.role}): {r.task.description}\n{r.content}")
        return "\n\n".join(parts)

    async def run(self, plan: Plan, *, deferred_reward: set[str] | None = None) -> dict[str, TaskResult]:
        """``deferred_reward``: task ids whose strategy is credited later (e.g. by a critic)."""
        deferred = deferred_reward or set()
        sem = asyncio.Semaphore(self.max_concurrency)
        done = {t.id: asyncio.Event() for t in plan.tasks}
        results: dict[str, TaskResult] = {}

        async def run_task(task: PlanTask) -> None:
            try:
                for dep in task.depends_on:
                    await done[dep].wait()
                ctx = self._context_for(task, results)
                agent = self.agents[task.role]
                output, error, attempts = None, None, 0
                for attempt in range(1, self.max_retries + 2):
                    attempts = attempt
                    try:
                        async with sem:
                            output = await agent.run(task.description, goal=plan.goal, context=ctx)
                        break
                    except BudgetExceeded:
                        raise
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        self.tracer.event("task.retry", kind="note", text=f"{task.id} failed, retrying", error=error)
                result = TaskResult(task, output, ctx, None if output else error, attempts)

                if output and output.confidence < self.low_confidence and task.role in self.follow_up_roles:
                    self.tracer.event(
                        "metacognition", kind="note", text=f"{task.id} low confidence {output.confidence}, deepening"
                    )
                    async with sem:
                        extra = await agent.run(
                            f"Deepen the investigation with different search terms: {task.description}",
                            goal=plan.goal,
                            context=f"{ctx}\n\nPrevious attempt:\n{output.content}",
                        )
                    result.output, result.followed_up = output.merge(extra), True

                if result.output and task.id not in deferred:
                    agent.learn(result.output.strategy, result.output.confidence)
                if self.blackboard is not None:
                    await self.blackboard.post(
                        "result",
                        result.content,
                        agent.name,
                        confidence=result.output.confidence if result.output else 0.0,
                        tags=(task.id, task.role),
                    )
                results[task.id] = result
            finally:
                done[task.id].set()

        async with self.tracer.span("dag.execute", tasks=len(plan.tasks), waves=len(plan.levels())):
            await asyncio.gather(*(run_task(t) for t in plan.topo_order()))
        return results
