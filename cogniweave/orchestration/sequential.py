"""Sequential pipeline: the baseline every other pattern is compared against."""

from __future__ import annotations

from ..crew import Crew
from .result import RunResult

DEFAULT_PIPELINE = (
    ("researcher", "Research the question and gather cited evidence"),
    ("analyst", "Synthesise the research into key findings and trade-offs, keeping citations"),
    ("writer", "Write a concise, cited markdown report answering the goal"),
)


class SequentialOrchestrator:
    def __init__(self, crew: Crew, pipeline: tuple[tuple[str, str], ...] = DEFAULT_PIPELINE) -> None:
        self.crew = crew
        self.tracer = crew.tracer
        self.pipeline = pipeline

    async def run(self, goal: str) -> RunResult:
        context = ""
        async with self.tracer.span("sequential.run", goal=goal):
            await self.crew.blackboard.post("goal", goal, "user")
            for role, task in self.pipeline:
                task_text = f"{task}: {goal}" if role == "researcher" else task
                out = await self.crew[role].run(task_text, goal=goal, context=context)
                await self.crew.blackboard.post("result", out.content, out.agent, confidence=out.confidence)
                context = out.content
        return RunResult("sequential", goal, context, self.tracer, blackboard=self.crew.blackboard.snapshot())
