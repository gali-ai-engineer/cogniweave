"""Blackboard orchestration (Hearsay-II style, Erman et al. 1980).

There is no fixed plan. Knowledge sources watch the shared board and report an
*activation* score for how useful they could be right now. A control loop
repeatedly schedules the most activated source, which reads the board and
posts new entries. Control emerges from the data: questions trigger research,
complete research triggers analysis, a draft triggers critique, a "revise"
critique re-triggers the writer.
"""

from __future__ import annotations

import asyncio

from ..core.agent import CognitiveAgent
from ..core.blackboard import Blackboard, Entry
from ..crew import Crew
from .plan import Review, extract_json
from .result import RunResult


class KnowledgeSource:
    name = "ks"

    def __init__(self, agent: CognitiveAgent) -> None:
        self.agent = agent

    def activation(self, board: Blackboard) -> float:
        raise NotImplementedError

    async def contribute(self, board: Blackboard, goal: str) -> None:
        raise NotImplementedError


class Decomposer(KnowledgeSource):
    name = "decomposer"

    def activation(self, board: Blackboard) -> float:
        return 1.0 if not board.query("question") else 0.0

    async def contribute(self, board: Blackboard, goal: str) -> None:
        out = await self.agent.run(
            'Decompose the goal into 2-4 research questions. Reply with JSON only: {"questions": [str]}.',
            goal=goal,
            metadata={"phase": "questions"},
        )
        try:
            questions = [str(q) for q in extract_json(out.content)["questions"]][:4]
        except Exception:
            questions = [goal]
        for q in questions or [goal]:
            await board.post("question", q, self.agent.name)


class Researcher(KnowledgeSource):
    name = "researcher"

    @staticmethod
    def open_questions(board: Blackboard) -> list[Entry]:
        return [q for q in board.query("question") if not board.query("finding", ref=q.id)]

    def activation(self, board: Blackboard) -> float:
        return 0.9 if self.open_questions(board) else 0.0

    async def contribute(self, board: Blackboard, goal: str) -> None:
        async def answer(q: Entry) -> None:
            out = await self.agent.run(q.content, goal=goal)
            self.agent.learn(out.strategy, out.confidence)
            await board.post("finding", out.content, self.agent.name, confidence=out.confidence, refs=(q.id,))

        await asyncio.gather(*(answer(q) for q in self.open_questions(board)))  # opportunistic parallelism


class Analyst(KnowledgeSource):
    name = "analyst"

    def activation(self, board: Blackboard) -> float:
        findings = board.query("finding")
        latest = board.latest("analysis")
        stale = latest is None or any(f.id > latest.id for f in findings)
        return 0.8 if findings and not Researcher.open_questions(board) and stale else 0.0

    async def contribute(self, board: Blackboard, goal: str) -> None:
        findings = board.query("finding")
        ctx = "\n\n".join(f"### Finding {f.id} (confidence {f.confidence})\n{f.content}" for f in findings)
        out = await self.agent.run("Synthesise the findings into key findings and trade-offs.", goal=goal, context=ctx)
        await board.post(
            "analysis", out.content, self.agent.name, confidence=out.confidence, refs=tuple(f.id for f in findings)
        )


class Writer(KnowledgeSource):
    name = "writer"

    def __init__(self, agent: CognitiveAgent, max_drafts: int = 3) -> None:
        super().__init__(agent)
        self.max_drafts = max_drafts

    def _pending_critique(self, board: Blackboard) -> Entry | None:
        draft, critique = board.latest("draft"), board.latest("critique")
        if draft and critique and draft.id in critique.refs and "revise" in critique.tags:
            return critique
        return None

    def activation(self, board: Blackboard) -> float:
        analysis = board.latest("analysis")
        drafts = board.query("draft")
        if analysis is None or len(drafts) >= self.max_drafts:
            return 0.0
        if not drafts or drafts[-1].id < analysis.id:
            return 0.7
        return 0.7 if self._pending_critique(board) else 0.0

    async def contribute(self, board: Blackboard, goal: str) -> None:
        analysis = board.latest("analysis")
        task = "Write a concise, cited markdown report answering the goal"
        critique = self._pending_critique(board)
        feedback = critique.content if critique else ""
        if critique:
            await self.agent.reflect(task, feedback)
        out = await self.agent.run(task, goal=goal, context=analysis.content if analysis else "", feedback=feedback)
        await board.post(
            "draft",
            out.content,
            self.agent.name,
            confidence=out.confidence,
            refs=(analysis.id,) if analysis else (),
            tags=(out.strategy,),
        )


class Critic(KnowledgeSource):
    name = "critic"

    def activation(self, board: Blackboard) -> float:
        draft = board.latest("draft")
        return 0.75 if draft and not board.query("critique", ref=draft.id) else 0.0

    async def contribute(self, board: Blackboard, goal: str) -> None:
        draft = board.latest("draft")
        assert draft is not None
        out = await self.agent.run("Review the draft report against the goal.", goal=goal, context=draft.content)
        review = Review.parse(out.content)
        await board.post(
            "critique",
            review.to_json(),
            self.agent.name,
            confidence=review.score,
            refs=(draft.id,),
            tags=(review.verdict,),
        )


class BlackboardOrchestrator:
    def __init__(self, crew: Crew, *, max_cycles: int = 12, accept_threshold: float = 0.8, max_drafts: int = 3):
        self.crew = crew
        self.tracer = crew.tracer
        self.max_cycles = max_cycles
        self.accept_threshold = accept_threshold
        self.sources: list[KnowledgeSource] = [
            Decomposer(crew["planner"]),
            Researcher(crew["researcher"]),
            Analyst(crew["analyst"]),
            Writer(crew["writer"], max_drafts=max_drafts),
            Critic(crew["critic"]),
        ]

    def _accepted(self, board: Blackboard) -> Review | None:
        draft, critique = board.latest("draft"), board.latest("critique")
        if draft and critique and draft.id in critique.refs:
            review = Review.from_json(critique.content)
            if review.score >= self.accept_threshold:
                return review
        return None

    async def run(self, goal: str) -> RunResult:
        board = self.crew.blackboard
        schedule: list[str] = []
        async with self.tracer.span("blackboard.run", goal=goal):
            await board.post("goal", goal, "user")
            for cycle in range(1, self.max_cycles + 1):
                if self._accepted(board):
                    break
                scored = sorted(((ks.activation(board), ks) for ks in self.sources), key=lambda p: p[0], reverse=True)
                activation, best = scored[0]
                if activation <= 0:
                    break
                schedule.append(best.name)
                self.tracer.event("schedule", kind="note", text=f"cycle {cycle} {best.name} a={activation}")
                await best.contribute(board, goal)

        draft = board.latest("draft")
        critique = board.latest("critique")
        review = Review.from_json(critique.content) if critique else None
        if draft and review:
            self.crew["writer"].learn(next(iter(draft.tags), ""), review.score)
        return RunResult(
            "blackboard",
            goal,
            draft.content if draft else "(no draft produced)",
            self.tracer,
            review=review,
            revisions=max(0, len(board.query("draft")) - 1),
            blackboard=board.snapshot(),
            extra={"schedule": schedule},
        )
