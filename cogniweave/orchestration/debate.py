"""Multi-agent debate (Du et al., 2023) with a judge.

Debaters with different personas answer independently from shared retrieved
evidence, then see each other's positions and rebut for ``rounds``. A judge
aggregates the final round into a verdict. Debate trades extra calls for
reduced single-model bias and better-calibrated answers on contested questions.
"""

from __future__ import annotations

import asyncio

from ..crew import Crew
from .result import RunResult


class DebateOrchestrator:
    def __init__(self, crew: Crew, *, rounds: int = 2, evidence_k: int = 8) -> None:
        self.crew = crew
        self.tracer = crew.tracer
        self.rounds = rounds
        self.evidence_k = evidence_k
        self.debaters = [a for a in crew.agents.values() if a.role == "debater"]
        self.judge = crew["judge"]

    async def run(self, question: str) -> RunResult:
        board = self.crew.blackboard
        async with self.tracer.span("debate.run", goal=question):
            await board.post("goal", question, "user")
            hits = self.crew.memory.semantic.search(question, k=self.evidence_k)
            evidence = "\n".join(f"[{c.id}] {c.text}" for c, _ in hits)
            transcript: dict[str, str] = {}
            for rnd in range(1, self.rounds + 1):
                peers = "\n\n".join(f"### {name}\n{text}" for name, text in transcript.items())
                outputs = await asyncio.gather(
                    *(
                        d.run(question, goal=question, context=evidence, feedback=peers, metadata={"round": rnd})
                        for d in self.debaters
                    )
                )
                transcript = {o.agent: o.content for o in outputs}
                for o in outputs:
                    await board.post("argument", o.content, o.agent, confidence=o.confidence, tags=(f"round-{rnd}",))
                self.tracer.event("debate.round", kind="note", text=f"round {rnd} complete")
            final_args = "\n\n".join(f"### {name}\n{text}" for name, text in transcript.items())
            verdict = await self.judge.run("Deliver a verdict on the debate.", goal=question, context=final_args)
            await board.post("verdict", verdict.content, self.judge.name, confidence=verdict.confidence)
        return RunResult(
            "debate",
            question,
            verdict.content,
            self.tracer,
            blackboard=board.snapshot(),
            extra={"rounds": self.rounds, "final_arguments": transcript},
        )
