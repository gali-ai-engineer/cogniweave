"""Factory for the research crew and debate panel used by the demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .cognition.memory import MemorySystem
from .core.agent import CognitiveAgent
from .core.blackboard import Blackboard
from .core.tools import ToolRegistry, calculator, make_corpus_search
from .core.tracing import Tracer
from .llm.base import LLMClient

DEFAULT_CORPUS = Path(__file__).parent / "data" / "corpus"

ROLE_INSTRUCTIONS = {
    "planner": (
        "Decompose the goal into a small DAG of tasks. Reply with JSON only: "
        '{"goal": str, "tasks": [{"id": str, "role": str, "description": str, "depends_on": [str]}]}. '
        "Independent research tasks must not depend on each other so they can run in parallel."
    ),
    "researcher": (
        "Gather evidence with your tools. Answer with 'FINDINGS:' followed by bullets, each ending with its "
        "source in square brackets exactly as the tool returned it, e.g. '- claim [doc.md#2]'. Never invent sources."
    ),
    "analyst": (
        "Synthesise the provided findings into '## Analysis' with '### Key findings' and '### Trade-offs and risks'. "
        "Keep every bullet's [source] citation. Do not add claims without a source."
    ),
    "writer": (
        "Write a concise markdown report: a '# Title', '## Executive summary', '## Key findings', '## Trade-offs', "
        "and '## Sources'. Keep inline [source] citations from the context. Apply any lessons and feedback given."
    ),
    "critic": (
        "Review the draft in <context>. Reply with JSON only: "
        '{"score": float 0-1, "verdict": "accept"|"revise", "issues": [str]}. '
        "Check that: claims carry inline [source] citations; every source listed under Sources is cited in the "
        "body; most findings actually address the question; and the Limitations section names gaps specific "
        "to this answer rather than generic caveats. A reworded draft with the same problems must not pass."
    ),
    "debater": (
        "Argue your persona's position on the question using the evidence in <context>. Format: 'POSITION: ...', "
        "'EVIDENCE:' bullets with [source] citations, and in later rounds a 'REBUTTAL:' addressing peers in <feedback>."
    ),
    "judge": (
        "Weigh the debaters' arguments in <context>. Produce '## Verdict' with 'Final answer: ...', the positions "
        "considered, and the strongest cited evidence."
    ),
}

ROLE_STRATEGIES = {
    "researcher": ["breadth-first: survey several sources", "depth-first: exhaust the most relevant source"],
    "writer": ["structured report with headed sections", "answer-first: lead with a one-paragraph verdict"],
}


@dataclass
class Crew:
    agents: dict[str, CognitiveAgent]
    memory: MemorySystem
    tracer: Tracer
    blackboard: Blackboard = field(default_factory=Blackboard)

    def __getitem__(self, role: str) -> CognitiveAgent:
        return self.agents[role]

    @property
    def roles(self) -> list[str]:
        return list(self.agents)


def _memory_with_corpus(memory: MemorySystem | None, corpus_dir: str | Path | None) -> MemorySystem:
    memory = memory or MemorySystem()
    if not memory.semantic.chunks:
        memory.semantic.load_dir(corpus_dir or DEFAULT_CORPUS)
    return memory


def build_research_crew(
    llm: LLMClient,
    *,
    memory: MemorySystem | None = None,
    tracer: Tracer | None = None,
    corpus_dir: str | Path | None = None,
) -> Crew:
    memory = _memory_with_corpus(memory, corpus_dir)
    tracer = tracer or Tracer()
    research_tools = ToolRegistry([make_corpus_search(memory.semantic), calculator])
    agents = {
        role: CognitiveAgent(
            role,
            role,
            llm,
            instructions=ROLE_INSTRUCTIONS[role],
            tools=research_tools if role == "researcher" else None,
            memory=memory,
            tracer=tracer,
            strategies=ROLE_STRATEGIES.get(role),
        )
        for role in ("planner", "researcher", "analyst", "writer", "critic")
    }
    return Crew(agents, memory, tracer)


def build_debate_panel(
    llm: LLMClient,
    *,
    personas: tuple[str, ...] = ("advocate", "skeptic", "pragmatist"),
    memory: MemorySystem | None = None,
    tracer: Tracer | None = None,
    corpus_dir: str | Path | None = None,
) -> Crew:
    memory = _memory_with_corpus(memory, corpus_dir)
    tracer = tracer or Tracer()
    agents = {
        p: CognitiveAgent(
            p, "debater", llm, instructions=ROLE_INSTRUCTIONS["debater"], memory=memory, tracer=tracer, persona=p
        )
        for p in personas
    }
    agents["judge"] = CognitiveAgent(
        "judge", "judge", llm, instructions=ROLE_INSTRUCTIONS["judge"], memory=memory, tracer=tracer
    )
    return Crew(agents, memory, tracer)
