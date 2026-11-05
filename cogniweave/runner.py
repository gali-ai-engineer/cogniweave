"""One entry point for every orchestration pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cognition.memory import MemorySystem
from .core.tracing import Tracer
from .crew import build_debate_panel, build_research_crew
from .llm.base import LLMClient
from .llm.mock import MockLLM
from .orchestration.blackboard_pattern import BlackboardOrchestrator
from .orchestration.debate import DebateOrchestrator
from .orchestration.hierarchical import HierarchicalOrchestrator
from .orchestration.result import RunResult
from .orchestration.sequential import SequentialOrchestrator

PATTERNS = ("hierarchical", "blackboard", "debate", "sequential")


async def run_pattern(
    pattern: str,
    goal: str,
    *,
    llm: LLMClient | None = None,
    memory: MemorySystem | None = None,
    tracer: Tracer | None = None,
    corpus_dir: str | Path | None = None,
    **options: Any,
) -> RunResult:
    llm = llm or MockLLM()
    tracer = tracer or Tracer(max_llm_calls=options.pop("max_llm_calls", 80))
    if pattern == "debate":
        crew = build_debate_panel(llm, memory=memory, tracer=tracer, corpus_dir=corpus_dir)
        return await DebateOrchestrator(crew, **options).run(goal)
    crew = build_research_crew(llm, memory=memory, tracer=tracer, corpus_dir=corpus_dir)
    if pattern == "hierarchical":
        return await HierarchicalOrchestrator(crew, **options).run(goal)
    if pattern == "blackboard":
        return await BlackboardOrchestrator(crew, **options).run(goal)
    if pattern == "sequential":
        return await SequentialOrchestrator(crew, **options).run(goal)
    raise ValueError(f"Unknown pattern {pattern!r}; choose one of {PATTERNS}")
