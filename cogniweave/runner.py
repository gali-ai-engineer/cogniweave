"""One entry point for every orchestration pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cognition.memory import MemorySystem
from .core.tracing import Tracer
from .crew import Crew, build_debate_panel, build_research_crew
from .llm.base import LLMClient
from .llm.mock import MockLLM
from .orchestration.blackboard_pattern import BlackboardOrchestrator
from .orchestration.debate import DebateOrchestrator
from .orchestration.hierarchical import HierarchicalOrchestrator
from .orchestration.result import RunResult
from .orchestration.sequential import SequentialOrchestrator

PATTERNS = ("hierarchical", "blackboard", "debate", "sequential")


_ORCHESTRATORS = {
    "hierarchical": HierarchicalOrchestrator,
    "blackboard": BlackboardOrchestrator,
    "debate": DebateOrchestrator,
    "sequential": SequentialOrchestrator,
}


def build_orchestrator(
    pattern: str,
    *,
    llm: LLMClient | None = None,
    memory: MemorySystem | None = None,
    tracer: Tracer | None = None,
    corpus_dir: str | Path | None = None,
    **options: Any,
) -> tuple[Crew, Any]:
    """Assemble the crew and orchestrator without running them (lets callers attach observers)."""
    if pattern not in _ORCHESTRATORS:
        raise ValueError(f"Unknown pattern {pattern!r}; choose one of {PATTERNS}")
    llm = llm or MockLLM()
    tracer = tracer or Tracer(max_llm_calls=options.pop("max_llm_calls", 80))
    options.pop("max_llm_calls", None)
    builder = build_debate_panel if pattern == "debate" else build_research_crew
    crew = builder(llm, memory=memory, tracer=tracer, corpus_dir=corpus_dir)
    return crew, _ORCHESTRATORS[pattern](crew, **options)


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
    _crew, orchestrator = build_orchestrator(
        pattern, llm=llm, memory=memory, tracer=tracer, corpus_dir=corpus_dir, **options
    )
    return await orchestrator.run(goal)
