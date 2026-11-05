import asyncio
import json
import re
import time

import pytest

from cogniweave import (
    BlackboardOrchestrator,
    HierarchicalOrchestrator,
    MemorySystem,
    MockLLM,
    Tracer,
    build_research_crew,
    run_pattern,
)
from cogniweave.core.tracing import BudgetExceeded
from cogniweave.llm.mock import _planner

QUESTION = "What are the trade-offs between multi-agent and single-agent LLM systems?"


def run(coro):
    return asyncio.run(coro)


def _citations(text):
    return re.findall(r"\[[^\]]+\.md#\d+\]", text)


def test_hierarchical_plans_executes_reviews_and_revises():
    result = run(run_pattern("hierarchical", QUESTION))
    assert [t.role for t in result.plan.tasks] == ["researcher", "researcher", "analyst", "writer"]
    assert result.review.score >= 0.8 and result.revisions == 1
    assert "## Limitations" in result.answer and "## Sources" in result.answer
    assert len(_citations(result.answer)) >= 3
    kinds = [e["kind"] for e in result.blackboard]
    assert kinds[:2] == ["goal", "plan"] and kinds.count("critique") == 2


def test_lessons_persist_across_runs(tmp_path):
    path = tmp_path / "memory.json"
    first_mem = MemorySystem()
    first = run(run_pattern("hierarchical", QUESTION, memory=first_mem))
    first_mem.save(path)

    second_mem = MemorySystem()
    second_mem.load(path)
    second = run(run_pattern("hierarchical", QUESTION, memory=second_mem))
    assert first.revisions == 1 and second.revisions == 0
    assert second.tracer.total_calls < first.tracer.total_calls


def test_independent_tasks_run_in_parallel():
    llm = MockLLM(latency_s=0.05)
    crew = build_research_crew(llm)
    orch = HierarchicalOrchestrator(crew)
    plan = run(orch.make_plan(QUESTION))
    start = time.perf_counter()
    run(orch.execute(plan))
    elapsed = time.perf_counter() - start
    # 2 parallel researchers (2 calls each) + analyst + writer = 4 sequential call slots, not 6
    assert elapsed < 0.05 * 5.5


def test_invalid_plan_is_repaired_from_feedback():
    attempts = []

    def flaky_planner(messages, meta):
        attempts.append(messages[0].content)
        if len(attempts) == 1:
            return '{"tasks": [{"id": "x", "role": "astronaut", "description": "fly"}]}'
        return _planner(messages, meta)

    crew = build_research_crew(MockLLM({"planner": flaky_planner}))
    plan = run(HierarchicalOrchestrator(crew).make_plan(QUESTION))
    assert len(attempts) == 2 and "unknown roles" in attempts[1]
    assert len(plan.tasks) == 4


def test_planner_that_never_succeeds_uses_fallback():
    crew = build_research_crew(MockLLM({"planner": lambda m, meta: "I refuse to plan."}))
    result = run(HierarchicalOrchestrator(crew).run(QUESTION))
    assert [t.id for t in result.plan.tasks] == ["research", "write"]
    assert result.answer


def test_low_confidence_triggers_metacognitive_follow_up():
    def unsure(messages, meta):
        if not any(m.content.startswith("OBSERVATION:") for m in messages):
            return 'ACTION: corpus_search {"query": "agents", "k": 1}'
        return "FINDINGS:\n- thin evidence [multi_agent_systems.md#0]\nCONFIDENCE: 0.2"

    result = run(run_pattern("hierarchical", QUESTION, llm=MockLLM({"researcher": unsure})))
    assert set(result.extra["follow_ups"]) == {"t1", "t2"}


def test_blackboard_control_emerges_from_board_state():
    crew = build_research_crew(MockLLM())
    result = run(BlackboardOrchestrator(crew).run(QUESTION))
    assert result.extra["schedule"] == ["decomposer", "researcher", "analyst", "writer", "critic", "writer", "critic"]
    assert result.review.score >= 0.8
    findings = [e for e in result.blackboard if e["kind"] == "finding"]
    assert len(findings) == 2 and all(e["refs"] for e in findings)


def test_debate_produces_verdict_with_rebuttals():
    result = run(run_pattern("debate", QUESTION))
    assert result.answer.startswith("## Verdict")
    args = result.extra["final_arguments"]
    assert set(args) == {"advocate", "skeptic", "pragmatist"}
    assert all("REBUTTAL:" in text for text in args.values())


def test_sequential_baseline():
    result = run(run_pattern("sequential", QUESTION))
    assert result.tracer.total_calls == 4 and _citations(result.answer)


def test_budget_guardrail_stops_runaway_runs():
    with pytest.raises(BudgetExceeded):
        run(run_pattern("hierarchical", QUESTION, tracer=Tracer(max_llm_calls=3)))


def test_trace_exports(tmp_path):
    result = run(run_pattern("hierarchical", QUESTION))
    mermaid = result.tracer.to_mermaid()
    assert mermaid.startswith("sequenceDiagram") and "Note over writer: reflect" in mermaid
    path = tmp_path / "trace.json"
    result.tracer.save_json(path)
    data = json.loads(path.read_text())
    assert data["totals"]["llm_calls"] == result.tracer.total_calls


def test_unknown_pattern():
    with pytest.raises(ValueError):
        run(run_pattern("telepathy", QUESTION))
