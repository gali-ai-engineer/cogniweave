import asyncio

import pytest
from pydantic import ValidationError

from cogniweave.core.blackboard import Blackboard
from cogniweave.core.tools import ToolRegistry, calculator, tool
from cogniweave.core.tracing import BudgetExceeded, Tracer
from cogniweave.orchestration.plan import Plan, Review, extract_json


def _plan(tasks):
    return Plan.model_validate({"goal": "g", "tasks": tasks})


def test_plan_topological_order_and_levels():
    plan = _plan(
        [
            {"id": "w", "role": "writer", "description": "write", "depends_on": ["a"]},
            {"id": "r1", "role": "researcher", "description": "r1"},
            {"id": "r2", "role": "researcher", "description": "r2"},
            {"id": "a", "role": "analyst", "description": "a", "depends_on": ["r1", "r2"]},
        ]
    )
    order = [t.id for t in plan.topo_order()]
    assert order.index("a") > order.index("r1") and order.index("w") > order.index("a")
    assert [[t.id for t in wave] for wave in plan.levels()] == [["r1", "r2"], ["a"], ["w"]]
    assert [t.id for t in plan.sinks()] == ["w"]
    assert "r1 --> a" in plan.to_mermaid()


@pytest.mark.parametrize(
    "tasks, message",
    [
        (
            [
                {"id": "a", "role": "x", "description": "d", "depends_on": ["b"]},
                {"id": "b", "role": "x", "description": "d", "depends_on": ["a"]},
            ],
            "cycle",
        ),
        ([{"id": "a", "role": "x", "description": "d", "depends_on": ["zzz"]}], "unknown"),
        ([{"id": "a", "role": "x", "description": "d"}, {"id": "a", "role": "x", "description": "d"}], "duplicate"),
    ],
)
def test_plan_rejects_invalid_dags(tasks, message):
    with pytest.raises(ValidationError, match=message):
        _plan(tasks)


def test_extract_json_handles_fenced_and_bare():
    assert extract_json('noise ```json\n{"a": 1}\n``` tail') == {"a": 1}
    assert extract_json('prefix {"b": [1, 2]} suffix') == {"b": [1, 2]}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_review_parse_falls_back_on_garbage():
    assert Review.parse('{"score": 0.9, "verdict": "accept", "issues": []}').score == 0.9
    assert Review.parse("not json").score == 0.5


def test_calculator_is_safe():
    assert calculator.fn("2 * (3 + 4) ** 2") == "98"
    for bad in ("__import__('os')", "2 ** 1000", "a + 1"):
        with pytest.raises(ValueError):
            calculator.fn(bad)


def test_registry_turns_errors_into_observations():
    @tool(description="divide", a="float", b="float")
    def divide(a, b):
        return a / b

    async def go():
        reg = ToolRegistry([divide])
        assert await reg.invoke("divide", {"a": 6, "b": 3}) == "2.0"
        assert (await reg.invoke("divide", {"a": 1, "b": 0})).startswith("ERROR: ZeroDivisionError")
        assert "unknown tool" in await reg.invoke("nope", {})

    asyncio.run(go())


def test_blackboard_post_query_subscribe():
    async def go():
        board, seen = Blackboard(), []
        board.subscribe("finding", lambda e: seen.append(e.id))
        q = await board.post("question", "why?", "planner")
        f = await board.post("finding", "because", "researcher", refs=(q.id,), tags=("t1",))
        assert board.query("finding", ref=q.id) == [f]
        assert board.latest("question") == q and seen == [f.id]
        assert board.snapshot()[1]["tags"] == ["t1"]

    asyncio.run(go())


def test_tracer_nests_spans_and_enforces_budget():
    async def go():
        tr = Tracer(max_llm_calls=1)
        async with tr.span("outer") as outer, tr.span("inner") as inner:
            pass
        assert inner.parent_id == outer.id
        tr.record_llm("a", 10, 5)
        assert tr.total_tokens == 15
        with pytest.raises(BudgetExceeded):
            tr.check_budget()

    asyncio.run(go())
