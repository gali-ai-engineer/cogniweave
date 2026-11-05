import asyncio

import pytest

pytest.importorskip("langgraph")

from cogniweave import MockLLM, build_research_crew  # noqa: E402
from cogniweave.adapters.langgraph_adapter import build_research_graph  # noqa: E402


def test_langgraph_runs_same_workflow():
    graph = build_research_graph(build_research_crew(MockLLM()))
    state = asyncio.run(graph.ainvoke({"goal": "What are the trade-offs of multi-agent systems?"}))
    assert state["review"]["score"] >= 0.8
    assert state["revisions"] == 1
    assert "## Sources" in state["draft"]
