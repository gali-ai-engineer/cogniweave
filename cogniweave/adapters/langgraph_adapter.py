"""Run the hierarchical research workflow as a LangGraph ``StateGraph``.

The cognitive agents, memory and tracing are unchanged; LangGraph only owns the
control flow. This shows the core is framework-agnostic: the same phases
(plan → execute → review ⇄ revise) map one-to-one onto graph nodes, and you
gain LangGraph features such as checkpointing and streaming.

    pip install "cogniweave[langgraph]"
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..crew import Crew
from ..orchestration.hierarchical import HierarchicalOrchestrator


class ResearchState(TypedDict, total=False):
    goal: str
    plan: Any
    results: dict[str, Any]
    final_id: str
    draft: str
    review: dict[str, Any]
    revisions: int


def build_research_graph(crew: Crew, *, max_revisions: int = 2, accept_threshold: float = 0.8):
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError("LangGraph is optional: pip install 'cogniweave[langgraph]'") from exc

    orch = HierarchicalOrchestrator(crew, max_revisions=max_revisions, accept_threshold=accept_threshold)

    async def plan_node(state: ResearchState) -> ResearchState:
        return {"plan": await orch.make_plan(state["goal"]), "revisions": 0}

    async def execute_node(state: ResearchState) -> ResearchState:
        plan = state["plan"]
        results = await orch.execute(plan)
        final_id = plan.sinks()[-1].id
        return {"results": results, "final_id": final_id, "draft": results[final_id].content}

    async def review_node(state: ResearchState) -> ResearchState:
        review = await orch.review(state["goal"], state["draft"])
        return {"review": review.to_dict()}

    async def revise_node(state: ResearchState) -> ResearchState:
        from ..orchestration.plan import Review

        final = state["results"][state["final_id"]]
        draft = await orch.revise(final, state["goal"], Review.from_dict(state["review"]))
        return {"draft": draft, "revisions": state.get("revisions", 0) + 1}

    def route(state: ResearchState) -> str:
        if state["review"]["score"] >= accept_threshold or state.get("revisions", 0) >= max_revisions:
            return "done"
        return "revise"

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("review", review_node)
    graph.add_node("revise", revise_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "review")
    graph.add_conditional_edges("review", route, {"revise": "revise", "done": END})
    graph.add_edge("revise", "review")
    return graph.compile()
