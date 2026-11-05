"""The same hierarchical workflow, with LangGraph owning the control flow.

pip install "cogniweave[langgraph]"
python examples/04_langgraph.py
"""

import asyncio

from cogniweave import MockLLM, build_research_crew
from cogniweave.adapters.langgraph_adapter import build_research_graph


async def main() -> None:
    graph = build_research_graph(build_research_crew(MockLLM()))
    state = await graph.ainvoke({"goal": "What are the trade-offs between multi-agent and single-agent LLM systems?"})
    print(state["draft"])
    print(f"\nreview score {state['review']['score']} after {state['revisions']} revision(s)")


if __name__ == "__main__":
    asyncio.run(main())
