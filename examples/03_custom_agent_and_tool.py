"""Build your own agent with a custom tool, without any orchestrator.

The mock handler stands in for a real model so this runs offline; swap
``MockLLM(...)`` for ``get_llm("anthropic")`` to use a hosted model.
"""

import asyncio
import json

from cogniweave import CognitiveAgent, MockLLM, ToolRegistry, Tracer, tool
from cogniweave.core.tools import calculator

FX = {"USD": 1.0, "INR": 88.6, "EUR": 0.86}  # illustrative static rates


@tool(description="Convert an amount between currencies using a static table.", amount="float", src="str", dst="str")
def convert(amount: float, src: str, dst: str) -> str:
    return f"{amount * FX[dst] / FX[src]:.2f} {dst}"


def finance_bot(messages, meta):
    """Stand-in for a model following the ReAct protocol."""
    observations = [m.content for m in messages if m.content.startswith("OBSERVATION:")]
    if not observations:
        return "ACTION: convert " + json.dumps({"amount": 1200, "src": "USD", "dst": "INR"})
    if len(observations) == 1:
        return 'ACTION: calculator {"expression": "1200 * 12"}'
    inr, yearly = observations[0].split()[-2], observations[1].split()[-1]
    return f"1,200 USD/month is {inr} INR; yearly USD total is {yearly}.\nCONFIDENCE: 0.9"


async def main() -> None:
    tracer = Tracer()
    agent = CognitiveAgent(
        "fin",
        "finance-analyst",
        MockLLM({"finance-analyst": finance_bot}),
        tools=ToolRegistry([convert, calculator]),
        tracer=tracer,
    )
    out = await agent.run("How much is a 1,200 USD monthly subscription in INR, and per year in USD?")
    print(out.content)
    print("tool calls:", [c["tool"] for c in out.tool_calls], "| confidence:", out.confidence)
    print(tracer.to_mermaid())


if __name__ == "__main__":
    asyncio.run(main())
