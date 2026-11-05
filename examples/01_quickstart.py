"""Run every orchestration pattern on the same question and compare cost.

python examples/01_quickstart.py                 # offline, deterministic
COGNIWEAVE_PROVIDER=anthropic python examples/01_quickstart.py
"""

import asyncio
import os

from cogniweave import PATTERNS, get_llm, run_pattern

QUESTION = "What are the trade-offs between multi-agent and single-agent LLM systems?"


async def main() -> None:
    llm = get_llm(os.getenv("COGNIWEAVE_PROVIDER", "mock"))
    print(f"{'pattern':<14}{'llm calls':>10}{'tokens':>9}{'review':>8}{'revisions':>11}")
    for pattern in PATTERNS:
        result = await run_pattern(pattern, QUESTION, llm=llm)
        score = f"{result.review.score:.2f}" if result.review else "-"
        print(
            f"{pattern:<14}{result.tracer.total_calls:>10}{result.tracer.total_tokens:>9}{score:>8}{result.revisions:>11}"
        )

    result = await run_pattern("hierarchical", QUESTION, llm=llm)
    print("\n" + result.answer)


if __name__ == "__main__":
    asyncio.run(main())
