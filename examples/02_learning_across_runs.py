"""Reflexion + persistent episodic memory: the team stops repeating a mistake.

Run 1: the critic rejects the first draft (no Limitations section), the writer
reflects, stores a lesson and revises. Run 2 loads that memory, recalls the
lesson before drafting, and passes review first time with fewer LLM calls.
"""

import asyncio
import tempfile
from pathlib import Path

from cogniweave import MemorySystem, run_pattern

QUESTION = "What are the trade-offs between multi-agent and single-agent LLM systems?"


async def main() -> None:
    path = Path(tempfile.mkdtemp()) / "memory.json"
    for run in (1, 2):
        memory = MemorySystem()
        memory.load(path)
        result = await run_pattern("hierarchical", QUESTION, memory=memory)
        memory.save(path)
        lessons = [e.content for e in memory.episodic.episodes if e.kind == "reflection"]
        print(f"run {run}: {result.summary()}")
        print(f"        lessons in memory: {lessons}")


if __name__ == "__main__":
    asyncio.run(main())
