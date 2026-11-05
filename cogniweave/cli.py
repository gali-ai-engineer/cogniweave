"""Command-line interface.

cogniweave run "Should we use a multi-agent or single-agent design?" --pattern hierarchical
cogniweave run "..." --pattern debate --provider anthropic
cogniweave memory --memory .cogniweave/memory.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path

from .cognition.memory import MemorySystem
from .llm.providers import get_llm
from .runner import PATTERNS, run_pattern

DEFAULT_QUESTION = "What are the trade-offs between multi-agent and single-agent LLM systems?"
DEFAULT_MEMORY = Path(".cogniweave/memory.json")


def _cmd_run(args: argparse.Namespace) -> int:
    memory = MemorySystem()
    memory_path = None if args.no_memory else Path(args.memory)
    if memory_path and memory.load(memory_path):
        print(f"· loaded {len(memory.episodic.episodes)} episodes from {memory_path}", file=sys.stderr)

    result = asyncio.run(
        run_pattern(
            args.pattern,
            args.question,
            llm=get_llm(args.provider, args.model),
            memory=memory,
            max_llm_calls=args.max_llm_calls,
        )
    )
    if memory_path:
        memory.save(memory_path)

    print(result.answer)
    print("\n---")
    print(result.summary())
    if result.plan:
        print("plan: " + " | ".join(f"{t.id}:{t.role}<-{','.join(t.depends_on) or '∅'}" for t in result.plan.tasks))
    if "schedule" in result.extra:
        print("schedule: " + " → ".join(result.extra["schedule"]))
    if args.usage:
        print("\n" + result.tracer.usage_table())
    if args.trace_out:
        result.tracer.save_json(args.trace_out)
        print(f"trace written to {args.trace_out}", file=sys.stderr)
    if args.mermaid:
        print("\n```mermaid\n" + result.tracer.to_mermaid() + "\n```")
    return 0


def _cmd_memory(args: argparse.Namespace) -> int:
    memory = MemorySystem()
    if not memory.load(args.memory):
        print(f"No memory file at {args.memory}")
        return 1
    lessons = [e for e in memory.episodic.episodes if e.kind == "reflection"]
    print(f"{len(memory.episodic.episodes)} episodes, {len(lessons)} lessons")
    for e in lessons:
        print(f"- [{e.agent}] {e.content}")
    for role, table in memory.procedural.stats.items():
        for strategy, (n, total) in table.items():
            print(f"  {role:<10} {strategy:<45} n={int(n)} mean_reward={total / n:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cogniweave", description="Multi-agent orchestration & cognitive architecture"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an orchestration pattern on a question")
    run.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    run.add_argument("--pattern", choices=PATTERNS, default="hierarchical")
    run.add_argument("--provider", default="mock", help="mock | anthropic | openai | ollama")
    run.add_argument("--model", default=None)
    run.add_argument("--memory", default=str(DEFAULT_MEMORY), help="path to persistent memory JSON")
    run.add_argument("--no-memory", action="store_true", help="do not load or save long-term memory")
    run.add_argument("--max-llm-calls", type=int, default=80)
    run.add_argument("--usage", action="store_true", help="print a per-agent token usage table")
    run.add_argument("--trace-out", default=None, help="write the full trace as JSON")
    run.add_argument("--mermaid", action="store_true", help="print a Mermaid sequence diagram of the run")
    run.set_defaults(func=_cmd_run)

    mem = sub.add_parser("memory", help="inspect learned lessons and strategy statistics")
    mem.add_argument("--memory", default=str(DEFAULT_MEMORY))
    mem.set_defaults(func=_cmd_memory)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
