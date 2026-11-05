"""Tool registry and built-in tools.

Tools are plain (sync or async) Python callables with a JSON-ish parameter
description. Agents call them through a ReAct-style protocol::

    ACTION: corpus_search {"query": "...", "k": 4}

Errors are returned to the agent as observations instead of crashing the run,
so the model can recover.
"""

from __future__ import annotations

import ast
import inspect
import json
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..cognition.memory import SemanticMemory


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, str] = field(default_factory=dict)

    async def __call__(self, **kwargs: Any) -> str:
        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, str) else json.dumps(result, default=str)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __bool__(self) -> bool:
        return bool(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        return "\n".join(
            f"- {t.name}({', '.join(f'{k}: {v}' for k, v in t.parameters.items())}): {t.description}"
            for t in self._tools.values()
        )

    async def invoke(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}. Available: {', '.join(self._tools)}"
        try:
            return await tool(**args)
        except Exception as exc:  # surfaced to the model as an observation
            return f"ERROR: {type(exc).__name__}: {exc}"


def tool(name: str | None = None, description: str | None = None, **parameters: str):
    """Decorator turning a function into a :class:`Tool`."""

    def wrap(fn: Callable[..., Any]) -> Tool:
        return Tool(name or fn.__name__, description or (fn.__doc__ or "").strip(), fn, parameters)

    return wrap


# ------------------------------------------------------------------ built-ins
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@tool(description="Evaluate an arithmetic expression safely (no names, calls or attributes).", expression="str")
def calculator(expression: str) -> str:
    return str(_safe_eval(ast.parse(expression, mode="eval")))


def make_corpus_search(semantic: SemanticMemory) -> Tool:
    def corpus_search(query: str, k: int = 4) -> str:
        hits = semantic.search(query, k=int(k))
        if not hits:
            return "No results."
        return "\n".join(f"[{chunk.id}] {chunk.text}" for chunk, _ in hits)

    return Tool(
        "corpus_search",
        "Search the knowledge base. Returns lines of the form '[source#chunk] text'.",
        corpus_search,
        {"query": "str", "k": "int = 4"},
    )
