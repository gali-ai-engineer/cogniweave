"""The cognitive agent: perceive → recall → deliberate/act → learn.

::

    ┌──────────┐   ┌──────────┐   ┌──────────────────────┐   ┌──────────┐
    │ perceive │ → │  recall  │ → │ deliberate ⇄ act     │ → │  learn   │
    │ task,ctx │   │ episodic │   │ ReAct loop over tools │   │ episode, │
    │          │   │ lessons, │   │ inside a token-bound  │   │ strategy │
    │          │   │ strategy │   │ working memory        │   │ reward   │
    └──────────┘   └──────────┘   └──────────────────────┘   └──────────┘
                                                    ▲              │
                                    reflect(feedback)  ◀───────────┘  (Reflexion)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..cognition.memory import Episode, MemorySystem, WorkingMemory
from ..llm.base import LLMClient, LLMError, Message
from .tools import ToolRegistry
from .tracing import BudgetExceeded, Tracer

_ACTION = re.compile(r"^ACTION:\s*([\w\-]+)\s*(\{.*\})?\s*$", re.M | re.S)
_CONFIDENCE = re.compile(r"^CONFIDENCE:\s*([0-9]*\.?[0-9]+)\s*$", re.M)
_THOUGHT = re.compile(r"^THOUGHT:.*$", re.M)


async def _backoff(delay: float) -> None:
    """Retry backoff; a module-level hook so tests can make it instant."""
    await asyncio.sleep(delay)


def parse_action(text: str) -> tuple[str, dict[str, Any]] | None:
    m = _ACTION.search(text)
    if not m:
        return None
    try:
        args = json.loads(m.group(2)) if m.group(2) else {}
    except json.JSONDecodeError:
        args = {}
    return m.group(1), args if isinstance(args, dict) else {}


def split_confidence(text: str, default: float = 0.7) -> tuple[str, float]:
    m = _CONFIDENCE.search(text)
    conf = max(0.0, min(1.0, float(m.group(1)))) if m else default
    body = _THOUGHT.sub("", _CONFIDENCE.sub("", text)).strip()
    return body, conf


@dataclass
class AgentOutput:
    agent: str
    role: str
    task: str
    content: str
    confidence: float
    strategy: str = ""
    steps: int = 1
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def merge(self, other: AgentOutput) -> AgentOutput:
        return AgentOutput(
            self.agent,
            self.role,
            self.task,
            f"{self.content}\n{other.content}",
            max(self.confidence, other.confidence),
            self.strategy,
            self.steps + other.steps,
            self.tool_calls + other.tool_calls,
        )


TOOL_PROTOCOL = """You can use tools. To call one, reply with exactly one line:
ACTION: <tool_name> <JSON arguments>
You will then receive a message starting with OBSERVATION:. When you have enough
information, reply with your final answer and no ACTION line.
Available tools:
{tools}"""


class CognitiveAgent:
    def __init__(
        self,
        name: str,
        role: str,
        llm: LLMClient,
        *,
        instructions: str = "",
        tools: ToolRegistry | None = None,
        memory: MemorySystem | None = None,
        tracer: Tracer | None = None,
        strategies: list[str] | None = None,
        persona: str | None = None,
        max_steps: int = 4,
        context_budget_tokens: int = 3000,
        max_tokens: int = 1500,
        retries: int = 2,
    ) -> None:
        self.name, self.role, self.llm = name, role, llm
        self.instructions = instructions
        self.tools = tools or ToolRegistry()
        self.memory = memory or MemorySystem()
        self.tracer = tracer or Tracer()
        self.strategies = strategies or []
        self.persona = persona
        self.max_steps = max_steps
        self.context_budget_tokens = context_budget_tokens
        self.max_tokens = max_tokens
        self.retries = retries

    def __repr__(self) -> str:
        return f"CognitiveAgent({self.name!r}, role={self.role!r})"

    # ----------------------------------------------------------- LLM access
    @property
    def system_prompt(self) -> str:
        parts = [f"You are {self.name}, the {self.role} in a multi-agent team."]
        if self.persona:
            parts.append(f"Your persona: {self.persona}.")
        if self.instructions:
            parts.append(self.instructions)
        if self.tools:
            parts.append(TOOL_PROTOCOL.format(tools=self.tools.describe()))
        parts.append("End every final answer with a line `CONFIDENCE: <number between 0 and 1>`.")
        return "\n\n".join(parts)

    async def _complete(self, messages: list[Message], meta: dict[str, Any]) -> str:
        self.tracer.check_budget()
        delay = 0.5
        for attempt in range(self.retries + 1):
            try:
                async with self.tracer.span("llm.complete", agent=self.name, role=meta.get("role")):
                    resp = await self.llm.complete(
                        messages, system=self.system_prompt, max_tokens=self.max_tokens, metadata=meta
                    )
                self.tracer.record_llm(self.name, resp.input_tokens, resp.output_tokens)
                return resp.text
            except BudgetExceeded:
                raise
            except Exception as exc:
                if attempt == self.retries:
                    raise LLMError(f"{self.name}: LLM call failed after {attempt + 1} attempts: {exc}") from exc
                await _backoff(delay)
                delay *= 2
        raise AssertionError("unreachable")

    # ----------------------------------------------------------- the cycle
    def _build_prompt(self, task: str, goal: str, wm: WorkingMemory, feedback: str, strategy: str) -> str:
        sections = [f"<role>{self.role}</role>"]
        if goal:
            sections.append(f"<goal>{goal}</goal>")
        sections.append(f"<task>{task}</task>")
        if strategy:
            sections.append(f"<strategy>{strategy}</strategy>")
        if lessons := wm.render("lesson"):
            sections.append(f"<lessons>\n{lessons}\n</lessons>")
        if experience := wm.render("experience"):
            sections.append(f"<past_experience>\n{experience}\n</past_experience>")
        if context := wm.render("context"):
            sections.append(f"<context>\n{context}\n</context>")
        if feedback:
            sections.append(f"<feedback>\n{feedback}\n</feedback>")
        return "\n".join(sections)

    async def run(
        self,
        task: str,
        *,
        goal: str = "",
        context: str = "",
        feedback: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AgentOutput:
        meta: dict[str, Any] = {"role": self.role, "agent": self.name, **(metadata or {})}
        if self.persona:
            meta.setdefault("persona", self.persona)

        async with self.tracer.span("agent.run", agent=self.name, kind="agent_run", task=task[:160]) as span:
            # 1. perceive: load the task context into a token-bounded working memory
            wm = WorkingMemory(self.context_budget_tokens)
            wm.add(context, salience=1.0, kind="context")

            # 2. recall: lessons (reflections) and the most relevant past experience
            recalled = self.memory.episodic.recall(task, k=5, agent=self.name)
            for ep, _score in recalled:
                if ep.kind == "reflection":
                    wm.add(f"- {ep.content}", salience=0.9, kind="lesson")
            for ep, score in [r for r in recalled if r[0].kind == "action"][:1]:
                wm.add(f"- {ep.content[:240]}", salience=min(0.6, 0.2 * score), kind="experience")
            strategy = self.memory.procedural.select(self.role, self.strategies)

            # 3. deliberate ⇄ act (ReAct)
            messages = [Message("user", self._build_prompt(task, goal, wm, feedback, strategy))]
            tool_calls: list[dict[str, Any]] = []
            text, step = "", 0
            for step in range(1, self.max_steps + 1):
                text = await self._complete(messages, meta)
                action = parse_action(text) if self.tools else None
                if action is None:
                    break
                name, args = action
                async with self.tracer.span("tool.call", agent=self.name, kind="tool", tool=name):
                    observation = await self.tools.invoke(name, args)
                tool_calls.append({"tool": name, "args": args, "observation": observation[:500]})
                messages += [Message("assistant", text), Message("user", f"OBSERVATION:\n{observation}")]
                if step == self.max_steps - 1:
                    messages.append(Message("user", "Tool budget reached. Give your final answer now."))

            content, confidence = split_confidence(text)
            span.attrs.update(confidence=confidence, steps=step, strategy=strategy, evicted=len(wm.evicted))

            # 4. learn: store the episode (low-confidence outcomes are more memorable)
            self.memory.episodic.add(
                Episode(
                    self.name,
                    task,
                    content[:500],
                    "action",
                    importance=round(0.2 + 0.5 * (1 - confidence), 2),
                    success=confidence,
                )
            )
            return AgentOutput(self.name, self.role, task, content, confidence, strategy, step, tool_calls)

    async def reflect(self, task: str, feedback: str) -> str:
        """Reflexion: turn external feedback into a reusable lesson in episodic memory."""
        prompt = (
            f"<role>reflector</role>\n<task>{task}</task>\n<feedback>\n{feedback}\n</feedback>\n"
            "Write one concise, reusable lesson that would have prevented this feedback. "
            "Start it with 'LESSON:'."
        )
        text = await self._complete([Message("user", prompt)], {"role": "reflector", "agent": self.name})
        lesson = text.split("LESSON:", 1)[-1].strip().splitlines()[0] if text.strip() else ""
        self.memory.episodic.add(Episode(self.name, task, lesson, "reflection", importance=0.9))
        self.tracer.event("reflection", agent=self.name, kind="reflection", lesson=lesson)
        return lesson

    def learn(self, strategy: str, reward: float) -> None:
        """Procedural learning: credit the strategy used with an outcome reward."""
        self.memory.procedural.update(self.role, strategy, reward)
