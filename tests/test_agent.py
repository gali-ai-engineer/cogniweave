import asyncio

import pytest

from cogniweave.cognition.memory import MemorySystem
from cogniweave.core.agent import CognitiveAgent, parse_action, split_confidence
from cogniweave.core.tools import ToolRegistry, make_corpus_search
from cogniweave.core.tracing import Tracer
from cogniweave.llm.base import LLMError
from cogniweave.llm.mock import MockLLM


def test_parse_action_and_confidence():
    assert parse_action('THOUGHT: x\nACTION: corpus_search {"query": "q", "k": 2}') == (
        "corpus_search",
        {"query": "q", "k": 2},
    )
    assert parse_action("just an answer") is None
    body, conf = split_confidence("THOUGHT: hmm\nanswer\nCONFIDENCE: 0.42")
    assert body == "answer" and conf == 0.42
    assert split_confidence("no marker")[1] == 0.7


def _researcher(memory=None):
    memory = memory or MemorySystem()
    memory.semantic.add_document(
        "Parallelism lets independent research tasks run concurrently and reduces latency.\n\n"
        "Coordination between agents adds token cost and latency for every hand-off.",
        "notes.md",
    )
    return CognitiveAgent(
        "researcher",
        "researcher",
        MockLLM(),
        tools=ToolRegistry([make_corpus_search(memory.semantic)]),
        memory=memory,
        tracer=Tracer(),
    )


def test_react_loop_calls_tool_then_answers_with_citations():
    agent = _researcher()
    out = asyncio.run(agent.run("parallelism and coordination cost"))
    assert out.steps == 2 and out.tool_calls[0]["tool"] == "corpus_search"
    assert "[notes.md#" in out.content and out.confidence == 0.85
    assert agent.memory.episodic.episodes[-1].agent == "researcher"
    assert agent.tracer.usage["researcher"]["calls"] == 2


def test_reflection_is_stored_and_recalled_into_next_prompt():
    seen_prompts = []

    def writer(messages, meta):
        seen_prompts.append(messages[0].content)
        return "draft\nCONFIDENCE: 0.8"

    agent = CognitiveAgent("writer", "writer", MockLLM({"writer": writer}), tracer=Tracer())
    asyncio.run(agent.run("write the report"))
    lesson = asyncio.run(agent.reflect("write the report", '{"issues": ["Add a Limitations section."]}'))
    assert "Limitations" in lesson
    asyncio.run(agent.run("write the report"))
    assert "<lessons>" not in seen_prompts[0] and "Limitations" in seen_prompts[1]


def test_llm_failures_are_retried_then_raised(monkeypatch):
    async def instant(_delay):
        return None

    monkeypatch.setattr("cogniweave.core.agent._backoff", instant)
    calls = {"n": 0}

    class Flaky(MockLLM):
        async def complete(self, messages, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return await super().complete(messages, **kw)

    agent = CognitiveAgent("a", "analyst", Flaky(), tracer=Tracer(), retries=2)
    assert asyncio.run(agent.run("x")).content
    assert calls["n"] == 3

    class Dead(MockLLM):
        async def complete(self, messages, **kw):
            raise ConnectionError("down")

    dead = CognitiveAgent("d", "analyst", Dead(), tracer=Tracer(), retries=1)
    with pytest.raises(LLMError):
        asyncio.run(dead.run("x"))
