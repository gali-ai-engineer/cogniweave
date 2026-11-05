"""Real LLM providers. SDKs are optional extras and imported lazily."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from typing import Any

from .base import LLMResponse, Message


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError("Install the extra: pip install 'cogniweave[anthropic]'") from exc
        self.model = model or os.getenv("COGNIWEAVE_MODEL", "claude-sonnet-5-5")
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = await self._client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in resp.content)
        return LLMResponse(text, resp.model, resp.usage.input_tokens, resp.usage.output_tokens)


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError("Install the extra: pip install 'cogniweave[openai]'") from exc
        self.model = model or os.getenv("COGNIWEAVE_MODEL", "gpt-4.1-mini")
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m.role, "content": m.content} for m in messages
        ]
        kwargs: dict[str, Any] = {"model": self.model, "messages": msgs, "max_completion_tokens": max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = await self._client.chat.completions.create(**kwargs)
        usage = resp.usage
        return LLMResponse(
            resp.choices[0].message.content or "",
            resp.model,
            getattr(usage, "prompt_tokens", 0),
            getattr(usage, "completion_tokens", 0),
        )


class OllamaLLM:
    """Local models through Ollama's HTTP API (no SDK needed)."""

    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or os.getenv("COGNIWEAVE_MODEL", "llama3.1")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m.role, "content": m.content} for m in messages
        ]
        options: dict[str, Any] = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        body = json.dumps({"model": self.model, "messages": msgs, "stream": False, "options": options})

        def _post() -> dict[str, Any]:
            req = urllib.request.Request(
                f"{self.host}/api/chat", data=body.encode(), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 - local, user-configured host
                return json.loads(r.read())

        data = await asyncio.to_thread(_post)
        return LLMResponse(
            data["message"]["content"], self.model, data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        )


def get_llm(provider: str = "mock", model: str | None = None):
    """Factory: ``mock`` | ``anthropic`` | ``openai`` | ``ollama``."""
    provider = provider.lower()
    if provider == "mock":
        from .mock import MockLLM

        return MockLLM()
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "openai":
        return OpenAILLM(model)
    if provider == "ollama":
        return OllamaLLM(model)
    raise ValueError(f"Unknown provider {provider!r}; choose mock, anthropic, openai or ollama")
