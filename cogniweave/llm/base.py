"""Provider-agnostic LLM interface.

Every agent talks to models through :class:`LLMClient`, so the orchestration
and cognition layers never depend on a vendor SDK. ``metadata`` is ignored by
real providers; the offline :class:`~cogniweave.llm.mock.MockLLM` uses it to
pick a deterministic, role-specific behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMError(RuntimeError):
    """Raised when a provider call fails after retries."""


@runtime_checkable
class LLMClient(Protocol):
    name: str

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used for budgets and the mock."""
    return max(1, len(text) // 4)
