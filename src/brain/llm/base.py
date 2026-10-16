"""LLM client contract. Everything that calls a model depends on this, never on a vendor SDK."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Base class for model-call failures."""


class LLMUnavailable(LLMError):
    """The backend can't be reached, or the requested model isn't installed."""


class LLMOutputError(LLMError):
    """The model answered, but not with output matching the requested schema."""


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def chat(
        self, system: str, user: str, *, model: str | None = None, max_tokens: int | None = None
    ) -> str:
        """Plain-text completion."""

    def structured(self, system: str, user: str, schema: type[T], *, model: str | None = None) -> T:
        """Completion constrained to (and validated against) a pydantic schema."""
