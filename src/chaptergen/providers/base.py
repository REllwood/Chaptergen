"""Abstract base for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

# One chat turn: {"role": "user" | "assistant", "content": "..."}
Message = dict[str, str]


class LLMProvider(ABC):
    """Common interface all providers must implement."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        history: Sequence[Message] = (),
    ) -> str:
        """Send a prompt and return the raw text response.

        ``history`` holds earlier user/assistant turns, sent between the system
        prompt and ``user_prompt``. ``temperature=None`` means the provider's default.
        """

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Return (ok, human-readable message) about provider/model readiness."""
