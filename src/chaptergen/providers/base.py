"""Abstract base for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface all providers must implement."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        """Send a prompt and return the raw text response."""

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Return (ok, human-readable message) about provider/model readiness."""
