"""Provider and model config resolution (flags > env > defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved configuration for a single generation run."""

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0


def resolve_config(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> ProviderConfig:
    """Build a :class:`ProviderConfig` from CLI flags and env vars.

    Priority: explicit flag > env var > built-in default.
    """
    resolved_provider = (provider or os.environ.get("CHAPTERGEN_PROVIDER", DEFAULT_PROVIDER)).lower()

    if resolved_provider == "ollama":
        resolved_model = model or os.environ.get("CHAPTERGEN_MODEL", DEFAULT_OLLAMA_MODEL)
        resolved_url = base_url or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL)
        return ProviderConfig(
            provider=resolved_provider,
            model=resolved_model,
            base_url=resolved_url,
            temperature=temperature,
        )

    if resolved_provider == "openai":
        resolved_model = model or os.environ.get("CHAPTERGEN_MODEL", DEFAULT_OPENAI_MODEL)
        resolved_key = _resolve_api_key(api_key, api_key_env, "OPENAI_API_KEY")
        resolved_url = base_url or os.environ.get("OPENAI_BASE_URL")
        return ProviderConfig(
            provider=resolved_provider,
            model=resolved_model,
            api_key=resolved_key,
            base_url=resolved_url,
            temperature=temperature,
        )

    raise ValueError(f"Unknown provider '{resolved_provider}'. Supported: ollama, openai")


def _resolve_api_key(
    explicit: str | None,
    env_name: str | None,
    default_env: str,
) -> str | None:
    if explicit:
        return explicit
    env_var = env_name or default_env
    return os.environ.get(env_var)
