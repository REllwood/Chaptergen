"""Provider registry and factory."""

from __future__ import annotations

from chaptergen.config import ProviderConfig
from chaptergen.providers.base import LLMProvider
from chaptergen.providers.ollama import OllamaProvider
from chaptergen.providers.openai import OpenAIProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
}


def get_provider(config: ProviderConfig) -> LLMProvider:
    """Instantiate the correct provider from a resolved config."""
    cls = _PROVIDERS.get(config.provider)
    if cls is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown provider '{config.provider}'. Supported: {supported}")
    return cls(config)
