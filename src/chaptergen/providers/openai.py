"""OpenAI-compatible cloud LLM provider."""

from __future__ import annotations

from chaptergen.config import ProviderConfig
from chaptergen.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):

    def __init__(self, config: ProviderConfig) -> None:
        try:
            import openai  # noqa: F811
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install chaptergen[openai]"
            )

        if not config.api_key:
            raise ValueError(
                "An API key is required for the OpenAI provider. "
                "Set OPENAI_API_KEY or pass --api-key / --api-key-env."
            )

        kwargs: dict = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self._client = openai.OpenAI(**kwargs)
        self._model = config.model

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = resp.choices[0]
        return choice.message.content or ""

    def health_check(self) -> tuple[bool, str]:
        try:
            self._client.models.list()
            return True, f"OpenAI API reachable. Model: {self._model}"
        except Exception as exc:
            return False, f"OpenAI API error: {exc}"
