"""OpenAI-compatible cloud LLM provider."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.markup import escape

from chaptergen.config import ProviderConfig
from chaptergen.providers.base import LLMProvider, Message

_console = Console(stderr=True)


class OpenAIProvider(LLMProvider):

    def __init__(self, config: ProviderConfig) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install chaptergen[openai]"
            ) from exc

        if not config.api_key:
            raise ValueError(
                "An API key is required for the OpenAI provider. "
                "Set OPENAI_API_KEY or pass --api-key / --api-key-env."
            )

        kwargs: dict = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self._openai = openai
        self._client = openai.OpenAI(**kwargs)
        self._model = config.model
        self._temperature_supported = True

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        history: Sequence[Message] = (),
    ) -> str:
        request: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *history,
                {"role": "user", "content": user_prompt},
            ],
        }
        if temperature is not None and self._temperature_supported:
            request["temperature"] = temperature

        try:
            resp = self._client.chat.completions.create(**request)
        except self._openai.BadRequestError as exc:
            # Reasoning models (o-series, GPT-5 family) only accept their default temperature
            if "temperature" not in request or (exc.param != "temperature" and "temperature" not in str(exc.body)):
                raise
            self._temperature_supported = False
            _console.print(
                f"[yellow]{escape(self._model)} doesn't support --temperature; using its default.[/yellow]"
            )
            del request["temperature"]
            resp = self._client.chat.completions.create(**request)

        choice = resp.choices[0]
        return choice.message.content or ""

    def health_check(self) -> tuple[bool, str]:
        try:
            available = {m.id for m in self._client.models.list()}
        except Exception as exc:
            return False, f"OpenAI API error: {exc}"

        if not available:
            # Some OpenAI-compatible servers don't list models
            return True, f"API reachable, but it didn't list its models, so '{self._model}' couldn't be confirmed."
        if self._model not in available:
            return False, f"API reachable, but model '{self._model}' isn't available to this API key."
        return True, f"OpenAI API reachable. Model '{self._model}' is available."
