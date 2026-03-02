"""Ollama local LLM provider."""

from __future__ import annotations

import httpx

from chaptergen.config import ProviderConfig
from chaptergen.providers.base import LLMProvider

_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)


class OllamaProvider(LLMProvider):

    def __init__(self, config: ProviderConfig) -> None:
        self._base_url = (config.base_url or "http://localhost:11434").rstrip("/")
        self._model = config.model

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            resp = httpx.post(url, json=payload, timeout=_TIMEOUT)
            resp.raise_for_status()
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Ollama returned {exc.response.status_code}: {exc.response.text}")

        data = resp.json()
        return data.get("message", {}).get("content", "")

    def health_check(self) -> tuple[bool, str]:
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=httpx.Timeout(5.0))
            resp.raise_for_status()
        except Exception:
            return False, f"Cannot reach Ollama at {self._base_url}. Is it running?"

        models = [m["name"] for m in resp.json().get("models", [])]
        # Ollama model names may include a tag suffix like ":latest"
        base_names = [m.split(":")[0] for m in models]

        if self._model in models or self._model in base_names:
            return True, f"Ollama is running. Model '{self._model}' is available."

        if models:
            available = ", ".join(models[:10])
            return False, (
                f"Ollama is running but model '{self._model}' not found. "
                f"Available: {available}. Pull it with: ollama pull {self._model}"
            )

        return False, (
            f"Ollama is running but no models are installed. "
            f"Pull one with: ollama pull {self._model}"
        )
