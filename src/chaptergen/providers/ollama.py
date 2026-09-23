"""Ollama local LLM provider."""

from __future__ import annotations

import httpx

from chaptergen.config import ProviderConfig
from chaptergen.providers.base import LLMProvider

_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)


def _with_tag(name: str) -> str:
    """Ollama treats an untagged model name as ``<name>:latest``."""
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


def _error_detail(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("error") or resp.text)
    except ValueError:
        return resp.text


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
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise RuntimeError(
                    f"Model '{self._model}' not found in Ollama. Pull it with: ollama pull {self._model}"
                ) from exc
            raise RuntimeError(f"Ollama returned {exc.response.status_code}: {_error_detail(exc.response)}") from exc

        data = resp.json()
        return data.get("message", {}).get("content", "")

    def health_check(self) -> tuple[bool, str]:
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=httpx.Timeout(5.0))
            resp.raise_for_status()
        except Exception:
            return False, f"Cannot reach Ollama at {self._base_url}. Is it running?"

        try:
            models = [m["name"] for m in resp.json().get("models", [])]
        except (ValueError, KeyError, TypeError, AttributeError):
            return False, f"{self._base_url} responded, but not like an Ollama server."

        # Match on the full name:tag. "llama3.1" only means "llama3.1:latest", so an
        # installed "llama3.1:70b" doesn't make it available.
        if _with_tag(self._model) in {_with_tag(m) for m in models}:
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
