"""Ollama local LLM provider."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from chaptergen.config import DEFAULT_OLLAMA_URL, ProviderConfig
from chaptergen.providers.base import LLMProvider, Message

# Long transcripts on CPU-only machines can take several minutes to process
_READ_TIMEOUT_SECONDS = 600
_TIMEOUT = httpx.Timeout(connect=10.0, read=_READ_TIMEOUT_SECONDS, write=10.0, pool=10.0)

# Ollama's default context window is only a few thousand tokens and it silently
# drops the start of longer prompts, so size it to fit each request.
_MIN_CONTEXT = 4096
_MAX_CONTEXT = 131072
_REPLY_ALLOWANCE = 4096  # the JSON answer, plus thinking for reasoning models
_CHARS_PER_TOKEN = 3  # English averages ~4; err on the side of a bigger window


def _context_size(prompt_chars: int, model_limit: int | None) -> int:
    """Pick a num_ctx that fits the prompt and reply, rounded up to a multiple of 2048."""
    needed = prompt_chars // _CHARS_PER_TOKEN + _REPLY_ALLOWANCE
    size = max(_MIN_CONTEXT, -(-needed // 2048) * 2048)
    return min(size, model_limit or _MAX_CONTEXT)


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
        self._base_url = (config.base_url or DEFAULT_OLLAMA_URL).rstrip("/")
        self._model = config.model
        self._model_limit: int | None = None
        self._model_limit_checked = False

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        history: Sequence[Message] = (),
    ) -> str:
        url = f"{self._base_url}/api/chat"
        prompt_chars = len(system_prompt) + len(user_prompt) + sum(len(m["content"]) for m in history)
        num_ctx = _context_size(prompt_chars, self._context_limit())
        payload = {
            "model": self._model,
            "stream": False,
            # Default to 0 for the most repeatable chapters from local models
            "options": {"temperature": 0.0 if temperature is None else temperature, "num_ctx": num_ctx},
            "messages": [
                {"role": "system", "content": system_prompt},
                *history,
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
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama didn't respond within {_READ_TIMEOUT_SECONDS // 60} minutes. "
                "Try a smaller model or a shorter transcript."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise RuntimeError(
                    f"Model '{self._model}' not found in Ollama. Pull it with: ollama pull {self._model}"
                ) from exc
            raise RuntimeError(f"Ollama returned {exc.response.status_code}: {_error_detail(exc.response)}") from exc
        except httpx.HTTPError as exc:
            raise ConnectionError(f"Lost connection to Ollama at {self._base_url}: {exc}") from exc

        data = resp.json()
        return data.get("message", {}).get("content", "")

    def _context_limit(self) -> int | None:
        """The model's trained context length from /api/show, looked up once."""
        if not self._model_limit_checked:
            self._model_limit_checked = True
            try:
                resp = httpx.post(f"{self._base_url}/api/show", json={"model": self._model}, timeout=httpx.Timeout(10.0))
                resp.raise_for_status()
                info = resp.json().get("model_info") or {}
                self._model_limit = next(
                    (v for k, v in info.items() if k.endswith(".context_length") and isinstance(v, int)),
                    None,
                )
            except (httpx.HTTPError, ValueError, AttributeError):
                pass  # older Ollama or unknown model: fall back to _MAX_CONTEXT
        return self._model_limit

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
