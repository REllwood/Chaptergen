"""Provider and model config resolution (flags > env > defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved configuration for a single generation run."""

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None  # None: provider default


def resolve_config(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
) -> ProviderConfig:
    """Build a :class:`ProviderConfig` from CLI flags and env vars.

    Priority: explicit flag > env var > built-in default.
    """
    resolved_provider = (provider or os.environ.get("CHAPTERGEN_PROVIDER", DEFAULT_PROVIDER)).lower()

    if resolved_provider == "ollama":
        resolved_model = model or os.environ.get("CHAPTERGEN_MODEL", DEFAULT_OLLAMA_MODEL)
        resolved_url = normalise_ollama_url(base_url or os.environ.get("OLLAMA_HOST", ""))
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


def normalise_ollama_url(value: str) -> str:
    """Turn an ``OLLAMA_HOST``-style value into a full URL, as the Ollama CLI does.

    Accepts forms like ``0.0.0.0``, ``myhost:11434``, ``http://myhost`` and
    ``https://myhost/prefix``. Without a scheme the port defaults to 11434;
    with ``http://``/``https://`` it defaults to 80/443. Wildcard bind
    addresses (``0.0.0.0``, ``::``) are mapped to localhost because they
    can't be connected to on every OS.
    """
    value = value.strip()
    if not value:
        return DEFAULT_OLLAMA_URL

    scheme, sep, rest = value.partition("://")
    if sep:
        scheme = scheme.lower()
        default_port = {"http": "80", "https": "443"}.get(scheme)
        if default_port is None:
            raise ValueError(f"Invalid Ollama URL '{value}': scheme must be http or https")
    else:
        scheme, rest, default_port = "http", value, "11434"

    hostport, _, path = rest.partition("/")
    if hostport.startswith("["):  # [IPv6]:port
        host, _, tail = hostport[1:].partition("]")
        port = tail[1:] if tail.startswith(":") else ""
    elif hostport.count(":") == 1:
        host, port = hostport.split(":")
    else:  # hostname, IPv4 or bare IPv6
        host, port = hostport, ""

    if port and not port.isdigit():
        raise ValueError(f"Invalid Ollama URL '{value}': port must be a number")

    host = {"": "127.0.0.1", "0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    if ":" in host:
        host = f"[{host}]"

    url = f"{scheme}://{host}:{port or default_port}"
    path = path.strip("/")
    return f"{url}/{path}" if path else url


def _resolve_api_key(
    explicit: str | None,
    env_name: str | None,
    default_env: str,
) -> str | None:
    if explicit:
        return explicit
    env_var = env_name or default_env
    return os.environ.get(env_var)
