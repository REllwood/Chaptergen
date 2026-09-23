"""Tests for the LLM provider implementations (network calls are faked)."""

import httpx
import pytest

from chaptergen.config import ProviderConfig
from chaptergen.providers import ollama as ollama_module
from chaptergen.providers.ollama import OllamaProvider


def _response(status: int, **kwargs) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "http://ollama.test"), **kwargs)


def _ollama(model: str = "llama3.1") -> OllamaProvider:
    return OllamaProvider(ProviderConfig(provider="ollama", model=model, base_url="http://ollama.test"))


def _installed(monkeypatch, *names: str) -> None:
    payload = {"models": [{"name": n} for n in names]}
    monkeypatch.setattr(ollama_module.httpx, "get", lambda *a, **k: _response(200, json=payload))


class TestOllamaHealthCheck:

    @pytest.mark.parametrize(("requested", "installed"), [
        ("llama3.1", "llama3.1:latest"),
        ("llama3.1:latest", "llama3.1:latest"),
        ("llama3.1:70b", "llama3.1:70b"),
        ("hf.co/user/model", "hf.co/user/model:latest"),
    ])
    def test_model_available(self, monkeypatch, requested, installed):
        _installed(monkeypatch, "mistral:latest", installed)
        ok, msg = _ollama(requested).health_check()
        assert ok, msg

    def test_other_tag_does_not_count(self, monkeypatch):
        # Asking for "llama3.1" means llama3.1:latest; the 70b tag alone won't serve it
        _installed(monkeypatch, "llama3.1:70b")
        ok, msg = _ollama("llama3.1").health_check()
        assert not ok
        assert "ollama pull llama3.1" in msg

    def test_no_models_installed(self, monkeypatch):
        _installed(monkeypatch)
        ok, msg = _ollama().health_check()
        assert not ok
        assert "no models are installed" in msg

    def test_not_an_ollama_server(self, monkeypatch):
        monkeypatch.setattr(ollama_module.httpx, "get", lambda *a, **k: _response(200, text="<html>hi</html>"))
        ok, msg = _ollama().health_check()
        assert not ok
        assert "not like an Ollama server" in msg

    def test_unreachable(self, monkeypatch):
        def refuse(*a, **k):
            raise httpx.ConnectError("refused")
        monkeypatch.setattr(ollama_module.httpx, "get", refuse)
        ok, msg = _ollama().health_check()
        assert not ok
        assert "Cannot reach Ollama" in msg


class TestOllamaComplete:

    def test_returns_message_content(self, monkeypatch):
        monkeypatch.setattr(
            ollama_module.httpx, "post",
            lambda *a, **k: _response(200, json={"message": {"role": "assistant", "content": "[]"}}),
        )
        assert _ollama().complete("system", "user") == "[]"

    def test_missing_model_gives_pull_hint(self, monkeypatch):
        monkeypatch.setattr(
            ollama_module.httpx, "post",
            lambda *a, **k: _response(404, json={"error": 'model "llama3.1" not found, try pulling it first'}),
        )
        with pytest.raises(RuntimeError, match="ollama pull llama3.1"):
            _ollama().complete("system", "user")

    def test_server_error_shows_ollama_message(self, monkeypatch):
        monkeypatch.setattr(ollama_module.httpx, "post", lambda *a, **k: _response(500, json={"error": "out of memory"}))
        with pytest.raises(RuntimeError, match="500: out of memory"):
            _ollama().complete("system", "user")

    def test_connection_refused(self, monkeypatch):
        def refuse(*a, **k):
            raise httpx.ConnectError("refused")
        monkeypatch.setattr(ollama_module.httpx, "post", refuse)
        with pytest.raises(ConnectionError, match="ollama serve"):
            _ollama().complete("system", "user")
