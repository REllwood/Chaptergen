"""Tests for provider configuration resolution."""

import pytest

from chaptergen.config import normalise_ollama_url, resolve_config


class TestNormaliseOllamaUrl:

    @pytest.mark.parametrize(("value", "expected"), [
        ("", "http://localhost:11434"),
        ("0.0.0.0", "http://127.0.0.1:11434"),
        ("0.0.0.0:11434", "http://127.0.0.1:11434"),
        ("localhost", "http://localhost:11434"),
        ("myhost:8080", "http://myhost:8080"),
        ("  gpu-box:11434  ", "http://gpu-box:11434"),
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://localhost:11434/", "http://localhost:11434"),
        ("http://myhost", "http://myhost:80"),
        ("HTTPS://ollama.example.com", "https://ollama.example.com:443"),
        ("https://ollama.example.com/prefix/", "https://ollama.example.com:443/prefix"),
        ("[::1]:11434", "http://[::1]:11434"),
        ("::", "http://[::1]:11434"),
    ])
    def test_forms(self, value, expected):
        assert normalise_ollama_url(value) == expected

    @pytest.mark.parametrize("value", ["myhost:abc", "ftp://myhost"])
    def test_invalid(self, value):
        with pytest.raises(ValueError, match="Invalid Ollama URL"):
            normalise_ollama_url(value)


class TestResolveConfig:

    def test_ollama_host_env_is_normalised(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
        cfg = resolve_config(provider="ollama")
        assert cfg.base_url == "http://127.0.0.1:11434"

    def test_base_url_flag_beats_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0")
        cfg = resolve_config(provider="ollama", base_url="gpu-box:11434")
        assert cfg.base_url == "http://gpu-box:11434"
