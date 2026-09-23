"""Tests for provider configuration resolution."""

import pytest

from chaptergen.config import DEFAULT_OLLAMA_MODEL, DEFAULT_OPENAI_MODEL, normalise_ollama_url, resolve_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ["CHAPTERGEN_PROVIDER", "CHAPTERGEN_MODEL", "OLLAMA_HOST", "OPENAI_API_KEY", "OPENAI_BASE_URL"]:
        monkeypatch.delenv(name, raising=False)


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

    def test_defaults(self):
        cfg = resolve_config()
        assert (cfg.provider, cfg.model, cfg.base_url) == ("ollama", DEFAULT_OLLAMA_MODEL, "http://localhost:11434")
        assert cfg.temperature is None

    def test_openai_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        cfg = resolve_config(provider="openai")
        assert (cfg.model, cfg.api_key, cfg.base_url) == (DEFAULT_OPENAI_MODEL, "sk-env", None)

    def test_provider_env_and_flag_priority(self, monkeypatch):
        monkeypatch.setenv("CHAPTERGEN_PROVIDER", "OpenAI")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        assert resolve_config().provider == "openai"
        assert resolve_config(provider="ollama").provider == "ollama"

    def test_model_env_and_flag_priority(self, monkeypatch):
        monkeypatch.setenv("CHAPTERGEN_MODEL", "mistral")
        assert resolve_config().model == "mistral"
        assert resolve_config(model="qwen3").model == "qwen3"

    def test_api_key_priority(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-default")
        monkeypatch.setenv("MY_KEY", "sk-custom")
        assert resolve_config(provider="openai").api_key == "sk-default"
        assert resolve_config(provider="openai", api_key_env="MY_KEY").api_key == "sk-custom"
        assert resolve_config(provider="openai", api_key="sk-flag", api_key_env="MY_KEY").api_key == "sk-flag"

    def test_api_key_env_not_set(self):
        with pytest.raises(ValueError, match="MY_MISSING_KEY"):
            resolve_config(provider="openai", api_key_env="MY_MISSING_KEY")

    def test_openai_base_url_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        assert resolve_config(provider="openai").base_url == "http://localhost:1234/v1"

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider 'gemini'"):
            resolve_config(provider="gemini")
