"""Tests for the LLM provider implementations (network calls are faked)."""

import httpx
import pytest

from chaptergen.config import ProviderConfig
from chaptergen.providers import ollama as ollama_module
from chaptergen.providers.ollama import OllamaProvider, _context_size


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


class TestOllamaContextWindow:

    def test_short_prompt_still_leaves_room_for_reply(self):
        size = _context_size(2_000, None)
        assert 2_000 // 3 + 4096 <= size <= 8192
        assert size % 2048 == 0

    def test_long_prompt_gets_room_for_transcript_and_reply(self):
        size = _context_size(90_000, None)
        assert size >= 90_000 // 3 + 4096
        assert size % 2048 == 0

    def test_capped_at_model_limit(self):
        assert _context_size(1_000_000, 8192) == 8192

    def test_request_sets_num_ctx_from_model_limit(self, monkeypatch):
        sent = []

        def fake_post(url, json, timeout):
            sent.append((url, json))
            if url.endswith("/api/show"):
                return _response(200, json={"model_info": {"llama.context_length": 8192}})
            return _response(200, json={"message": {"content": "[]"}})

        monkeypatch.setattr(ollama_module.httpx, "post", fake_post)
        provider = _ollama()
        provider.complete("system", "x" * 100_000)
        provider.complete("system", "short")
        chat_calls = [body for url, body in sent if url.endswith("/api/chat")]
        assert chat_calls[0]["options"]["num_ctx"] == 8192
        assert chat_calls[1]["options"]["num_ctx"] == _context_size(len("system") + len("short"), 8192)
        assert sum(url.endswith("/api/show") for url, _ in sent) == 1

    def test_timeout_gives_clear_error(self, monkeypatch):
        def slow(*a, **k):
            raise httpx.ReadTimeout("timed out")
        monkeypatch.setattr(ollama_module.httpx, "post", slow)
        with pytest.raises(RuntimeError, match="didn't respond within 10 minutes"):
            _ollama().complete("system", "user")


class FakeOpenAIClient:
    """Stands in for openai.OpenAI: records requests and replays scripted results."""

    def __init__(self, *results, models=("gpt-5-mini", "gpt-4o-mini")):
        self.requests: list[dict] = []
        self._results = list(results)
        self._models = models
        self.chat = self
        self.completions = self

    def create(self, **request):
        self.requests.append(request)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def list(self):
        from types import SimpleNamespace
        return [SimpleNamespace(id=m) for m in self._models]

    @property
    def models(self):
        return self


def _completion(content: str):
    from types import SimpleNamespace
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _openai(client: FakeOpenAIClient, model: str = "gpt-5-mini"):
    pytest.importorskip("openai")
    from chaptergen.providers.openai import OpenAIProvider

    provider = OpenAIProvider(ProviderConfig(provider="openai", model=model, api_key="sk-test"))
    provider._client = client
    return provider


def _temperature_error():
    import openai

    return openai.BadRequestError(
        "Error code: 400",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")),
        body={
            "message": "Unsupported value: 'temperature' does not support 0.0 with this model.",
            "param": "temperature",
            "code": "unsupported_value",
        },
    )


class TestOpenAIProvider:

    def test_temperature_omitted_by_default(self):
        client = FakeOpenAIClient(_completion("[]"))
        assert _openai(client).complete("system", "user") == "[]"
        assert "temperature" not in client.requests[0]

    def test_temperature_sent_when_given(self):
        client = FakeOpenAIClient(_completion("[]"))
        _openai(client, model="gpt-4o-mini").complete("system", "user", temperature=0.2)
        assert client.requests[0]["temperature"] == 0.2

    def test_unsupported_temperature_retried_without_it(self):
        client = FakeOpenAIClient(_temperature_error(), _completion("[]"), _completion("[]"))
        provider = _openai(client)
        assert provider.complete("system", "user", temperature=0.0) == "[]"
        assert "temperature" in client.requests[0]
        assert "temperature" not in client.requests[1]
        # Remembered for later calls, e.g. the repair retry
        provider.complete("system", "user", temperature=0.0)
        assert "temperature" not in client.requests[2]

    def test_other_bad_requests_still_raise(self):
        import openai

        error = openai.BadRequestError(
            "Error code: 400",
            response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")),
            body={"message": "Context too long", "param": "messages", "code": "context_length_exceeded"},
        )
        client = FakeOpenAIClient(error)
        with pytest.raises(openai.BadRequestError):
            _openai(client).complete("system", "user", temperature=0.0)

    def test_health_check_model_available(self):
        ok, msg = _openai(FakeOpenAIClient()).health_check()
        assert ok, msg

    def test_health_check_model_missing(self):
        ok, msg = _openai(FakeOpenAIClient(models=("gpt-4o-mini",))).health_check()
        assert not ok
        assert "isn't available" in msg

    def test_health_check_server_without_model_list(self):
        ok, msg = _openai(FakeOpenAIClient(models=())).health_check()
        assert ok
        assert "couldn't be confirmed" in msg


class TestOllamaTemperature:

    def test_defaults_to_zero(self, monkeypatch):
        sent = []
        monkeypatch.setattr(
            ollama_module.httpx, "post",
            lambda url, json, timeout: sent.append(json) or _response(200, json={"message": {"content": "[]"}}),
        )
        _ollama().complete("system", "user")
        _ollama().complete("system", "user", temperature=0.7)
        chat = [body for body in sent if "messages" in body]
        assert chat[0]["options"]["temperature"] == 0.0
        assert chat[1]["options"]["temperature"] == 0.7


HISTORY = [{"role": "user", "content": "transcript"}, {"role": "assistant", "content": "oops"}]


class TestConversationHistory:

    def test_ollama_sends_history_between_system_and_user(self, monkeypatch):
        sent = []
        monkeypatch.setattr(
            ollama_module.httpx, "post",
            lambda url, json, timeout: sent.append(json) or _response(200, json={"message": {"content": "[]"}}),
        )
        _ollama().complete("rules", "fix it", history=HISTORY)
        chat = next(body for body in sent if "messages" in body)
        assert [m["content"] for m in chat["messages"]] == ["rules", "transcript", "oops", "fix it"]

    def test_openai_sends_history_between_system_and_user(self):
        client = FakeOpenAIClient(_completion("[]"))
        _openai(client).complete("rules", "fix it", history=HISTORY)
        assert [m["content"] for m in client.requests[0]["messages"]] == ["rules", "transcript", "oops", "fix it"]
