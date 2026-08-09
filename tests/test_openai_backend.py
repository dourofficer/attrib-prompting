"""OpenAI-compatible backend, exercised through an injected fake client."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from baselines.shared.backends.openai_api import OpenAIBackend


def _resp(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeClient:
    def __init__(self, handler):
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

        def create(**kwargs):
            with self._lock:
                self.requests.append(kwargs)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                return handler(kwargs)
            finally:
                with self._lock:
                    self.active -= 1

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class Err(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


MESSAGES = [{"role": "user", "content": "hi"}]


def test_payload_is_exactly_spec_params():
    gpt5 = FakeClient(lambda kw: _resp("ok"))
    be = OpenAIBackend("gpt-5", params={"max_completion_tokens": 4096}, client=gpt5)
    assert be.generate([MESSAGES]) == ["ok"]
    (req,) = gpt5.requests
    assert req == {"model": "gpt-5", "messages": MESSAGES, "max_completion_tokens": 4096}
    assert "temperature" not in req  # never invented

    gpt4o = FakeClient(lambda kw: _resp("ok"))
    params = {"max_tokens": 1024, "temperature": 0.6, "top_p": 0.95, "seed": 0}
    be = OpenAIBackend("gpt-4o", params=params, client=gpt4o)
    be.generate([MESSAGES])
    (req,) = gpt4o.requests
    assert req == {"model": "gpt-4o", "messages": MESSAGES, **params}
    assert be.request_params == {"model": "gpt-4o", **params}


def test_retry_on_429_then_success(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    attempts = []

    def handler(kw):
        attempts.append(1)
        if len(attempts) < 3:
            raise Err(429)
        return _resp("recovered")

    be = OpenAIBackend("m", client=FakeClient(handler), max_retries=6)
    assert be.generate([MESSAGES]) == ["recovered"]
    assert len(attempts) == 3


def test_non_retryable_raises_immediately(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    client = FakeClient(lambda kw: (_ for _ in ()).throw(Err(400)))
    be = OpenAIBackend("m", client=client)
    with pytest.raises(Err):
        be.generate([MESSAGES])
    assert len(client.requests) == 1


def test_retries_exhausted_raises(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    client = FakeClient(lambda kw: (_ for _ in ()).throw(Err(503)))
    be = OpenAIBackend("m", client=client, max_retries=4)
    with pytest.raises(Err):
        be.generate([MESSAGES])
    assert len(client.requests) == 5  # first try + 4 retries


def test_concurrency_bounded():
    def slow(kw):
        time.sleep(0.02)
        return _resp("ok")

    client = FakeClient(slow)
    be = OpenAIBackend("m", client=client, concurrency=2)
    out = be.generate([MESSAGES] * 8)
    assert out == ["ok"] * 8
    assert client.max_active <= 2


def test_missing_key_env_errors():
    with pytest.raises(RuntimeError, match="NOT_A_REAL_KEY_ENV"):
        OpenAIBackend("gpt-4o", api_key_env="NOT_A_REAL_KEY_ENV")


def test_headers_are_forwarded_to_sdk(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("TEST_API_KEY", "secret")
    OpenAIBackend(
        "gpt-5",
        api_key_env="TEST_API_KEY",
        headers={"X-Llmhub-Channel": "1"},
    )
    assert captured["default_headers"] == {"X-Llmhub-Channel": "1"}
