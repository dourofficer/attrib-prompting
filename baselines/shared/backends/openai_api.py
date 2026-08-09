"""OpenAI-compatible chat-completions backend.

Covers the OpenAI API itself (GPT-5, GPT-4o, ...) and any provider speaking the
same protocol (OpenRouter, DeepSeek, a ``vllm serve`` endpoint, ...) via
``base_url``.

Design points:

- **Param mapping is config-declared, not hardcoded.** The model spec's
  ``params`` dict is sent verbatim alongside ``model`` and ``messages`` — e.g.
  ``{max_tokens, temperature, top_p, seed}`` for GPT-4o-class models,
  ``{max_completion_tokens}`` (and *no* temperature) for GPT-5-class reasoning
  models, which reject one. The run snapshot records exactly what was sent, so
  the repo never pretends a model honored a parameter it never received.
- **Bounded concurrency** across ALL in-flight requests via a semaphore
  (``concurrency``), regardless of whether calls arrive from one big
  ``generate`` batch or many per-trajectory threads.
- **Retries with exponential backoff + jitter** on transient failures (429,
  5xx, timeouts, connection errors). Non-transient API errors (400 invalid
  param, 401 auth) raise immediately. After ``max_retries`` transient failures
  the exception propagates; the streaming runner then skips that trajectory so
  a rerun resumes it.
- **Injectable client** (any object with ``chat.completions.create``) so tests
  run without the ``openai`` package or an API key.
"""
from __future__ import annotations

import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
_RETRYABLE_NAMES = ("timeout", "connection", "ratelimit", "internalserver", "apiconnection")


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in _RETRYABLE_STATUS
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return any(k in type(exc).__name__.lower() for k in _RETRYABLE_NAMES)


class OpenAIBackend:
    prefers_streaming = True

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        headers: dict[str, str] | None = None,
        params: dict | None = None,
        concurrency: int = 8,
        max_retries: int = 6,
        timeout: float = 300.0,
        client=None,
    ) -> None:
        self.model = model
        self.params = dict(params or {})
        self.max_retries = max_retries
        self._sem = threading.BoundedSemaphore(concurrency)
        self.concurrency = concurrency
        if client is None:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"API key environment variable {api_key_env!r} is not set "
                    f"(needed for model {model!r})."
                )
            import openai  # deferred: only needed for real API runs

            # The backend owns retry behaviour; disable the SDK's built-in retries.
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=dict(headers or {}),
                timeout=timeout,
                max_retries=0,
            )
        self.client = client

    @property
    def request_params(self) -> dict:
        return {"model": self.model, **self.params}

    def generate(self, message_lists: list[list[dict]]) -> list[str]:
        if not message_lists:
            return []
        if len(message_lists) == 1:
            return [self._one(message_lists[0])]
        # Order-preserving fan-out; total in-flight requests are still bounded
        # by the shared semaphore even when called from many trajectory threads.
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self._one, message_lists))

    def _one(self, messages: list[dict]) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._sem:
                    resp = self.client.chat.completions.create(
                        model=self.model, messages=messages, **self.params
                    )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 — classified below
                if not _is_retryable(exc) or attempt == self.max_retries:
                    raise
                last_exc = exc
                delay = min(2.0 ** attempt + random.random(), 60.0)
                time.sleep(delay)
        raise last_exc  # pragma: no cover — loop always returns or raises
