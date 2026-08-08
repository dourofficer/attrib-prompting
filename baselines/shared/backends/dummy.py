"""Deterministic fake backend for tests and keyless end-to-end runs.

Two modes:

- ``responder``: a callable ``(messages) -> str`` evaluated per request
  (deterministic function of the prompt — safe under any concurrency).
- ``script``: a list of responses consumed in call order (thread-safe); useful
  for scripting multi-round methods in tests. Raises when exhausted.

Default responder emits a response that every method can parse: an
``Agent Name: ... / Step Number: 0`` block that also starts with ``1. Yes`` and
mentions ``upper half``.
"""
from __future__ import annotations

import threading


def _default_responder(messages: list[dict]) -> str:
    return (
        "1. Yes.\n2. Reason: dummy response.\n"
        "Agent Name: DummyAgent\nStep Number: 0\n"
        "Reason for Mistake: dummy (upper half)."
    )


class DummyBackend:
    prefers_streaming = True

    def __init__(self, responder=None, script: list[str] | None = None) -> None:
        if responder is not None and script is not None:
            raise ValueError("pass either responder or script, not both")
        self._responder = responder
        self._script = list(script) if script is not None else None
        self._lock = threading.Lock()
        self.request_params = {"backend": "dummy"}
        self.calls: list[list[dict]] = []  # every request, in call order

    def generate(self, message_lists: list[list[dict]]) -> list[str]:
        out = []
        for messages in message_lists:
            with self._lock:
                self.calls.append(messages)
                if self._script is not None:
                    if not self._script:
                        raise RuntimeError("DummyBackend script exhausted")
                    out.append(self._script.pop(0))
                    continue
            out.append((self._responder or _default_responder)(messages))
        return out
