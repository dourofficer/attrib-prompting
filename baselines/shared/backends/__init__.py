"""Inference backends shared by all baselines.

A backend is anything with:

- ``generate(message_lists: list[list[dict]]) -> list[str]`` — OpenAI-style
  chat message lists in, decoded texts out, order-preserving. This is the whole
  coupling surface between the methods and the model.
- ``request_params: dict`` — the exact generation parameters the backend sends,
  recorded verbatim in the run snapshot (``_run.json``).
- ``prefers_streaming: bool`` — ``False`` for local engines that want one giant
  cross-trajectory batch (vLLM), ``True`` for per-request engines where
  trajectories should complete (and be written) independently.

Backends:

- ``vllm``   — local checkpoint via vLLM (:class:`~.vllm.VllmBackend`).
- ``openai`` — any OpenAI-compatible chat-completions API
  (:class:`~.openai_api.OpenAIBackend`): OpenAI itself, or another provider via
  ``base_url``. Adding a new OpenAI-compatible provider is a config entry, not
  code; a differently-shaped API means adding one new module here implementing
  the three-member surface above and one branch in :func:`get_backend`.
- ``dummy``  — deterministic canned responses for tests and keyless end-to-end
  runs (:class:`~.dummy.DummyBackend`).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    request_params: dict
    prefers_streaming: bool

    def generate(self, message_lists: list[list[dict]]) -> list[str]: ...


def get_backend(kind: str, **opts) -> Backend:
    """Construct a backend by name; opts are backend-specific ctor kwargs."""
    if kind == "vllm":
        from .vllm import VllmBackend
        return VllmBackend(**opts)
    if kind == "openai":
        from .openai_api import OpenAIBackend
        return OpenAIBackend(**opts)
    if kind == "dummy":
        from .dummy import DummyBackend
        return DummyBackend(**opts)
    raise ValueError(f"unknown backend {kind!r} (expected vllm | openai | dummy)")
