"""Back-compat shim — the engine moved.

``PromptEngine`` now lives at :class:`baselines.prompting.backends.vllm.VllmBackend`
and ``strip_think`` at :func:`baselines.prompting.methods.strip_think`. This shim
keeps the historical import path working for ``baselines/chief`` and
``baselines/correct`` until they are adapted to the backend interface.
"""
from __future__ import annotations

from .backends.vllm import VllmBackend as PromptEngine
from .methods import strip_think

__all__ = ["PromptEngine", "strip_think"]
