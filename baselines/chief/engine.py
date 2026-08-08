"""LLM engine for the CHIEF baseline.

CHIEF's vendored implementation calls an OpenAI-compatible endpoint one prompt at
a time (``client.chat.completions.create``). Here we swap that for **local vLLM
inference** by reusing the prompting baseline's :class:`PromptEngine` verbatim — it
already loads the right chat template per architecture (via
``src.models.get_adapter``), toggles ``enable_thinking``, and batches a *list* of
chat-message lists through one ``LLM.chat`` call. Reusing it keeps a single
inference code-path across every local baseline in this repo.

``strip_think`` is re-exported (and a slightly hardened wrapper added) because
CHIEF's stage parsers are strict regexes over plain text and MUST never see a
``<think>`` reasoning block from qwen3.5-9b / deepseek-8b.
"""
from __future__ import annotations

import re

from baselines.prompting.engine import PromptEngine, strip_think as _strip_think

__all__ = ["PromptEngine", "strip_think"]

# Extra safety net on top of prompting's strip_think: some reasoning backbones emit
# a stray opener with no closer, or leftover bare tags. prompting.strip_think already
# handles ``<think>…</think>`` and a dangling ``</think>``; here we additionally drop
# an unterminated ``<think>`` prefix (opener, no closer) and any residual bare tags.
_DANGLING_OPEN = re.compile(r"<think>(?!.*</think>).*\Z", re.DOTALL)
_BARE_TAGS = re.compile(r"</?think>")


def strip_think(text: str) -> str:
    """Remove reasoning traces so CHIEF's regex parsers see only the answer."""
    out = _strip_think(text)
    out = _DANGLING_OPEN.sub("", out)
    out = _BARE_TAGS.sub("", out)
    return out.strip()
