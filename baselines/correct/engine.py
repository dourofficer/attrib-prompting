"""LLM engine for the CORRECT baseline.

The vendored implementation (``baselines/CORRECT/src/Lib/local_model.py``)
constructs a raw ``vllm.LLM`` per run with hand-rolled chat templating. Here we
reuse the prompting baseline's :class:`PromptEngine` verbatim — the same single
inference code-path every local baseline in this repo goes through. It batches a
list of chat-message lists through one ``LLM.chat`` call, loads the right chat
template per architecture (via ``src.models.get_adapter``), and toggles
``enable_thinking``.

``strip_think`` is the hardened variant (same as chief's) because CORRECT's
vendored response trimming and parsing regexes are plain-text operations that
MUST never see a ``<think>`` reasoning block from deepseek-8b — a stray
"Agent Name:" inside the reasoning would corrupt the vendored trim.
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
    """Remove reasoning traces so the vendored trim/parsers see only the answer."""
    out = _strip_think(text)
    out = _DANGLING_OPEN.sub("", out)
    out = _BARE_TAGS.sub("", out)
    return out.strip()
