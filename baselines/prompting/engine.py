"""VLLM inference wrapper for the prompting baselines.

Single point where all batching happens: :meth:`PromptEngine.generate` takes a
list of chat-message lists and issues **one** batched ``LLM.chat`` call.

The vendored Who&When baseline (``Agents_Failure_Attribution``) calls the model
one message-list at a time through HuggingFace ``transformers``; here we keep the
exact prompts/algorithms (see ``methods.py``) but let VLLM batch across
trajectories, which is the whole point of the rewrite.

Thinking is a *config toggle* (``enable_thinking``), not hardcoded. We start from
the architecture adapter's ``template_kwargs()`` (e.g. Qwen3.5 defaults thinking
off) and override ``enable_thinking`` with the run's value, so it can be turned
on/off per run when the checkpoint's chat template supports it. Templates that do
not accept the key simply ignore it (e.g. DeepSeek-R1-Distill, which always
reasons). Regardless of the flag, :func:`strip_think` removes any
``<think>...</think>`` block before parsing so the downstream parsers stay robust.
"""
from __future__ import annotations

import re
from typing import Any

from src.models import get_adapter

# ``<think> ... </think>`` (DOTALL). Also handles a dangling ``</think>`` with no
# opening tag (some templates inject the opener into the prompt, so the model only
# emits the closer).
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_DANGLING_CLOSE = re.compile(r"^.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove reasoning traces so the baseline parsers see only the answer."""
    if text is None:
        return ""
    out = _THINK_BLOCK.sub("", text)
    if "</think>" in out:
        out = _DANGLING_CLOSE.sub("", out)
    return out.strip()


_DTYPE_MAP = {
    "float32": "float32",
    "bfloat16": "bfloat16",
    "float16": "float16",
    "auto": "auto",
}


class PromptEngine:
    """Thin batched-inference wrapper around ``vllm.LLM``."""

    def __init__(
        self,
        model_path: str,
        *,
        tokenizer: str | None = None,
        dtype: str = "bfloat16",
        seed: int = 0,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_gen_tokens: int = 1024,
        enable_thinking: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        truncate_prompt_tokens: int | None = None,
    ) -> None:
        # Imported lazily so the module (and strip_think / methods parsing) can be
        # used without a VLLM install, e.g. for --help and prompt-parity tests.
        from vllm import LLM, SamplingParams

        self.model_path = model_path

        # Chat-template kwargs: adapter default, then let the run's toggle win.
        self._template_kwargs: dict[str, Any] = dict(get_adapter(model_path).template_kwargs())
        self._template_kwargs["enable_thinking"] = enable_thinking

        # tokenizer override: some checkpoints ship a tokenizer_config that makes
        # AutoTokenizer build the wrong (e.g. SentencePiece) tokenizer; pass a
        # corrected tokenizer dir here to fix decoding without touching weights.
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer,
            dtype=_DTYPE_MAP.get(dtype, dtype),
            seed=seed,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_gen_tokens,
            seed=seed,
        )
        # truncate_prompt_tokens (optional safety net): if set, VLLM keeps only the
        # last N prompt tokens instead of erroring on over-length prompts. Off by
        # default — the faithful behaviour is to size max_model_len to cover the
        # data (both target checkpoints natively support ≥128k; the longest
        # trajectory prompt is ~98k tokens). It is passed via tokenization_kwargs
        # on chat() (it is not a SamplingParams field in this VLLM version).
        self._tokenization_kwargs = (
            {"truncate_prompt_tokens": truncate_prompt_tokens}
            if truncate_prompt_tokens is not None else None
        )

    def generate(self, message_lists: list[list[dict]]) -> list[str]:
        """Batched chat generation.

        Parameters
        ----------
        message_lists : list of OpenAI-style chat message lists
            e.g. ``[[{"role": "system", ...}, {"role": "user", ...}], ...]``

        Returns
        -------
        list[str] — decoded text per input, in the same order.
        """
        if not message_lists:
            return []
        chat_kwargs: dict[str, Any] = dict(
            chat_template_kwargs=self._template_kwargs,
            use_tqdm=True,
        )
        if self._tokenization_kwargs is not None:
            chat_kwargs["tokenization_kwargs"] = self._tokenization_kwargs
        outputs = self.llm.chat(message_lists, self.sampling_params, **chat_kwargs)
        return [o.outputs[0].text for o in outputs]
