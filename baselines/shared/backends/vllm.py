"""Local-checkpoint backend via vLLM (the original ``PromptEngine``).

Single point where all batching happens: :meth:`VllmBackend.generate` takes a
list of chat-message lists and issues **one** batched ``LLM.chat`` call.

The vendored Who&When baseline (``Agents_Failure_Attribution``) calls the model
one message-list at a time through HuggingFace ``transformers``; here we keep the
exact prompts/algorithms (see ``methods.py``) but let vLLM batch across
trajectories.

Thinking is a *config toggle* (``enable_thinking``), not hardcoded. We start from
the architecture's template kwargs (e.g. Qwen3.5 defaults thinking off) and
override ``enable_thinking`` with the run's value, so it can be turned on/off per
run when the checkpoint's chat template supports it. Templates that do not accept
the key simply ignore it (e.g. DeepSeek-R1-Distill, which always reasons).
Regardless of the flag, ``methods.strip_think`` removes any ``<think>...</think>``
block before parsing so the downstream parsers stay robust.
"""
from __future__ import annotations

from typing import Any

_DTYPE_MAP = {
    "float32": "float32",
    "bfloat16": "bfloat16",
    "float16": "float16",
    "auto": "auto",
}


def _template_kwargs(model_path: str) -> dict[str, Any]:
    """Architecture-default chat-template kwargs for a local checkpoint.

    Condensed from the attribscope model adapters: Qwen3.5 chat templates take
    ``enable_thinking`` (default off for the baselines); every other supported
    architecture takes no template kwargs.
    """
    from transformers import AutoConfig

    model_type = AutoConfig.from_pretrained(model_path, trust_remote_code=True).model_type
    if model_type == "qwen3_5":
        return {"enable_thinking": False}
    return {}


class VllmBackend:
    """Thin batched-inference wrapper around ``vllm.LLM``."""

    prefers_streaming = False

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
        # Imported lazily so the module can be imported without a vLLM install,
        # e.g. for --help, the factory, and prompt-parity tests.
        from vllm import LLM, SamplingParams

        self.model_path = model_path

        # Chat-template kwargs: architecture default, then the run's toggle wins.
        self._template_kwargs: dict[str, Any] = dict(_template_kwargs(model_path))
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
        # truncate_prompt_tokens (optional safety net): if set, vLLM keeps only the
        # last N prompt tokens instead of erroring on over-length prompts. Off by
        # default — the faithful behaviour is to size max_model_len to cover the
        # data (both target checkpoints natively support ≥128k; the longest
        # trajectory prompt is ~98k tokens). It is passed via tokenization_kwargs
        # on chat() (it is not a SamplingParams field in this vLLM version).
        self._tokenization_kwargs = (
            {"truncate_prompt_tokens": truncate_prompt_tokens}
            if truncate_prompt_tokens is not None else None
        )
        self.request_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_gen_tokens,
            "seed": seed,
            "dtype": dtype,
            "enable_thinking": enable_thinking,
            "truncate_prompt_tokens": truncate_prompt_tokens,
        }

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
