"""Load a frozen local checkpoint and hand back the bare decoder.

Both representation-based baselines start the same way: take a local model,
freeze it, run text through it once, and keep the vectors. What differs is
what they do next — OAT pools the tokens of a step's character span, StepFinder
pools the last token of a whole step — so the loading, and only the loading,
lives here.

Three things are adapted from the vendored code each baseline came with, for
reasons a paper targeting one fixed checkpoint on one machine could ignore:

- **The language-model head never runs.** We call the decoder underneath the
  causal-LM wrapper, which produces the identical hidden states. On an
  81k-token log with a 248k-token vocabulary the logits alone would be ~40 GB,
  and they are thrown away.
- **Checkpoints are loaded by architecture.** ``Qwen3.5-9B`` on this machine is
  a vision-language checkpoint whose text decoder sits one level down; loading
  it as a plain causal LM silently leaves weights uninitialized.
- **The truncation cap follows the checkpoint**, not a constant, so a model
  with a shorter context window is not asked for more than it has.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# vendored/OAT/config.py:23 — the tokenizer truncation cap.
MAX_SEQ_LENGTH = 262144


@dataclass
class Extractor:
    """A frozen model, its tokenizer, and the limits that come with them."""

    name: str
    model: object
    tokenizer: object
    max_length: int
    hidden_dim: int
    num_layers: int


_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def load_extractor(name: str, spec: dict) -> Extractor:
    """Load a local checkpoint and hand back the bare decoder.

    ``spec`` keys: ``path`` (required), ``tokenizer`` (defaults to ``path``),
    ``dtype`` (``bf16``/``fp16``/``fp32``), ``device``, ``max_length``
    (caps the value read off the checkpoint — StepFinder pins 8192 because
    ``vendored/StepFinder/Q3Emb.py:41`` does).

    The decoder, not the causal-LM wrapper, is what gets returned. Its hidden
    states are the same tensors; skipping the head is what keeps an 81k-token
    forward pass inside one GPU.
    """
    from transformers import AutoConfig, AutoTokenizer

    path = spec["path"]
    device = spec.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, _DTYPES.get(spec.get("dtype", "bf16"), "bfloat16"))
    if device == "cpu":
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(spec.get("tokenizer") or path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    architectures = list(getattr(cfg, "architectures", None) or [])
    is_multimodal = any(a.endswith("ForConditionalGeneration") for a in architectures)

    kwargs = {"dtype": dtype, "trust_remote_code": True}
    if spec.get("attn_implementation"):
        kwargs["attn_implementation"] = spec["attn_implementation"]

    if is_multimodal:
        from transformers import AutoModelForImageTextToText

        wrapper = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
        decoder = _text_decoder(wrapper)
    else:
        from transformers import AutoModelForCausalLM

        wrapper = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        decoder = getattr(wrapper, "model", wrapper)

    decoder.config.use_cache = False
    decoder.eval()
    decoder.to(device)

    text_cfg = getattr(cfg, "text_config", cfg)
    cap = int(spec.get("max_length") or MAX_SEQ_LENGTH)
    max_length = min(cap, int(getattr(text_cfg, "max_position_embeddings", cap)))
    return Extractor(
        name=name,
        model=decoder,
        tokenizer=tokenizer,
        max_length=max_length,
        hidden_dim=int(getattr(text_cfg, "hidden_size", 0)),
        num_layers=int(getattr(text_cfg, "num_hidden_layers", 0)),
    )


def _text_decoder(wrapper):
    """Dig the text decoder out of a vision-language wrapper."""
    for attr in ("language_model", "model"):
        inner = getattr(wrapper, attr, None)
        if inner is None:
            continue
        deeper = getattr(inner, "language_model", None)
        return deeper if deeper is not None else inner
    return wrapper


def dummy_extractor(name: str = "dummy", hidden_dim: int = 32, num_layers: int = 2) -> Extractor:
    """A checkpoint-free, GPU-free stand-in whose states are reproducible."""
    from rb_shared.dummy import DummyModel, DummyTokenizer

    return Extractor(
        name=name,
        model=DummyModel(hidden_dim, num_layers),
        tokenizer=DummyTokenizer(),
        max_length=MAX_SEQ_LENGTH,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
