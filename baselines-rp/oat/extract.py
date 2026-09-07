"""Read step vectors out of a frozen LLM.

One forward pass per trajectory over the document ``serialize.py`` built. The
model returns a hidden vector for every token; a step's vector is the mean of
the vectors of the tokens inside its character span. That is the whole of
"representation-based": no generation, no sampling, no prompt.

The pooling and token-span code is verbatim from
``vendored/OAT/{data_pipeline,extract_states}.py``. Three things are adapted,
each for a reason the vendored code could ignore because it targeted one fixed
checkpoint on one machine:

- **Only the layers we asked for are kept.** The vendored code stores all 65
  layers of every trajectory — tens of gigabytes for this repo's 2,586
  trajectories. Passing ``layers=None`` restores the vendored behaviour and is
  what the parity test uses.
- **The language-model head never runs.** The vendored code calls the full
  ``AutoModelForCausalLM``; on an 81k-token log with a 248k-token vocabulary
  its logits alone would be ~40 GB, and they are thrown away. We call the
  decoder underneath, which produces the identical hidden states.
- **Checkpoints are loaded by architecture.** ``Qwen3.5-9B`` on this machine is
  a vision-language checkpoint whose text decoder sits one level down; loading
  it as a plain causal LM silently leaves weights uninitialized.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

# vendored/OAT/config.py:23 — the tokenizer truncation cap.
MAX_SEQ_LENGTH = 262144
DEFAULT_LAYER = -1
DEFAULT_AGGREGATION = "mean"


# --------------------------------------------------------------------------
# Token spans (verbatim: vendored/OAT/data_pipeline.py:220-252)
# --------------------------------------------------------------------------

def get_step_last_token_indices(
    offset_mapping: Sequence[tuple[int, int]],
    char_boundaries: Sequence[tuple[int, int]],
) -> list[int]:
    results: list[int] = []
    for _start_char, end_char in char_boundaries:
        last_tok = 0
        lo, hi = 0, len(offset_mapping) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            tok_start, tok_end = offset_mapping[mid]
            if tok_start == 0 and tok_end == 0:
                lo = mid + 1
            elif tok_start <= end_char:
                last_tok = mid
                lo = mid + 1
            else:
                hi = mid - 1
        results.append(last_tok)
    return results


def get_step_token_ranges(
    offset_mapping: Sequence[tuple[int, int]],
    char_boundaries: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    n_tokens = len(offset_mapping)
    for start_char, end_char in char_boundaries:
        first_tok = n_tokens - 1
        last_tok = 0
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == 0 and tok_end == 0:
                continue
            if tok_end >= start_char and tok_start <= end_char:
                first_tok = min(first_tok, i)
                last_tok = max(last_tok, i)
        ranges.append((first_tok, last_tok))
    return ranges


# --------------------------------------------------------------------------
# The forward pass (vendored/OAT/extract_states.py:55-115, layer-sliced)
# --------------------------------------------------------------------------

@torch.inference_mode()
def extract_hidden_states_for_trajectory(
    text: str,
    char_boundaries: list,
    model,
    tokenizer,
    max_length: int = MAX_SEQ_LENGTH,
    aggregation: str = DEFAULT_AGGREGATION,
    layers: Optional[Sequence[int]] = (DEFAULT_LAYER,),
) -> Optional[torch.Tensor]:
    """Return a ``(T, len(layers), hidden_dim)`` float32 tensor, or ``None``.

    ``None`` means truncation swallowed every step — the vendored signal to
    skip the trajectory. ``layers=None`` keeps every layer, which is what the
    vendored code does.
    """
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"][0].tolist()
    seq_len = input_ids.shape[1]

    if seq_len >= max_length:
        max_char_covered = offset_mapping[-1][1] if offset_mapping[-1][1] > 0 else 0
        valid_steps = [i for i, (_, end) in enumerate(char_boundaries) if end <= max_char_covered]
        if not valid_steps:
            return None
        char_boundaries = char_boundaries[: valid_steps[-1] + 1]

    if aggregation == "mean":
        step_locations = get_step_token_ranges(offset_mapping, char_boundaries)
    elif aggregation == "last":
        step_locations = get_step_last_token_indices(offset_mapping, char_boundaries)
    else:
        raise ValueError("aggregation must be 'last' or 'mean'")

    input_device = _model_input_device(model)
    # Asking for every layer costs ~22 GB on an 80k-token log at 4,096
    # dimensions, and the default wants exactly one of them. A decoder's
    # `last_hidden_state` is bit-identical to `hidden_states[-1]` — both are the
    # output of the final norm — so the common case skips the whole stack.
    want_last_only = layers is not None and list(layers) == [-1]
    outputs = model(
        input_ids=input_ids.to(input_device),
        attention_mask=encoding["attention_mask"].to(input_device),
        output_hidden_states=not want_last_only,
        use_cache=False,
        return_dict=True,
    )
    if want_last_only:
        all_hidden = [outputs.last_hidden_state]
    else:
        all_hidden = outputs.hidden_states
        if layers is not None:
            all_hidden = [all_hidden[i] for i in layers]

    T = len(char_boundaries)
    num_layers = len(all_hidden)
    hidden_dim = all_hidden[0].shape[-1]

    if aggregation == "last":
        token_ids = torch.tensor(step_locations, device=input_device)
        return torch.stack([layer_hs[0].index_select(0, token_ids) for layer_hs in all_hidden], dim=1).float().cpu()

    result = torch.zeros(T, num_layers, hidden_dim, dtype=torch.float32)
    for layer_idx, layer_hs in enumerate(all_hidden):
        layer_hs = layer_hs[0].float()
        for step_idx, (start_tok, end_tok) in enumerate(step_locations):
            if start_tok <= end_tok:
                result[step_idx, layer_idx] = layer_hs[start_tok : end_tok + 1].mean(dim=0).cpu()
            else:
                result[step_idx, layer_idx] = layer_hs[end_tok].cpu()
    return result


def _model_input_device(model) -> torch.device:
    """Vendored ``_get_model_input_device`` (``extract_states.py:22-30``)."""
    if hasattr(model, "hf_device_map"):
        for _, device in model.hf_device_map.items():
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# --------------------------------------------------------------------------
# Loading a checkpoint, and the dummy stand-in
# --------------------------------------------------------------------------
# Both live in ``rb_shared`` now: StepFinder freezes the same checkpoints and
# needs the same GPU-free stand-in, and one copy is what keeps the two
# baselines loading ``Qwen3.5-9B`` the same way. Re-exported here because this
# module is the import path OAT and its tests were written against.

from rb_shared.dummy import _DummyModel, _DummyTokenizer  # noqa: E402,F401
from rb_shared.extractors import (  # noqa: E402,F401
    _DTYPES,
    _text_decoder,
    Extractor,
    dummy_extractor,
    load_extractor,
)
