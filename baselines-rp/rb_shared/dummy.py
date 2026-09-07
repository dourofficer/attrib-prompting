"""A model and tokenizer that need no checkpoint, no download and no GPU.

Every CLI test in this repo runs the real pipeline end to end. That is only
affordable because the frozen encoder can be swapped for something that
answers in microseconds and answers the *same way every time*: hidden states
seeded from the token text, so a rerun reproduces the previous run's
predictions exactly and a resume test can tell "already done" from "done
again differently".

Shared by both representation-based baselines, which ask different things of
it: OAT tokenizes one long document and needs honest character offsets;
StepFinder tokenizes one short step at a time and needs last-token pooling
over a left-padded batch. Both paths are here.
"""

from __future__ import annotations

import hashlib

import torch


class DummyTokenizer:
    """Splits on whitespace and reports honest character offsets.

    Accepts a single string (OAT's document) or a list of strings (a
    StepFinder batch). Padding is on the left, matching the setting
    Qwen3-Embedding requires (``vendored/StepFinder/Q3Emb.py:38``), so
    last-token pooling reads the final real token.
    """

    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"

    def __call__(
        self,
        text,
        return_tensors=None,
        truncation=True,
        max_length=None,
        padding=False,
        return_offsets_mapping=False,
    ):
        texts = [text] if isinstance(text, str) else list(text)
        per_text = [self._offsets(t, max_length) for t in texts]

        width = max((len(o) for o in per_text), default=0)
        if not padding:
            width = len(per_text[0]) if per_text else 0

        ids, mask, offsets = [], [], []
        for text_i, offs in zip(texts, per_text):
            row = [abs(hash(text_i[s:e])) % 1000 for s, e in offs]
            pad = width - len(row)
            # left padding: the real tokens end at the last column
            ids.append([0] * pad + row)
            mask.append([0] * pad + [1] * len(row))
            offsets.append([(0, 0)] * pad + list(offs))

        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }
        if return_offsets_mapping:
            out["offset_mapping"] = torch.tensor(offsets, dtype=torch.long)
        return out

    @staticmethod
    def _offsets(text: str, max_length):
        offsets, cursor = [], 0
        for token in text.split(" "):
            if token:
                start = text.index(token, cursor)
                offsets.append((start, start + len(token)))
                cursor = start + len(token)
        if max_length is not None:
            offsets = offsets[:max_length]
        return offsets


class DummyModel(torch.nn.Module):
    """Hidden states seeded by the token text, so runs are reproducible."""

    def __init__(self, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.config = type("cfg", (), {"use_cache": False})()
        self._anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=True, use_cache=False, return_dict=True):
        rows, n = int(input_ids.shape[0]), int(input_ids.shape[1])
        hidden = []
        for layer in range(self.num_layers + 1):
            per_row = []
            for r in range(rows):
                gen = torch.Generator().manual_seed(
                    int(hashlib.sha1(f"{layer}:{input_ids[r].sum().item()}:{n}".encode()).hexdigest()[:8], 16)
                )
                per_row.append(torch.randn(1, n, self.hidden_dim, generator=gen))
            hidden.append(torch.cat(per_row, dim=0))
        # mirrors a real decoder: the last layer is also `last_hidden_state`,
        # and the full stack appears only when it was asked for
        return type("out", (), {
            "hidden_states": tuple(hidden) if output_hidden_states else None,
            "last_hidden_state": hidden[-1],
        })()


# The leading-underscore spellings OAT was written against.
_DummyTokenizer = DummyTokenizer
_DummyModel = DummyModel
