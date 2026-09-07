"""Turn a step's text into a vector, once per distinct string.

This is StepFinder's only use of a language model, and it generates nothing:
one forward pass per string, the last token's hidden state kept, sliced to the
width the network wants. The pooling and the slicing are transcribed from
``vendored/StepFinder/Q3Emb.py:43-92`` and
``feature_construction.py:41-64``; three things around them are adapted.

**Loading goes through the shared loader.** The vendored ``Qwen3Embedding``
calls ``AutoModel``, which cannot load ``../hub/Qwen/Qwen3.5-9B`` — a
vision-language checkpoint whose text decoder sits a level down and whose
``hidden_size`` lives under ``text_config``. ``rb_shared.extractors`` already
unwraps that for OAT, and the decoder it returns emits the same
``last_hidden_state`` the vendored pooling consumes.

**The context cap stays at the vendored 8192**, not the checkpoint's own
maximum, so a step embeds identically whichever encoder reads it. Steps longer
than that truncate from the right — no ``truncation_side`` is set upstream
either — which means last-token pooling reads the 8192nd token rather than the
step's real end. Rare enough to count rather than fix: ``features.py`` records
how often it happens.

**Distinct strings are encoded once.** Agent identity is a short name drawn
from a tiny vocabulary — thirty distinct strings across every corpus here —
so encoding each step's agent separately repeats the same forward pass
thousands of times. A memo keyed on ``(text, dim)`` removes the repetition and
changes no number, because the vendored call is deterministic under
``no_grad``. Batching would not be free in the same way: padding several
strings together changes what attention sees, so it stays one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# vendored/StepFinder/Q3Emb.py:18 — the tokenizer truncation cap.
MAX_LENGTH = 8192
CONTENT_DIM = 128
AGENT_DIM = 32


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Verbatim ``Qwen3Embedding._last_token_pool`` (``Q3Emb.py:43-58``)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


@dataclass
class Encoder:
    """A frozen model, its tokenizer, and a memo of what it has already said."""

    name: str
    model: object
    tokenizer: object
    max_length: int = MAX_LENGTH
    memo: bool = True
    _cache: dict = field(default_factory=dict, repr=False)
    n_truncated: int = 0
    n_forward: int = 0

    def embed(self, text: str, dim: int | None) -> np.ndarray:
        """One step's text as a ``dim``-wide float32 vector.

        Verbatim ``TemporalFeatureExtractor._encode``
        (``feature_construction.py:41-64``): empty or whitespace-only text is a
        zero vector, a short embedding is right-padded, a long one truncated,
        and any failure degrades to zeros with a warning rather than losing the
        trajectory.

        ``dim=None`` keeps the encoder's full hidden width — the input the PCA
        reduction is fitted on, where a prefix slice would throw away exactly
        the information the reduction is meant to keep.
        """
        if not text or not text.strip():
            return np.zeros(dim if dim is not None else self.width(), dtype=np.float32)

        key = (text, dim)
        if self.memo and key in self._cache:
            return self._cache[key]

        try:
            embedding = self._forward(text, dim)
        except Exception as e:  # noqa: BLE001 — the vendored contract
            print(f"[WARNING] Encoding failed ({e}), returning zero vector.")
            return np.zeros(dim, dtype=np.float32)

        if dim is not None and embedding.shape[0] > dim:
            embedding = embedding[:dim]
        elif dim is not None and embedding.shape[0] < dim:
            embedding = np.pad(embedding, (0, dim - embedding.shape[0]), mode="constant")

        if self.memo:
            self._cache[key] = embedding
        return embedding

    def width(self) -> int:
        """The encoder's native hidden width, probed once with one forward pass."""
        if not hasattr(self, "_width"):
            self._width = int(self._forward(".", None).shape[0])
        return self._width

    def _forward(self, text: str, dim: int | None) -> np.ndarray:
        """Verbatim ``Qwen3Embedding.encode`` (``Q3Emb.py:60-92``) for one string."""
        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if int(inputs["input_ids"].shape[1]) >= self.max_length:
            self.n_truncated += 1
        device = _model_device(self.model)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs)
            pooled = last_token_pool(out.last_hidden_state, inputs["attention_mask"])
            if dim is not None and dim > 0:
                pooled = pooled[:, :dim]
        self.n_forward += 1
        return pooled.squeeze(0).float().cpu().numpy().astype(np.float32)


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_encoder(name: str, spec: dict) -> Encoder:
    """Load a local checkpoint as a StepFinder encoder.

    ``spec`` takes the same keys as ``rb_shared.extractors.load_extractor``
    (``path``, ``tokenizer``, ``dtype``, ``device``), with ``dtype`` defaulting
    to ``fp16`` because the vendored wrapper pins it (``Q3Emb.py:24, 30``).
    """
    from rb_shared.extractors import load_extractor

    spec = {"dtype": "fp16", **spec, "max_length": MAX_LENGTH}
    ex = load_extractor(name, spec)
    # Qwen3-Embedding pads on the left so the last real token is the last
    # column; the pooling handles either, but matching upstream costs nothing.
    if hasattr(ex.tokenizer, "padding_side"):
        ex.tokenizer.padding_side = "left"
    return Encoder(name=name, model=ex.model, tokenizer=ex.tokenizer, max_length=ex.max_length)


def dummy_encoder(name: str = "dummy", hidden_dim: int = 160) -> Encoder:
    """A checkpoint-free stand-in. 160 > 128 so the width slice is a real slice."""
    from rb_shared.dummy import DummyModel, DummyTokenizer

    return Encoder(name=name, model=DummyModel(hidden_dim, 2), tokenizer=DummyTokenizer())
