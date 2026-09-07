"""Padding a list of trajectories into one batch.

Transcribed verbatim from ``vendored/StepFinder/collate_fn.py``, plus the
one-line dataset wrapper from ``main.py:59-67``.

Two properties of the vendored code are load-bearing downstream and worth
naming. Padding is on the **right**, and ``mask`` is a float32 tensor rather
than a bool — every consumer in ``model.py`` was written against that (it does
``mask == 0``, ``~mask.bool()``, ``mask.sum(dim=1)``), so the dtype is not an
oversight to tidy up. And the collate reads only three keys, which is why our
cached feature payloads can carry as much provenance metadata as they like and
still be fed straight to the vendored trainer in a parity test.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """``main.py:59-67`` — a list of feature dicts, indexable."""

    def __init__(self, data: list) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def sequence_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of feature dicts into a padded batch.

    Each sample in the batch is expected to contain:
        content_features: np.ndarray [T, content_dim]
        agent_features:   np.ndarray [T, agent_dim]
        mistake_labels:   np.ndarray [T], int64

    Returns:
        Dict with keys:
            content_seq:    [B, T_max, content_dim]
            agent_seq:      [B, T_max, agent_dim]
            mistake_labels: [B, T_max]
            mask:           [B, T_max]  1 = valid step, 0 = padding
    """
    batch_size = len(batch)
    T_max = max(sample["content_features"].shape[0] for sample in batch)
    D_content = batch[0]["content_features"].shape[1]
    D_agent = batch[0]["agent_features"].shape[1]

    content_seq = torch.zeros((batch_size, T_max, D_content), dtype=torch.float32)
    agent_seq = torch.zeros((batch_size, T_max, D_agent), dtype=torch.float32)
    mistake_labels = torch.zeros((batch_size, T_max), dtype=torch.long)
    mask = torch.zeros((batch_size, T_max), dtype=torch.float32)

    for i, sample in enumerate(batch):
        L = sample["content_features"].shape[0]
        content_seq[i, :L] = torch.tensor(sample["content_features"], dtype=torch.float32)
        agent_seq[i, :L] = torch.tensor(sample["agent_features"], dtype=torch.float32)
        mistake_labels[i, :L] = torch.tensor(sample["mistake_labels"], dtype=torch.long)
        mask[i, :L] = 1.0

    return {
        "content_seq": content_seq,
        "agent_seq": agent_seq,
        "mistake_labels": mistake_labels,
        "mask": mask,
    }