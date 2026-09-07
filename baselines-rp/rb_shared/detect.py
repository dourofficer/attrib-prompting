"""Turning per-step scores into a predicted set of steps.

Both papers report metrics over a *set* of suspect steps rather than a single
one, and both build that set the same way: take the k highest-scoring steps,
then sort them by position. The sort is what makes the set comparable across
methods — a top-3 prediction means the same thing in an OAT file and a
StepFinder file — so the rule lives in one place.

Verbatim from ``vendored/OAT/train.py:112-117``.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def topk_detection(scores: np.ndarray, step_indices: Sequence[int], k: int) -> List[int]:
    if scores.size == 0:
        return []
    k_eff = max(1, min(int(k), int(scores.size)))
    return sorted(int(step_indices[i]) for i in np.argsort(-scores)[:k_eff])
