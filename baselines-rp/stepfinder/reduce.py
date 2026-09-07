"""PCA reduction of full-width embeddings to StepFinder's 128/32 inputs.

The vendored featurizer keeps the *first* 128 coordinates of a step embedding
and the first 32 of an agent embedding (``Q3Emb.py:88-90``). For
Qwen3-Embedding-0.6B that slice is principled — the model is Matryoshka-
trained, so its leading coordinates carry the most information. The decoder
backbones have no such training; their coordinate order means nothing, and the
slice is an arbitrary 2–5% sample of the hidden state. This module offers the
principled alternative: project onto the top principal directions of the
*training* features, so every backbone hands the network the most informative
128 numbers it has, not the first 128.

Two rules keep the reduction honest and faithful:

- **Fit on training features only.** The reducer is estimated once per
  (encoder, training set) from the vendored regenerated corpus and applied
  frozen to every test trajectory. Test features never influence the basis.
- **No centering.** Standard PCA subtracts the mean first, which shifts every
  vector by the same offset — and StepFinder consumes agent vectors only
  through ``cos(r_i, r_j)`` and a mean-then-gate, so centering would rewrite
  every cosine the attention bias depends on. Uncentered SVD (the top
  eigenvectors of the raw Gram matrix) is an orthonormal projection, which
  can only shrink angles' distortion, never relocate the origin. Content
  vectors get the same treatment: the network's first layer is a LayerNorm,
  which already handles location and scale.

Frequent agents appear as many duplicate rows and therefore weigh more in the
basis — deliberate: the directions that matter are the ones the model will
actually see often.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from stepfinder.encode import AGENT_DIM, CONTENT_DIM

REDUCER_FILENAME = "reducer.pt"


def _components(X: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Top-``k`` right singular vectors of ``X`` (uncentered), and the
    fraction of total squared norm they keep.

    Computed from the ``d × d`` Gram matrix, which is exact, deterministic,
    and independent of the number of rows. When the input is already ``k``
    wide or narrower, the identity (zero-padded to ``k`` columns) comes back —
    projection then equals the slice, so a reducer is always safe to apply.
    """
    d = X.shape[1]
    if d <= k:
        return np.eye(d, k, dtype=np.float32), 1.0
    gram = (X.astype(np.float64).T @ X.astype(np.float64))
    eigvals, eigvecs = np.linalg.eigh(gram)          # ascending
    order = np.argsort(eigvals)[::-1][:k]
    kept, total = float(eigvals[order].sum()), float(max(eigvals.sum(), 1e-30))
    V = eigvecs[:, order]
    # eigh's column signs are arbitrary; pin each so the largest-magnitude
    # entry is positive, making refits bit-reproducible.
    flip = np.sign(V[np.abs(V).argmax(axis=0), np.arange(V.shape[1])])
    flip[flip == 0] = 1.0
    return (V * flip).astype(np.float32), kept / total


def fit_reducer(payloads: list, content_k: int = CONTENT_DIM,
                agent_k: int = AGENT_DIM) -> dict:
    """One reducer from full-width training payloads: a projection per stream."""
    content = np.concatenate([p["content_features"] for p in payloads], axis=0)
    agent = np.concatenate([p["agent_features"] for p in payloads], axis=0)
    content_V, content_var = _components(content, content_k)
    agent_V, agent_var = _components(agent, agent_k)
    return {
        "content_V": content_V,
        "agent_V": agent_V,
        "content_var_kept": round(content_var, 6),
        "agent_var_kept": round(agent_var, 6),
        "content_dim_in": int(content.shape[1]),
        "agent_dim_in": int(agent.shape[1]),
        "n_fit_steps": int(content.shape[0]),
        "n_fit_trajectories": len(payloads),
        "centered": False,
    }


def apply_reducer(payload: dict, reducer: dict) -> dict:
    """The payload with both feature streams projected; everything else copied."""
    return dict(
        payload,
        content_features=payload["content_features"] @ reducer["content_V"],
        agent_features=payload["agent_features"] @ reducer["agent_V"],
        reduce="pca",
    )


def save_reducer(reducer: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REDUCER_FILENAME
    torch.save(reducer, path)
    return path


def load_reducer(directory: Path) -> dict | None:
    path = Path(directory) / REDUCER_FILENAME
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)
