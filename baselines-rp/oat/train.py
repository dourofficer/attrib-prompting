"""Fit the flow of success, then score deviations from it.

Three stages, all transcribed from ``vendored/OAT/{data_pipeline,train}.py``:

1. **Project.** Hidden states are 4,096-dimensional; the model works in 64.
   PCA learns the 64 directions that carry the most variance *among successful
   trajectories*, and the same projection is then applied to failures. Latents
   are z-normalized with the successes' own mean and spread, so "typical"
   means typical-of-success.
2. **Train.** One-step-ahead prediction on successes only: from the trajectory
   so far, predict the next step's latent vector, and minimize the squared
   error. Nothing about failure enters training — that is what makes the method
   *one-class*.
3. **Score.** On a failed trajectory, each step's anomaly score is that same
   squared error. The step with the largest score is the prediction; ``top-k``
   and conformal thresholding turn the scores into a *set* of steps, the shape
   the paper reports.

Between 1 and 3 sits CORAL, which the paper uses whenever training and test
come from different sources — as they do here, always. It recentres and
rescales the test latents so their covariance matches the training set's,
using no labels at all.
"""

from __future__ import annotations

import copy
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from oat.model import OATModel

# --- vendored/OAT/config.py, verbatim values ------------------------------
LATENT_DIM = 64
ENCODER_TYPE = "pca"
LEARNING_RATE = 1e-4        # config.py:48. Paper Table 6 says 4e-5; the code wins.
BATCH_SIZE = 32
EPOCHS = 300
PATIENCE = 20
WEIGHT_DECAY = 1e-5
N_SEED_RUNS = 5
RANDOM_SEED = 42
DETECTION_TOP_K = 3
CONFORMAL_ALPHA = 0.2
CONFORMAL_MIN_DETECTIONS = 1
OOD_ALIGN_EPS = 1e-4
# --------------------------------------------------------------------------

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = tuple(RANDOM_SEED + i for i in range(N_SEED_RUNS))


# --------------------------------------------------------------------------
# Latent projection (verbatim: vendored/OAT/data_pipeline.py:317-355, 375-409)
# --------------------------------------------------------------------------

def get_time_points(length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Steps laid out on ``[0, 1]`` — the CDE integrates over time, not index."""
    t = torch.arange(length, dtype=torch.float32, device=device)
    if length > 1:
        t = t / (length - 1)
    return t


def fit_pca(trajectories: List[dict], n_components: int = LATENT_DIM) -> PCA:
    if ENCODER_TYPE != "pca":
        raise ValueError("OAT fixes encoder_type to pca.")
    X = np.concatenate([_as_numpy(t["hidden_states"]) for t in trajectories], axis=0)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def apply_pca(trajectories: List[dict], pca: PCA) -> List[dict]:
    for traj in trajectories:
        z = pca.transform(_as_numpy(traj["hidden_states"])).astype(np.float32)
        traj["latent_states"] = torch.from_numpy(z)
    return trajectories


def compute_normalization_stats(trajectories: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    all_z = torch.cat([t["latent_states"] for t in trajectories], dim=0)
    return all_z.mean(dim=0), all_z.std(dim=0).clamp(min=1e-8)


def normalize_trajectories(trajectories: List[dict], mean: torch.Tensor, std: torch.Tensor) -> List[dict]:
    for traj in trajectories:
        traj["latent_states"] = (traj["latent_states"] - mean) / std
    return trajectories


def _as_numpy(hidden_states) -> np.ndarray:
    """Hidden states are cached as float16; PCA wants float32 or better."""
    if isinstance(hidden_states, torch.Tensor):
        return hidden_states.float().numpy()
    return np.asarray(hidden_states, dtype=np.float32)


def _collect_latents(trajectories: list[dict]) -> torch.Tensor:
    return torch.cat([t["latent_states"] for t in trajectories], dim=0) if trajectories else torch.empty(0)


def _matrix_sqrt_psd(mat: torch.Tensor, eps: float) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.t()


def _matrix_invsqrt_psd(mat: torch.Tensor, eps: float) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.t()


def apply_coral_alignment(
    target_trajectories: list[dict],
    source_trajectories: list[dict],
    eps: float = OOD_ALIGN_EPS,
) -> None:
    """Whiten the test latents and recolour them with the training covariance."""
    xs = _collect_latents(source_trajectories)
    xt = _collect_latents(target_trajectories)
    if xs.numel() == 0 or xt.numel() == 0:
        return
    ms = xs.mean(dim=0)
    mt = xt.mean(dim=0)
    xs0 = xs - ms
    xt0 = xt - mt
    d = xs.shape[1]
    eye = torch.eye(d, dtype=xs.dtype)
    cs = (xs0.t() @ xs0) / max(xs0.shape[0] - 1, 1) + eps * eye
    ct = (xt0.t() @ xt0) / max(xt0.shape[0] - 1, 1) + eps * eye
    transform = _matrix_invsqrt_psd(ct, eps) @ _matrix_sqrt_psd(cs, eps)
    for traj in target_trajectories:
        traj["latent_states"] = (traj["latent_states"] - mt) @ transform + ms


# --------------------------------------------------------------------------
# Training (verbatim: vendored/OAT/train.py:18-107, 241-250)
# --------------------------------------------------------------------------

def build_model(latent_dim: int = LATENT_DIM) -> OATModel:
    return OATModel(latent_dim=latent_dim)


def _forward_one(model: torch.nn.Module, traj: dict, device: str) -> dict:
    z = traj["latent_states"].to(device)
    t = get_time_points(z.shape[0], device=z.device)
    return model(z, t)


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    trajectories: List[dict],
    device: str = DEFAULT_DEVICE,
    batch_size: int = BATCH_SIZE,
) -> float:
    model.train()
    indices = np.random.permutation(len(trajectories))
    total_loss = 0.0
    n_trajs = 0
    for start in range(0, len(indices), batch_size):
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        valid = 0
        for idx in indices[start : start + batch_size]:
            traj = trajectories[int(idx)]
            if traj["latent_states"].shape[0] < 2:
                continue
            out = _forward_one(model, traj, device)
            batch_loss = batch_loss + out["loss"]
            valid += 1
        if valid:
            loss = batch_loss / valid
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(batch_loss.item())
            n_trajs += valid
    return total_loss / max(n_trajs, 1)


def validate(model: torch.nn.Module, trajectories: List[dict], device: str = DEFAULT_DEVICE) -> float:
    model.eval()
    total_loss = 0.0
    n_trajs = 0
    with torch.no_grad():
        for traj in trajectories:
            if traj["latent_states"].shape[0] < 2:
                continue
            total_loss += float(_forward_one(model, traj, device)["loss"].item())
            n_trajs += 1
    return total_loss / max(n_trajs, 1)


def train_model(
    model: torch.nn.Module,
    train_trajs: List[dict],
    val_trajs: List[dict],
    device: str = DEFAULT_DEVICE,
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    log_every: int = 10,
) -> tuple[torch.nn.Module, dict]:
    """Train to the best validation loss and restore that checkpoint.

    ``log_every`` is the one liberty taken with this function: the vendored
    code prints all 300 epochs, which at five seeds and two extractors buries
    everything else. The arithmetic is untouched.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history = {"train_loss": [], "val_loss": [], "best_val_loss": best_val, "best_epoch": best_epoch}

    for epoch in range(int(epochs)):
        train_loss = train_epoch(model, optimizer, train_trajs, device=device)
        val_loss = validate(model, val_trajs, device=device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if log_every and (epoch % log_every == 0 or epoch + 1 == int(epochs)):
            print(f"epoch {epoch + 1:03d}/{epochs} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["best_epoch"] = best_epoch
    return model, history


def split_train_val(success: List[dict], seed: int = RANDOM_SEED) -> tuple[list[dict], list[dict]]:
    """80/20 over successes. The 20% is both early-stopping and calibration set."""
    if len(success) < 2:
        raise RuntimeError("Need at least two success trajectories for train/validation split.")
    train_idx, val_idx = train_test_split(
        np.arange(len(success)),
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
    return [success[int(i)] for i in train_idx], [success[int(i)] for i in val_idx]


# --------------------------------------------------------------------------
# Scoring and detection (verbatim: vendored/OAT/train.py:110-238)
# --------------------------------------------------------------------------

# Shared with StepFinder so ``topk_steps`` means the same thing in both
# methods' output files (``rb_shared/detect.py``).
from rb_shared.detect import topk_detection as _topk_detection  # noqa: E402


def _scorable_scores_and_indices(scores: torch.Tensor, traj: dict) -> Tuple[np.ndarray, List[int]]:
    """Drop the question row: it seeds the dynamics and never earns a score."""
    scores_np = np.asarray(scores.detach().cpu().numpy(), dtype=float)
    step_indices = [int(x) for x in traj.get("step_indices", list(range(len(scores_np))))]
    if len(step_indices) != len(scores_np):
        step_indices = list(range(len(scores_np)))
    kept = [i for i, step_idx in enumerate(step_indices) if step_idx >= 0]
    return scores_np[kept], [step_indices[i] for i in kept]


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Centre on the median, scale by the median absolute deviation.

    Robust to the one huge score a failure often has, which is exactly the
    thing a mean-and-standard-deviation scaling would let dominate.
    """
    if scores.size == 0:
        return scores
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    scale = max(1.4826 * mad, 1e-8)
    return (scores - median) / scale


def conformal_quantile_threshold(calibration_scores: np.ndarray, alpha: float = CONFORMAL_ALPHA) -> float:
    """The split-conformal quantile: flag at most ``alpha`` of normal steps."""
    if calibration_scores.size == 0:
        return float("inf")
    alpha = float(np.clip(alpha, 1e-8, 1.0 - 1e-8))
    n = calibration_scores.size
    q = float(np.ceil((n + 1) * (1.0 - alpha)) / n)
    q = float(np.clip(q, 0.0, 1.0))
    try:
        return float(np.quantile(calibration_scores, q, method="higher"))
    except TypeError:
        return float(np.quantile(calibration_scores, q, interpolation="higher"))


def collect_calibration_scores(
    model: torch.nn.Module,
    calibration_trajectories: List[dict],
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    model.eval()
    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for traj in calibration_trajectories:
            z = traj["latent_states"].to(device)
            if z.shape[0] < 2:
                continue
            t = get_time_points(z.shape[0], device=z.device)
            scores = model.anomaly_scores(z, t)
            scores_np, _ = _scorable_scores_and_indices(scores, traj)
            if scores_np.size > 0:
                all_scores.append(_normalize_scores(scores_np))
    if not all_scores:
        return np.empty(0, dtype=float)
    return np.concatenate(all_scores, axis=0)


def _conformal_detection(
    scores: np.ndarray,
    step_indices: List[int],
    threshold: float,
    min_detections: int,
) -> List[int]:
    normalized = _normalize_scores(scores)
    detected = [
        int(step_idx)
        for step_idx, score in zip(step_indices, normalized)
        if float(score) > float(threshold)
    ]
    if not detected and int(min_detections) > 0:
        return _topk_detection(normalized, step_indices, int(min_detections))
    return sorted(detected)


def score_trajectory(
    model: torch.nn.Module,
    traj: dict,
    device: str = DEFAULT_DEVICE,
    top_k: int = DETECTION_TOP_K,
    conformal_threshold: float = float("inf"),
    conformal_min_detections: int = CONFORMAL_MIN_DETECTIONS,
) -> Optional[dict]:
    """Score one trajectory. ``None`` when it has nothing scorable.

    The body is the per-trajectory half of the vendored
    ``score_failure_trajectories`` loop, lifted out so ``predict.py`` can write
    one output file at a time and resume where it left off.
    """
    model.eval()
    with torch.no_grad():
        z = traj["latent_states"].to(device)
        if z.shape[0] < 2:
            return None
        t = get_time_points(z.shape[0], device=z.device)
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))
        start = time.perf_counter()
        scores = model.anomaly_scores(z, t)
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))
        elapsed = time.perf_counter() - start
        scores_np, step_indices = _scorable_scores_and_indices(scores, traj)
        if len(step_indices) == 0:
            return None
        pred_local = int(np.argmax(scores_np))
        return {
            "predicted_step": int(step_indices[pred_local]),
            "predicted_local_step": pred_local,
            "score_step_indices": step_indices,
            "scores": scores_np.astype(np.float32),
            "topk_detection": _topk_detection(scores_np, step_indices, top_k),
            "conformal_detection": _conformal_detection(
                scores_np, step_indices, conformal_threshold, conformal_min_detections
            ),
            "num_scored_steps": len(step_indices),
            "inference_time_ms": elapsed * 1000.0,
        }


def score_failure_trajectories(
    model: torch.nn.Module,
    failures: List[dict],
    device: str = DEFAULT_DEVICE,
    top_k: int = DETECTION_TOP_K,
    conformal_threshold: float = float("inf"),
    conformal_min_detections: int = CONFORMAL_MIN_DETECTIONS,
) -> List[dict]:
    """The vendored batch entry point, kept so the parity test can drive both."""
    predictions = []
    for traj in failures:
        scored = score_trajectory(
            model, traj, device, top_k, conformal_threshold, conformal_min_detections
        )
        if scored is None:
            continue
        predictions.append(
            {
                "trajectory_id": traj["id"],
                **scored,
                "gt_error_steps": traj["error_steps"],
                "num_steps": traj.get("num_steps_original", traj["num_steps"]),
            }
        )
    return predictions


def filter_model_steps(
    hidden_states: torch.Tensor,
    step_indices: Sequence[int],
    step_is_model: Sequence[bool],
) -> tuple[torch.Tensor, list[int]]:
    """Keep the question row and the agent's own turns.

    Verbatim rule from ``vendored/OAT/data_pipeline.py:294-299``: a row stays
    if it is the question (index below zero) or an agent took it. The vendored
    loader then *discards* any failure whose annotated step did not survive;
    this repo keeps it and writes a prediction anyway, because the shared
    report expects one file per corpus entry and scores a missing file as
    wrong either way.
    """
    T = min(int(hidden_states.shape[0]), len(step_is_model), len(step_indices))
    kept = [i for i in range(T) if int(step_indices[i]) < 0 or bool(step_is_model[i])]
    if not kept:
        return hidden_states[:0], []
    return hidden_states[kept], [int(step_indices[i]) for i in kept]
