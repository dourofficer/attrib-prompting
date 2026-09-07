"""Fitting the network, lifted out of the vendored script body.

``vendored/StepFinder/main.py`` has no importable train function: data loading,
training, evaluation and checkpointing all live inside ``run_experiment``
(``main.py:146-260``). This module splits that body into pieces a pipeline can
call, changing the arithmetic nowhere — the epoch loop, the metric block and
the seeding are transcriptions.

One thing had to change, and it is the reason this file has a
``model_selection`` argument. The vendored loop evaluates on ``--test_dir``
every epoch and keeps the epoch that scored best there (``main.py:231-251``).
There is no validation split anywhere in the repository, so every published
number is a maximum over fifty epochs measured on the data being reported. That
is not a result this repo can print beside its other baselines. The default
here holds out part of the *training* data instead — split by question, so no
task appears on both sides — and never looks at the corpus under test.
``model_selection="vendored"`` restores the original rule for a parity check;
``predict.py`` refuses to write predictions in that mode.

Hyperparameters come from the paper's Table 1, which the code does not carry:
``main.py``'s CLI defaults are the Algorithm-Generated column only, and the
Hand-Crafted column exists nowhere but the paper.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from stepfinder.collate import SequenceDataset, sequence_collate_fn
from stepfinder.model import StepFinder, compute_loss


# --------------------------------------------------------------------------
# Hyperparameters (paper Table 1)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HParams:
    """The knobs Table 1 tunes per subset, plus the ones it holds fixed."""

    lr: float = 1e-3
    weight_decay: float = 1e-5
    alpha: float = 0.1
    beta: float = 0.9
    gamma: float = 0.4
    lambda_temporal: float = 0.9
    scales: tuple = (1, 2)
    dropout: float = 0.5
    content_dim: int = 128
    agent_dim: int = 32
    hidden_dim: int = 64
    num_heads: int = 2
    head_dim: int = 32

    def as_dict(self) -> dict:
        d = asdict(self)
        d["scales"] = list(self.scales)
        return d

    def fingerprint(self) -> str:
        """A short stable hash, so two presets cannot share a checkpoint path."""
        import hashlib
        import json

        blob = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:8]


# Table 1. The `alg` row is also `main.py:282-295`'s CLI defaults.
PRESETS = {
    "alg": HParams(lr=1e-3, alpha=0.1, beta=0.9, gamma=0.40, lambda_temporal=0.90),
    "hc": HParams(lr=1e-5, alpha=0.3, beta=0.1, gamma=0.75, lambda_temporal=0.02),
}

EPOCHS = 50
BATCH_SIZE = 16
PATIENCE = 10
GRAD_CLIP = 1.0
SEEDS = (42, 43, 44, 45, 46)


def build_model(hp: HParams) -> StepFinder:
    return StepFinder(
        content_dim=hp.content_dim,
        agent_dim=hp.agent_dim,
        hidden_dim=hp.hidden_dim,
        num_heads=hp.num_heads,
        head_dim=hp.head_dim,
        alpha=hp.alpha,
        beta=hp.beta,
        gamma=hp.gamma,
        scales=list(hp.scales),
        dropout=hp.dropout,
    )


# --------------------------------------------------------------------------
# Seeding (verbatim: main.py:19-30)
# --------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# --------------------------------------------------------------------------
# Evaluation (verbatim: main.py:74-139)
# --------------------------------------------------------------------------

def evaluate(model, dataloader, device) -> dict:
    """Acc, Acc@2, Acc@3, MRR@3 and tolerance accuracy — the paper's metrics."""
    model.eval()

    hits = top2_hits = top3_hits = total = 0
    all_inv_ranks = []
    tolerance_hits = {delta: 0 for delta in range(1, 6)}

    with torch.no_grad():
        for batch in dataloader:
            mask = batch["mask"].to(device)
            content = batch["content_seq"].to(device)
            agent = batch["agent_seq"].to(device)
            targets = batch["mistake_labels"].to(device)

            logits, _ = model(content, agent, mask)
            logits = logits.masked_fill(mask == 0, float("-inf"))
            _, top_indices = torch.sort(logits, dim=1, descending=True)

            for i in range(len(targets)):
                total += 1
                true_idx = int(torch.argmax(targets[i]).item())
                top1_idx = int(top_indices[i, 0].item())

                if top1_idx == true_idx:
                    hits += 1
                if true_idx in top_indices[i, :2]:
                    top2_hits += 1
                if true_idx in top_indices[i, :3]:
                    top3_hits += 1

                rank_tensor = (top_indices[i] == true_idx).nonzero(as_tuple=True)[0]
                rank = int(rank_tensor[0].item()) + 1
                all_inv_ranks.append(1.0 / rank if rank <= 3 else 0.0)

                for delta in range(1, 6):
                    if abs(top1_idx - true_idx) <= delta:
                        tolerance_hits[delta] += 1

    return {
        "acc": hits / total if total > 0 else 0.0,
        "top2": top2_hits / total if total > 0 else 0.0,
        "top3": top3_hits / total if total > 0 else 0.0,
        "mrr": float(np.mean(all_inv_ranks)) if all_inv_ranks else 0.0,
        "tolerance_acc": {delta: tolerance_hits[delta] / total for delta in range(1, 6)}
        if total > 0 else {delta: 0.0 for delta in range(1, 6)},
    }


# --------------------------------------------------------------------------
# The training loop (verbatim: main.py:200-258)
# --------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, lambda_temporal: float, device) -> float:
    model.train()
    epoch_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()

        mask = batch["mask"].to(device)
        content = batch["content_seq"].to(device)
        agent = batch["agent_seq"].to(device)
        targets = batch["mistake_labels"].to(device)

        logits, temporal_loss = model(content, agent, mask)
        loss = compute_loss(logits, targets, mask, temporal_loss,
                            lambda_temporal=lambda_temporal)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / max(1, len(loader))


def holdout_by_question(features: list, frac: float, seed: int) -> tuple[list, list]:
    """Split off a validation set without letting a task straddle the boundary.

    The training corpus holds roughly eighteen regenerated trajectories per
    task. Splitting those at random would put near-duplicates of the same task
    on both sides and turn validation accuracy into a memorization check, so
    the split is over *questions* and every trajectory follows its question.
    """
    groups: dict[str, list] = {}
    for item in features:
        groups.setdefault(str(item.get("question", "")), []).append(item)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, round(frac * len(keys))) if len(keys) > 1 else 0
    val_keys = set(keys[:n_val])

    train = [x for k in keys if k not in val_keys for x in groups[k]]
    val = [x for k in keys if k in val_keys for x in groups[k]]
    return train, val


def _snapshot(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@dataclass
class FitResult:
    state_dict: dict
    best_acc: float
    best_epoch: int
    epochs_run: int
    model_selection: str
    n_train: int
    n_val: int
    val_n_tasks: int
    history: list = field(default_factory=list)


def fit(
    train_features: list,
    val_features: list,
    hp: HParams,
    *,
    seed: int = 42,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = PATIENCE,
    device: str = "cpu",
    model_selection: str = "val",
    eval_batch_size: int | None = None,
    log_every: int = 10,
) -> FitResult:
    """Train one model and return the best epoch's weights.

    ``val_features`` is whatever the caller decided to select on. That is the
    entire difference between the honest protocol and the vendored one, and
    keeping it a parameter is what lets both run through identical code.

    With no validation set, early stopping is skipped and the last epoch is
    kept. Callers decide when that happens, and they should be generous about
    it: a holdout too small to measure accuracy on is worse than no holdout at
    all. Selecting on five trajectories picks epoch 1 as often as not, which
    hands back a barely-trained network — measured at `step@1` 0.012 on
    `correct-error/math500`, against 0.042 for simply training the full budget.
    The result records which rule fired so a reader never has to guess.
    """
    set_seed(seed)
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    model = build_model(hp).to(dev)
    optimizer = optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    train_loader = DataLoader(SequenceDataset(train_features), batch_size=batch_size,
                              shuffle=True, collate_fn=sequence_collate_fn)
    # ``eval_batch_size`` exists because the position bias is batch-dependent
    # (``model.py:242`` divides by the batch's padded width): selection must
    # measure the same function deployment computes. Scoring runs at batch 1
    # (the paper's Eq. 9), so a selection rule that reports through this
    # repo's protocol passes 1 here; the vendored parity rule keeps the
    # vendored batched evaluation. Training batches are untouched either way.
    val_loader = (
        DataLoader(SequenceDataset(val_features),
                   batch_size=eval_batch_size or batch_size,
                   shuffle=False, collate_fn=sequence_collate_fn)
        if val_features else None
    )

    selection = model_selection if val_loader is not None else f"{model_selection}(fallback:last)"
    best_acc, best_epoch, stagnant = 0.0, 0, 0
    best_state = None
    history, epochs_run = [], 0

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, hp.lambda_temporal, dev)
        epochs_run = epoch + 1

        if val_loader is None:
            history.append({"epoch": epochs_run, "loss": avg_loss})
            best_state = _snapshot(model)
            best_epoch = epochs_run
            if log_every and epochs_run % log_every == 0:
                print(f"  epoch {epochs_run}/{epochs}  loss {avg_loss:.4f}", flush=True)
            continue

        res = evaluate(model, val_loader, dev)
        history.append({"epoch": epochs_run, "loss": avg_loss, "val_acc": res["acc"]})
        if log_every and epochs_run % log_every == 0:
            print(f"  epoch {epochs_run}/{epochs}  loss {avg_loss:.4f}  "
                  f"val acc {res['acc']:.4f}", flush=True)

        if res["acc"] > best_acc:
            best_acc, best_epoch, stagnant = res["acc"], epochs_run, 0
            best_state = _snapshot(model)
        else:
            stagnant += 1
        if stagnant >= patience:
            break

    if best_state is None:
        # No epoch ever beat 0.0 — the vendored rule requires strict
        # improvement over a 0.0 start (`main.py:247`), so on a hard subset it
        # saves nothing at all and the run ends with no checkpoint on disk.
        # This repo must always hand back a model, and the last epoch's
        # weights are a far better answer than the initialization.
        best_state = _snapshot(model)
        best_epoch = epochs_run
        selection = f"{model_selection}(fallback:last)"

    return FitResult(
        state_dict=best_state,
        best_acc=float(best_acc),
        best_epoch=int(best_epoch),
        epochs_run=int(epochs_run),
        model_selection=selection,
        n_train=len(train_features),
        n_val=len(val_features or []),
        val_n_tasks=len({str(x.get("question", "")) for x in (val_features or [])}),
        history=history,
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_trajectory(model, payload: dict, device) -> Optional[dict]:
    """Per-step scores for one trajectory, scored alone.

    Alone is the point. ``model.py:242`` divides the position prior by the
    batch's padded width, so a trajectory batched with a longer one gets a
    flatter prior than the paper describes. At batch size one the divisor is
    the trajectory's own length and the term is Eq. 9 exactly — and, just as
    important for this repo, a prediction stops depending on which other
    trajectories happened to be in flight beside it.
    """
    T = int(payload["num_steps"])
    if T == 0:
        return None

    batch = sequence_collate_fn([payload])
    content = batch["content_seq"].to(device)
    agent = batch["agent_seq"].to(device)
    mask = batch["mask"].to(device)

    model.eval()
    with torch.no_grad():
        logits, _ = model(content, agent, mask)
        logits = logits.masked_fill(mask == 0, float("-inf"))
        probs = torch.softmax(logits, dim=1)

    logit_row = logits[0, :T].detach().float().cpu().numpy()
    prob_row = probs[0, :T].detach().float().cpu().numpy()
    return {
        "scores": [float(x) for x in prob_row],
        "scores_logit": [float(x) for x in logit_row],
        "step_indices": list(range(T)),
        "predicted_step": int(np.argmax(prob_row)),
    }
