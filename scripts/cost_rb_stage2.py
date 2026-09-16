#!/usr/bin/env python
"""Stage-2 inference cost of OAT and StepFinder, in process: what each pays per trajectory
AFTER its representations are stored.

  * OAT: project the cached last-layer states with the training PCA, align them with
    CORAL (once per subset — reported apart), then run the trained neural CDE and the
    conformal decoder on one trajectory (`oat.train.score_trajectory`).
  * StepFinder: run the trained BiLSTM + attention scorer on one trajectory's cached
    step features (`stepfinder.train.score_trajectory`).

Both read the caches and checkpoints of the reported runs (outputs-rb-nogt/, training
seed 42) and time every Who&When trajectory, several repeats, on CPU and on one GPU.

    .venv/bin/python scripts/cost_rb_stage2.py
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines-rp"))

from baselines.prompting.predict import load_records                       # noqa: E402
from oat import train as core                                              # noqa: E402
from oat.predict import (apply_projector, load_cached_trajectories,        # noqa: E402
                         load_projector, load_seed_model)
from rb_shared.cache import StateCache                                     # noqa: E402
from stepfinder import protocol as proto                                   # noqa: E402
from stepfinder.train import build_model                                   # noqa: E402
from stepfinder.train import score_trajectory as sf_score                  # noqa: E402

try:
    from stepfinder.protocol import PRESETS
except ImportError:                                                        # pragma: no cover
    from stepfinder.predict import PRESETS

MODEL = "qwen3.5-9b"
RB = ROOT / "outputs-rb-nogt"
SUBSETS = ["algorithm-generated", "hand-crafted"]
OUT_DIR = ROOT.parent / "soap" / "results-ablations" / "c4_extract_cost"
FIELDS = ["method", "subset", "device", "n_traj", "prepare_s_subset", "score_ms_per_traj", "repeats"]


def clock(device):
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def run_oat(subset, device, repeats):
    ckpt_dir = RB / "mcp-atlas" / "train" / MODEL / "_oat-ckpt"
    bundle = load_projector(ckpt_dir)
    records = load_records(str(ROOT / "data" / "ww" / subset))
    cache = StateCache(RB / "ww" / subset / MODEL / "_oat-states")
    trajs = load_cached_trajectories(cache, [r["id"] for r in records], -1)
    t0 = clock(device)
    apply_projector(trajs, bundle)
    core.apply_coral_alignment(trajs, [{"latent_states": bundle["source_latents"]}])
    prepare_s = clock(device) - t0
    model, ckpt = load_seed_model(ckpt_dir, 42, device)

    def score(traj):
        return core.score_trajectory(model, traj, device=device, top_k=3,
                                     conformal_threshold=ckpt["conformal_threshold"],
                                     conformal_min_detections=core.CONFORMAL_MIN_DETECTIONS)

    score(trajs[0])                                   # warm-up
    t0 = clock(device)
    for _ in range(repeats):
        for traj in trajs:
            score(traj)
    ms = (clock(device) - t0) / repeats / len(trajs) * 1e3
    return {"method": "OAT", "subset": subset, "device": device, "n_traj": len(trajs),
            "prepare_s_subset": round(prepare_s, 4), "score_ms_per_traj": round(ms, 3),
            "repeats": repeats}


def run_stepfinder(subset, device, repeats):
    train_set = proto.resolve_train_set(subset)
    preset = proto.resolve_preset(train_set)
    ckpt = RB / "stepfinder-regen" / train_set / MODEL / "_sf-ckpt" / preset / "val" / "s42" / "model.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(PRESETS[preset])
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    records = load_records(str(ROOT / "data" / "ww" / subset))
    feats = StateCache(RB / "ww" / subset / MODEL / "_sf-feats")
    payloads = [p for p in (feats.load(str(r["id"])) for r in records)
                if p is not None and int(p.get("num_steps", 0)) > 0]

    sf_score(model, payloads[0], device)              # warm-up
    t0 = clock(device)
    for _ in range(repeats):
        for p in payloads:
            sf_score(model, p, device)
    ms = (clock(device) - t0) / repeats / len(payloads) * 1e3
    return {"method": "StepFinder", "subset": subset, "device": device, "n_traj": len(payloads),
            "prepare_s_subset": 0.0, "score_ms_per_traj": round(ms, 3), "repeats": repeats}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    rows = []
    for subset in SUBSETS:
        for device in ("cpu", "cuda"):
            for fn in (run_oat, run_stepfinder):
                row = fn(subset, device, args.repeats)
                print(row, flush=True)
                rows.append(row)
    out = Path(args.out_dir) / f"rb_stage2_{MODEL}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
