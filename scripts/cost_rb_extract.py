#!/usr/bin/env python
"""Extraction cost of OAT and StepFinder as run, per trajectory: time, tokens, peak memory.

Both baselines are timed through their own code paths, on the Qwen3.5-9B backbone,
one trajectory at a time on one GPU, so the numbers are the cost of the reported runs:

  * OAT serializes the whole trajectory into one document and runs ONE forward pass
    over it (`oat.serialize` + `oat.extract.extract_hidden_states_for_trajectory`,
    last layer, mean-pooled per step, cap 262,144 tokens, bfloat16).
  * StepFinder embeds every step on its own, one forward pass per step for the step's
    text and one for its agent's name, memoized across the run
    (`stepfinder.features.build_features` with the repo's encoder, cap 8,192 tokens,
    float16). `passes` counts the forward passes the memo did not absorb.

Two phases per method. `inference` is the Who&When subsets the manuscript scores.
`setup` is the corpus each method must encode before it can train: the 103 MCP-Atlas
successes for OAT, and the vendored regenerated failures for StepFinder (one corpus
per Who&When subset). Rows are appended as they finish.

    .venv/bin/python scripts/cost_rb_extract.py --method oat
    .venv/bin/python scripts/cost_rb_extract.py --method stepfinder --limit 5
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
from oat.extract import extract_hidden_states_for_trajectory               # noqa: E402
from oat.serialize import (build_input_text, load_mcp_atlas_records,       # noqa: E402
                           question_text, serialize_trajectory)
from rb_shared.extractors import load_extractor                            # noqa: E402
from stepfinder import protocol as proto                                   # noqa: E402
from stepfinder.encode import AGENT_DIM, CONTENT_DIM, load_encoder         # noqa: E402
from stepfinder.features import (AGENT_FIELD_REPO, agent_text, build_features,  # noqa: E402
                                 content_text)

MODEL = "qwen3.5-9b"
MODEL_PATH = "../hub/Qwen/Qwen3.5-9B"
OAT_TRAIN_DIR = "vendored/OAT/dataset/MCP-atlas/Qwen3.5-27B"
SF_TRAIN_ROOT = "vendored/StepFinder/data"
SUBSETS = ["algorithm-generated", "hand-crafted"]
OUT_DIR = ROOT.parent / "soap" / "results-ablations" / "c4_extract_cost"
FIELDS = ["method", "phase", "corpus", "traj", "n_steps", "passes", "tokens", "seconds",
          "peak_gb", "peak_reserved_gb", "weights_gb", "truncated"]
GB = float(2 ** 30)
PHASES: set = {"inference", "setup"}
CORPORA: set = set()
SLICE = slice(0, None)


def timed(fn):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / GB, \
        torch.cuda.max_memory_reserved() / GB


class Writer:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("w", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=FIELDS, delimiter="\t")
        self.w.writeheader()

    def row(self, **kw):
        self.w.writerow(kw)
        self.fh.flush()
        print("  " + " ".join(f"{k}={v}" for k, v in kw.items()
                              if k in ("phase", "corpus", "traj", "n_steps", "passes",
                                       "tokens", "seconds", "peak_gb")), flush=True)


# ── OAT ──────────────────────────────────────────────────────────────────────
def run_oat(limit, out: Writer):
    ex = load_extractor(MODEL, {"path": MODEL_PATH, "dtype": "bf16", "device": "cuda"})
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated() / GB
    print(f"OAT extractor loaded: {weights:.2f} GB, cap {ex.max_length} tokens", flush=True)

    def one(phase, corpus, traj_id, text, bounds, n_steps):
        enc = ex.tokenizer(text, truncation=True, max_length=ex.max_length)
        n_tok = len(enc["input_ids"])
        hidden, secs, peak, reserved = timed(lambda: extract_hidden_states_for_trajectory(
            text, bounds, ex.model, ex.tokenizer, max_length=ex.max_length,
            aggregation="mean", layers=(-1,)))
        out.row(method="OAT", phase=phase, corpus=corpus, traj=traj_id, n_steps=n_steps,
                passes=1, tokens=n_tok, seconds=round(secs, 3), peak_gb=round(peak, 3),
                peak_reserved_gb=round(reserved, 3), weights_gb=round(weights, 3),
                truncated=int(n_tok >= ex.max_length))
        del hidden

    for subset in SUBSETS:
        records = load_records(str(ROOT / "data" / "ww" / subset))
        for rec in records[:limit]:
            step_text, bounds, _, _ = serialize_trajectory(rec["history"])
            text, all_bounds = build_input_text(question_text(rec, False), step_text, bounds)
            one("inference", subset, rec["id"], text, all_bounds, len(rec["history"]))
    train = [r for r in load_mcp_atlas_records(str(ROOT / OAT_TRAIN_DIR), include_success=True)
             if r["is_success"]]
    for rec in train[:limit]:
        text, all_bounds = build_input_text(rec["question"], rec["text"], rec["step_char_boundaries"])
        one("setup", "mcp-atlas", rec["id"], text, all_bounds, len(rec["step_char_boundaries"]))


# ── StepFinder ───────────────────────────────────────────────────────────────
def run_stepfinder(limit, out: Writer):
    enc = load_encoder(MODEL, {"path": MODEL_PATH, "dtype": "fp16", "device": "cuda"})
    enc.memo = True
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated() / GB
    print(f"StepFinder encoder loaded: {weights:.2f} GB, cap {enc.max_length} tokens", flush=True)

    def tokens_uncached(rec, agent_field, include_gt):
        """Tokens the forward passes will read: texts the memo has not seen yet."""
        n = 0
        seen = set()
        for i, step in enumerate(rec["history"]):
            for text, dim in ((content_text(step, i, include_gt, rec.get("ground_truth", "")), CONTENT_DIM),
                              (agent_text(step, agent_field, "standardize"), AGENT_DIM)):
                if not text or not text.strip() or (text, dim) in enc._cache or (text, dim) in seen:
                    continue
                seen.add((text, dim))
                n += min(len(enc.tokenizer(text)["input_ids"]), enc.max_length)
        return n

    def one(phase, corpus, rec, agent_field):
        n_tok = tokens_uncached(rec, agent_field, False)
        before, trunc = enc.n_forward, enc.n_truncated
        payload, secs, peak, reserved = timed(lambda: build_features(
            rec, enc, agent_field=agent_field, agent_normalize="standardize", include_gt=False))
        out.row(method="StepFinder", phase=phase, corpus=corpus, traj=str(rec["id"]),
                n_steps=len(rec["history"]), passes=enc.n_forward - before, tokens=n_tok,
                seconds=round(secs, 3), peak_gb=round(peak, 3),
                peak_reserved_gb=round(reserved, 3), weights_gb=round(weights, 3),
                truncated=enc.n_truncated - trunc)
        del payload

    if "inference" in PHASES:
        for subset in SUBSETS:
            records = load_records(str(ROOT / "data" / "ww" / subset))
            for rec in records[:limit]:
                one("inference", subset, rec, AGENT_FIELD_REPO)
    if "setup" in PHASES:
        for subset in SUBSETS:
            train_set = proto.resolve_train_set(subset)
            if CORPORA and train_set not in CORPORA:
                continue
            source = Path(ROOT / SF_TRAIN_ROOT) / proto.VENDORED_DIRS[train_set]
            records = proto.load_vendored_records(source)[SLICE]
            print(f"setup corpus {train_set}: {len(records)} trajectories from {source} "
                  f"(slice {SLICE.start}:{SLICE.stop})", flush=True)
            for rec in records[:limit]:
                one("setup", train_set, rec, proto.AGENT_FIELD_VENDORED)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["oat", "stepfinder"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--phase", action="append", choices=["inference", "setup"],
                    help="restrict to one phase (StepFinder); default both")
    ap.add_argument("--corpus", action="append",
                    help="restrict the setup phase to one training corpus (StepFinder)")
    ap.add_argument("--start", type=int, default=0, help="setup-corpus slice start (StepFinder)")
    ap.add_argument("--end", type=int, default=None, help="setup-corpus slice end (StepFinder)")
    ap.add_argument("--tag", default="", help="suffix for the output file, for sharded runs")
    args = ap.parse_args()
    global PHASES, CORPORA, SLICE
    PHASES = set(args.phase or ["inference", "setup"])
    CORPORA = set(args.corpus or [])
    SLICE = slice(args.start, args.end)
    out = Writer(Path(args.out_dir) / f"{args.method}_{MODEL}{args.tag}.tsv")
    (run_oat if args.method == "oat" else run_stepfinder)(args.limit, out)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
