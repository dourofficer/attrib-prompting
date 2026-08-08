"""Completion check + per-seed comparison tables for the prompting baselines.

Two jobs:

1. **Completion check** — for every ``(model, subset, method)``, report whether
   ``predictions_method-{method}.jsonl`` finished (row count vs #trajectories) and
   how many rows are unparsed (``predicted_step is None``). ``--check-only`` runs
   just this.

2. **Per-seed comparison tables** — for each ``(model, subset)`` emit one wide
   table with **one row per seed**, placing the three prompting methods next to
   SVD and CRR on the *identical* per-seed val/test splits, so the comparison is
   fair. Values are fractions in [0,1] (matching the CRR reduced tables).

Split reproduction is byte-identical to ``src/svd/reproduce.py`` /
``experiments/svd/run_all_positions.py``:

    files = sorted(reps_dir.glob("*.safetensors"), key=lambda x: int(x.stem))  # basenames
    trval, test = split_data(files, train+val, seed)          # test = 2nd slice
    _train, val = split_data(trval, train/(train+val), seed)  # val  = 2nd slice

The reps id-set equals the full data (extraction covered every trajectory), so the
split is model-independent — we use ``split_model`` (qwen3.5-9b, present for every
dataset) as the canonical id-source for both models' baselines. SVD/CRR numbers are
pulled from ``{crr_reduced_root}/{model}/{subset}/svd.tsv`` (best pooling per seed by
``disc_step_acc_test``); absent → those columns are blank (baseline-only fallback).

Usage
-----
python -m baselines.prompting.report --config baselines/prompting/configs/report_ww.yaml [--check-only]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.utils.utils import split_data, standardize_role

METHODS_DEFAULT = ["all_at_once", "step_by_step", "binary_search"]


def load_cfg(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts, node = key.split("."), cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Matching (mirrors vendored evaluate.py, with standardize_role normalization)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_agent(x) -> str | None:
    if x is None:
        return None
    return standardize_role(str(x)).strip().lower()


def _agent_hit(pred, gold) -> bool:
    p, g = _norm_agent(pred), _norm_agent(gold)
    if p is None or g is None or g == "":
        return False
    return p == g or g in p


def _step_hit(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return int(pred) == int(gold)
    except (TypeError, ValueError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# IO + split reproduction
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions(pred_file: Path) -> dict[str, dict]:
    """id -> prediction row."""
    preds: dict[str, dict] = {}
    with pred_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            preds[str(row["id"])] = row
    return preds


def reps_file_list(reps_dir: Path) -> list[str]:
    """Same construction SVD's reproduce.py uses: names sorted by int stem."""
    files = sorted(reps_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    return [f.name for f in files]


def val_test_ids(files: list[str], train: float, val: float, seed: int) -> tuple[list[str], list[str]]:
    """Reproduce (val_ids, test_ids) exactly as src/svd/reproduce.py does."""
    trval, test = split_data(files, train + val, seed)
    _train, va = split_data(trval, train / (train + val), seed)
    return [Path(f).stem for f in va], [Path(f).stem for f in test]


def _acc(ids: list[str], preds: dict[str, dict]) -> tuple[int, float, float]:
    """Return (n, agent_frac, step_frac) over `ids` (missing pred = wrong)."""
    agent_c = step_c = 0
    for i in ids:
        row = preds.get(str(i))
        if row is None:
            continue
        agent_c += _agent_hit(row.get("predicted_agent"), row.get("gold_agent"))
        step_c += _step_hit(row.get("predicted_step"), row.get("gold_step"))
    n = len(ids)
    return n, (agent_c / n if n else 0.0), (step_c / n if n else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SVD + CRR merge (from the discounted reduced tables)
# ─────────────────────────────────────────────────────────────────────────────

_CRR_COLS = {
    "svd_step_val": "undisc_step_acc_val",   "svd_step_test": "undisc_step_acc_test",
    "svd_agent_val": "undisc_agent_acc_val", "svd_agent_test": "undisc_agent_acc_test",
    "crr_step_val": "disc_step_acc_val",     "crr_step_test": "disc_step_acc_test",
    "crr_agent_val": "disc_agent_acc_val",   "crr_agent_test": "disc_agent_acc_test",
}


def load_crr_svd(reduced_file: Path) -> dict[int, dict]:
    """seed -> {svd_*, crr_*} using the best-pooling row per seed (by disc_step_acc_test)."""
    if not reduced_file.exists():
        return {}
    df = pd.read_csv(reduced_file, sep="\t")
    out: dict[int, dict] = {}
    for seed, g in df.groupby("seed"):
        g = g.sort_values(["disc_step_acc_test", "disc_agent_acc_test"], ascending=False)
        r = g.iloc[0]
        out[int(seed)] = {k: float(r[src]) for k, src in _CRR_COLS.items()}
    return out


def canonical_seeds(cfg: dict, subset: str) -> list[int]:
    """Seed set to report on. Config `seeds` wins; else the split_model's CRR table; else 1..10."""
    if cfg.get("seeds"):
        return list(dict.fromkeys(cfg["seeds"]))  # dedupe, keep order
    sm = cfg.get("split_model", "qwen3.5-9b")
    if cfg.get("crr_reduced_root"):
        red = Path(cfg["crr_reduced_root"]) / sm / subset / "svd.tsv"
        if red.exists():
            return sorted(int(s) for s in pd.read_csv(red, sep="\t")["seed"].unique())
    return list(range(1, 11))


# ─────────────────────────────────────────────────────────────────────────────
# Completion check
# ─────────────────────────────────────────────────────────────────────────────

def completion_status(cfg: dict) -> pd.DataFrame:
    split_model = cfg.get("split_model", "qwen3.5-9b")
    methods = cfg.get("methods", METHODS_DEFAULT)
    rows = []
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            reps_dir = Path(cfg["reps_root"]) / split_model / subset
            expected = len(reps_file_list(reps_dir)) if reps_dir.exists() else None
            for method in methods:
                pf = Path(cfg["pred_root"]) / model / subset / f"predictions_method-{method}.jsonl"
                if not pf.exists():
                    status, nrows, no_pred, fmt_fail = "MISSING", 0, 0, 0
                else:
                    preds = load_predictions(pf)
                    nrows = len(preds)
                    none_rows = [r for r in preds.values() if r.get("predicted_step") is None]
                    # no_pred: model emitted no prediction. Split into the benign
                    # "no error flagged" case (raw is None — step_by_step reached the
                    # end without a "1. yes") vs. a real format failure (raw present
                    # but the Agent/Step regex found nothing, e.g. deepseek narrating
                    # instead of following the template).
                    no_pred = sum(1 for r in none_rows if r.get("raw") is None)
                    fmt_fail = sum(1 for r in none_rows if r.get("raw") is not None)
                    if expected is None:
                        status = f"DONE?({nrows})"
                    elif nrows >= expected:
                        status = "DONE"
                    else:
                        status = f"PARTIAL({nrows}/{expected})"
                rows.append({"model": model, "subset": subset, "method": method,
                             "status": status, "rows": nrows, "expected": expected,
                             "no_pred": no_pred, "fmt_fail": fmt_fail})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-(model,subset) comparison table — one row per seed
# ─────────────────────────────────────────────────────────────────────────────

def build_table(model: str, subset: str, cfg: dict) -> pd.DataFrame | None:
    split_model = cfg.get("split_model", "qwen3.5-9b")
    reps_dir = Path(cfg["reps_root"]) / split_model / subset
    if not reps_dir.exists():
        print(f"  [{model}/{subset}] no reps for split ({reps_dir}) — skip")
        return None
    files = reps_file_list(reps_dir)
    splits = cfg["splits"]
    methods = cfg.get("methods", METHODS_DEFAULT)
    gt = bool(cfg.get("gt_in_prompt", False))

    preds_by_method = {}
    for m in methods:
        pf = Path(cfg["pred_root"]) / model / subset / f"predictions_method-{m}.jsonl"
        if pf.exists():
            preds_by_method[m] = load_predictions(pf)
    if not preds_by_method:
        print(f"  [{model}/{subset}] no predictions for any method — skip")
        return None

    crr = {}
    if cfg.get("crr_reduced_root"):
        crr = load_crr_svd(Path(cfg["crr_reduced_root"]) / model / subset / "svd.tsv")

    rows = []
    for seed in canonical_seeds(cfg, subset):
        val_ids, test_ids = val_test_ids(files, splits["train"], splits["val"], seed)
        row = {"seed": seed, "n_val": len(val_ids), "n_test": len(test_ids), "gt_in_prompt": gt}
        best_step = best_agent = None
        for m in methods:
            if m not in preds_by_method:
                for suf in ("step@1_val", "step@1_test", "agent@1_val", "agent@1_test"):
                    row[f"{m}_{suf}"] = None
                continue
            p = preds_by_method[m]
            _nv, av, sv = _acc(val_ids, p)
            _nt, at, st = _acc(test_ids, p)
            row[f"{m}_step@1_val"], row[f"{m}_step@1_test"] = sv, st
            row[f"{m}_agent@1_val"], row[f"{m}_agent@1_test"] = av, at
            best_step = st if best_step is None else max(best_step, st)
            best_agent = at if best_agent is None else max(best_agent, at)
        row["baseline_best_step@1_test"] = best_step
        row["baseline_best_agent@1_test"] = best_agent
        c = crr.get(int(seed), {})
        for k in _CRR_COLS:
            row[k] = c.get(k)
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(prog="baselines.prompting.report")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--check-only", action="store_true", help="Only print/write the completion status.")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    status = completion_status(cfg)
    status.to_csv(out_root / "completion_status.tsv", sep="\t", index=False)
    print("=== completion status ===")
    print(status.to_string(index=False))
    print(f"(written: {out_root/'completion_status.tsv'})")
    if args.check_only:
        return

    summaries = []
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            df = build_table(model, subset, cfg)
            if df is None:
                continue
            od = out_root / model / subset
            od.mkdir(parents=True, exist_ok=True)
            df.to_csv(od / "comparison_by_seed.tsv", sep="\t", index=False)
            s = df.drop(columns=["seed"]).mean(numeric_only=True).to_dict()
            s = {"model": model, "subset": subset, **s}
            summaries.append(s)
            print(f"wrote {od/'comparison_by_seed.tsv'}  ({len(df)} seeds)")

    if summaries:
        sm = pd.DataFrame(summaries)
        sm.to_csv(out_root / "summary_mean_over_seeds.tsv", sep="\t", index=False)
        print(f"\nwrote {out_root/'summary_mean_over_seeds.tsv'}")


if __name__ == "__main__":
    main()
