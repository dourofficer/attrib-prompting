"""Completion check + per-seed evaluation tables for the prompting baselines.

Inference (``predict.py``) covers every trajectory and is split-agnostic; this
is the evaluation half. Two jobs:

1. **Completion check** — for every ``(model, subset, method)``, report whether
   the per-trajectory outputs are complete (file count vs corpus size) and how
   many are unparsed (``predicted_step is None``). ``--check-only`` runs just
   this.

2. **Per-seed tables** — for each ``(model, subset)`` emit one table with one
   row per seed: step@1 / agent@1 on that seed's val/test splits, plus
   split-independent ``*_full`` columns over the whole corpus (constant across
   seed rows; they survive the mean into the summary unchanged).

Split reproduction mirrors the attribscope project exactly:

    files = corpus filenames sorted by int(stem)           # the universe
    trval, test = split_data(files, train+val, seed)       # test = 2nd slice
    _train, val = split_data(trval, train/(train+val), seed)  # val = 2nd slice

The universe comes from ``data/<ds>/<subset>/*.json`` — verified identical to
the id-set the attribscope splits are built over — so no artifacts outside
this repo are needed. Matching rules also mirror attribscope's main_table.py:
agent@1 normalizes with ``standardize_role``/strip/lower and accepts the gold
name as a substring of the prediction; step@1 is integer equality (gold steps
are strings in ww); a missing prediction counts as wrong.

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

from baselines.shared.common import (
    _get_sorted_json_files,
    nogt_root,
    split_data,
    standardize_role,
)

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
# Matching (mirrors attribscope src/reports/main_table.py — keep in lockstep)
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
# IO + split reproduction
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions(method_dir: Path) -> dict[str, dict]:
    """id -> prediction doc from a per-trajectory output directory."""
    preds: dict[str, dict] = {}
    for path in method_dir.glob("*.json"):
        if not path.stem.isdigit():  # skip _run.json and other metadata
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        preds[str(doc["id"])] = doc
    return preds


def universe_files(data_dir: Path) -> list[str]:
    """The split universe: corpus filenames, numerically sorted (verified
    identical to the id-set the attribscope splits are computed over)."""
    return _get_sorted_json_files(data_dir)


def val_test_ids(files: list[str], train: float, val: float, seed: int) -> tuple[list[str], list[str]]:
    """Reproduce (val_ids, test_ids) exactly as attribscope does."""
    trval, test = split_data(files, train + val, seed)
    _train, va = split_data(trval, train / (train + val), seed)
    return [Path(f).stem for f in va], [Path(f).stem for f in test]


def canonical_seeds(cfg: dict) -> list[int]:
    seeds = cfg.get("seeds")
    if not seeds:
        raise SystemExit("report config must set `seeds` explicitly "
                         "(ww/traceelephant/tracertraj: 1..20, correct-error: 1..3)")
    return list(dict.fromkeys(seeds))  # dedupe, keep order


def method_dir(cfg: dict, model: str, subset: str, method: str) -> Path:
    return Path(cfg["pred_root"]) / subset / model / method


# ─────────────────────────────────────────────────────────────────────────────
# Completion check
# ─────────────────────────────────────────────────────────────────────────────

def completion_status(cfg: dict) -> pd.DataFrame:
    methods = cfg.get("methods", METHODS_DEFAULT)
    rows = []
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            expected = len(universe_files(Path(cfg["data_dir"]) / subset))
            for method in methods:
                md = method_dir(cfg, model, subset, method)
                if not md.is_dir():
                    status, nrows, no_pred, fmt_fail = "MISSING", 0, 0, 0
                else:
                    preds = load_predictions(md)
                    nrows = len(preds)
                    none_rows = [r for r in preds.values() if r.get("predicted_step") is None]
                    # no_pred: model emitted no prediction. Split into the benign
                    # "no error flagged" case (raw is None — step_by_step reached the
                    # end without a "1. yes") vs. a real format failure (raw present
                    # but the Agent/Step regex found nothing, e.g. deepseek narrating
                    # instead of following the template).
                    no_pred = sum(1 for r in none_rows if r.get("raw") is None)
                    fmt_fail = sum(1 for r in none_rows if r.get("raw") is not None)
                    status = "DONE" if nrows >= expected else f"PARTIAL({nrows}/{expected})"
                rows.append({"model": model, "subset": subset, "method": method,
                             "status": status, "rows": nrows, "expected": expected,
                             "no_pred": no_pred, "fmt_fail": fmt_fail})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-(model,subset) table — one row per seed
# ─────────────────────────────────────────────────────────────────────────────

def build_table(model: str, subset: str, cfg: dict) -> pd.DataFrame | None:
    files = universe_files(Path(cfg["data_dir"]) / subset)
    if not files:
        print(f"  [{model}/{subset}] no corpus files under {cfg['data_dir']}/{subset} — skip")
        return None
    splits = cfg["splits"]
    methods = cfg.get("methods", METHODS_DEFAULT)
    gt = bool(cfg.get("gt_in_prompt", False))

    preds_by_method = {}
    for m in methods:
        md = method_dir(cfg, model, subset, m)
        if md.is_dir():
            preds = load_predictions(md)
            if preds:
                preds_by_method[m] = preds
    if not preds_by_method:
        print(f"  [{model}/{subset}] no predictions for any method — skip")
        return None

    # Split-independent accuracy over the whole corpus (predictions cover every
    # trajectory). Constant across seed rows by construction; repeating it per
    # row lets it flow through the seed-mean into the summary unchanged.
    all_ids = [Path(f).stem for f in files]
    full_acc = {m: _acc(all_ids, p) for m, p in preds_by_method.items()}

    rows = []
    for seed in canonical_seeds(cfg):
        val_ids, test_ids = val_test_ids(files, splits["train"], splits["val"], seed)
        row = {"seed": seed, "n_val": len(val_ids), "n_test": len(test_ids),
               "n_full": len(all_ids), "gt_in_prompt": gt}
        best = {k: None for k in ("step_test", "agent_test", "step_full", "agent_full")}
        for m in methods:
            if m not in preds_by_method:
                for suf in ("step@1_val", "step@1_test", "step@1_full",
                            "agent@1_val", "agent@1_test", "agent@1_full"):
                    row[f"{m}_{suf}"] = None
                continue
            p = preds_by_method[m]
            _nv, av, sv = _acc(val_ids, p)
            _nt, at, st = _acc(test_ids, p)
            _nf, af, sf = full_acc[m]
            row[f"{m}_step@1_val"], row[f"{m}_step@1_test"], row[f"{m}_step@1_full"] = sv, st, sf
            row[f"{m}_agent@1_val"], row[f"{m}_agent@1_test"], row[f"{m}_agent@1_full"] = av, at, af
            for key, v in (("step_test", st), ("agent_test", at),
                           ("step_full", sf), ("agent_full", af)):
                best[key] = v if best[key] is None else max(best[key], v)
        row["baseline_best_step@1_test"] = best["step_test"]
        row["baseline_best_agent@1_test"] = best["agent_test"]
        row["baseline_best_step@1_full"] = best["step_full"]
        row["baseline_best_agent@1_full"] = best["agent_full"]
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(prog="baselines.prompting.report")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--gt", default=None, choices=["with", "without"],
                   help="Which setting to evaluate (default: the config's `gt`, else "
                        "'with'). 'without' reads/writes the outputs-nogt/ mirror.")
    p.add_argument("--check-only", action="store_true", help="Only print/write the completion status.")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)

    # gt: 'without' evaluates the outputs-nogt/ mirror; the gt_in_prompt table
    # column then follows the run setting, not the corpus-level config flag.
    gt = args.gt or cfg.get("gt", "with")
    if gt not in ("with", "without"):
        raise SystemExit(f"gt must be 'with' or 'without', got {gt!r}")
    if gt == "without":
        cfg["pred_root"] = nogt_root(cfg["pred_root"])
        cfg["out_root"] = nogt_root(cfg["out_root"])
        cfg["gt_in_prompt"] = False

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
