"""Sweep driver for the RAFFLES baseline.

Grid over models × subsets; shells out one idempotent child process per combo:

    python -m baselines.raffles.sweep --config <yaml> \\
        [--set k.sub=v ...] [--gt with|without] [--dry-run]

Config schema = the prompting sweep's (``model_specs`` etc.) plus RAFFLES's own
knobs, all plain config keys:

  max_iters:  2       # K — extra iterations after the first pass (paper: 2)
  threshold:  350     # early-stop confidence (of 400)
  method_dir: null    # output dir override, e.g. raffles.k5 for side-by-side runs

GT axis (GUIDE.md "GT settings"): the paper evaluates Who&When without ground
truth, so this sweep defaults to ``--gt without`` — the paper setting — writing
under ``outputs-nogt/``. ``--gt with`` adds the answer line and writes under
``outputs/``.

--dry-run prints every command unconditionally (no completeness checks), so it
works on a clean checkout without models or keys.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from baselines.shared.common import nogt_root
from baselines.prompting.sweep import load_cfg, model_args, run


def main() -> None:
    p = argparse.ArgumentParser(prog="baselines.raffles.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--gt", default=None, choices=["with", "without"],
                   help="GT setting (default: the config's `gt`, else 'without' — "
                        "the paper setting for this baseline). 'without' writes "
                        "under outputs-nogt/.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    specs = cfg.get("model_specs", {})

    gt = args.gt or cfg.get("gt", "without")
    if gt not in ("with", "without"):
        raise SystemExit(f"gt must be 'with' or 'without', got {gt!r}")
    outputs_root = cfg["outputs_root"] if gt == "with" else nogt_root(cfg["outputs_root"])

    for model in cfg["models"]:
        if model not in specs:
            raise SystemExit(f"no model_specs entry for {model!r}")
        for subset in cfg["subsets"]:
            argv = [
                *model_args(model, specs[model], cfg),
                "--model-name", model,
                "--input", f"{cfg['data_dir']}/{subset}",
                "--output", f"{outputs_root}/{subset}/{model}",
                "--gt", gt,
            ]
            if cfg.get("max_iters") is not None:
                argv += ["--max-iters", str(cfg["max_iters"])]
            if cfg.get("threshold") is not None:
                argv += ["--threshold", str(cfg["threshold"])]
            if cfg.get("method_dir"):
                argv += ["--method-dir", str(cfg["method_dir"])]
            if cfg.get("start_idx") is not None:
                argv += ["--start_idx", str(cfg["start_idx"])]
            if cfg.get("end_idx") is not None:
                argv += ["--end_idx", str(cfg["end_idx"])]
            if cfg.get("overwrite"):
                argv += ["--overwrite"]
            run("baselines.raffles.predict", argv, args.dry_run)


if __name__ == "__main__":
    main()
