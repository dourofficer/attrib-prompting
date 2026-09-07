"""Sweep driver for the ErrorProbe baseline.

Grid over models × subsets × modes; shells out one idempotent child process per
combo:

    python -m baselines.errorprobe.sweep --config <yaml> \\
        [--set k.sub=v ...] [--gt with|without] [--dry-run]

Config schema = the prompting sweep's (``model_specs`` etc.) plus ErrorProbe's
own knobs, all plain config keys:

  modes: [truncated]   # truncated (vendored default), backward, and/or paper
  method_dir: null     # output dir override (only sensible with one mode)
  paper:               # knobs for the paper mode only (see predict.py --help)
    max_hypotheses: 3
    chunk_chars: 12000
    condensed_chars: 20000
    include_mast_examples: false
    sequential_edges: fallback

A model spec may carry its own ``paper`` block. Its keys override the
top-level block for that model, and it may also set any vLLM sampling or
runtime knob (``gen_max_tokens``, ``max_model_len``,
``gpu_memory_utilization``, ...) that applies to the paper mode alone — the
other modes keep the spec's and config's usual values. That is how one model
runs the paper mode on a smaller budget than another without a second config.

GT axis (GUIDE.md "GT settings"): the vendored prompts never carry the task
answer, so this sweep defaults to ``--gt without`` — the vendored setting —
writing under ``outputs-nogt/``. ``--gt with`` adds the answer line and writes
under ``outputs/``.

--dry-run prints every command unconditionally (no completeness checks), so it
works on a clean checkout without models or keys.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from baselines.shared.common import nogt_root
from baselines.prompting.sweep import _VLLM_KNOBS, load_cfg, model_args, run

from .paper.program import PAPER_DEFAULTS
from .predict import MODES


def paper_args(model: str, spec: dict, cfg: dict) -> list[str]:
    """Flags for the paper mode: the config's ``paper`` block, then the spec's.

    Paper knobs become ``--max-hypotheses``-style flags. vLLM knobs named in a
    paper block become their usual ``--gen_max_tokens``-style flags and, coming
    after ``model_args``, override the spec's values for this mode only; they
    are dropped for non-vLLM backends.
    """
    merged = {**(cfg.get("paper") or {}), **(spec.get("paper") or {})}
    argv: list[str] = []
    for key, val in merged.items():
        if key in PAPER_DEFAULTS:
            argv += [f"--{key.replace('_', '-')}", str(val)]
        elif key in _VLLM_KNOBS:
            if spec.get("backend", "vllm") == "vllm":
                argv += [f"--{key}", str(val)]
        else:
            raise SystemExit(f"model_specs.{model}.paper / paper: unknown knob {key!r}")
    return argv


def main() -> None:
    p = argparse.ArgumentParser(prog="baselines.errorprobe.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--gt", default=None, choices=["with", "without"],
                   help="GT setting (default: the config's `gt`, else 'without' — "
                        "the vendored setting for this baseline). 'without' "
                        "writes under outputs-nogt/.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    specs = cfg.get("model_specs", {})

    gt = args.gt or cfg.get("gt", "without")
    if gt not in ("with", "without"):
        raise SystemExit(f"gt must be 'with' or 'without', got {gt!r}")
    outputs_root = cfg["outputs_root"] if gt == "with" else nogt_root(cfg["outputs_root"])

    modes = cfg.get("modes", ["truncated"])
    for mode in modes:
        if mode not in MODES:
            raise SystemExit(f"unknown mode {mode!r} (expected one of {sorted(MODES)})")

    for model in cfg["models"]:
        if model not in specs:
            raise SystemExit(f"no model_specs entry for {model!r}")
        for subset in cfg["subsets"]:
            for mode in modes:
                argv = [
                    *model_args(model, specs[model], cfg),
                    "--model-name", model,
                    "--input", f"{cfg['data_dir']}/{subset}",
                    "--output", f"{outputs_root}/{subset}/{model}",
                    "--mode", mode,
                    "--gt", gt,
                ]
                if cfg.get("method_dir"):
                    argv += ["--method-dir", str(cfg["method_dir"])]
                if mode == "paper":
                    argv += paper_args(model, specs[model], cfg)
                if cfg.get("start_idx") is not None:
                    argv += ["--start_idx", str(cfg["start_idx"])]
                if cfg.get("end_idx") is not None:
                    argv += ["--end_idx", str(cfg["end_idx"])]
                if cfg.get("overwrite"):
                    argv += ["--overwrite"]
                run("baselines.errorprobe.predict", argv, args.dry_run)


if __name__ == "__main__":
    main()
