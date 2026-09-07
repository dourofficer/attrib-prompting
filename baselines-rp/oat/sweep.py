"""Sweep driver for OAT.

Grid over extractors × subsets; shells out one idempotent child per combo:

    python -m oat.sweep --config <yaml> [--set k=v ...] [--gt with|without] [--dry-run]

Config schema — the prompting sweep's shape, with OAT's own knobs as plain
config keys:

  models:      [qwen3.5-9b]      # extractor labels; the <model> path component
  subsets:     [hand-crafted]
  data_dir:    data/ww
  outputs_root: outputs-rb-gt/ww # --gt without mirrors this to outputs-rb-nogt/
  seeds:       [42, 43, 44, 45, 46]
  layer:       -1                 # which layer's hidden states to read
  aggregation: mean               # how to pool a step's tokens
  top_k:       3
  alpha:       0.2                # conformal miscoverage rate
  train_dir:   vendored/OAT/dataset/MCP-atlas/Qwen3.5-27B
  model_specs:
    qwen3.5-9b: {model_path: ../hub/Qwen/Qwen3.5-9B, dtype: bf16}

``model_specs`` here describes a *checkpoint to read hidden states from*, not a
generation server, so it takes ``model_path``, ``tokenizer``, ``dtype`` and
``device`` and nothing else — there is no sampling to configure.

GT axis: the vendored text carries no answer, so this sweep defaults to
``--gt without``, writing under ``outputs-rb-nogt/``.

``--dry-run`` prints every command unconditionally, so it works on a clean
checkout with no checkpoints present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.prompting.sweep import load_cfg, run
from baselines.shared.common import nogt_root


def extractor_args(model: str, spec: dict) -> list[str]:
    """predict argv for one extractor spec."""
    if "model_path" not in spec:
        raise SystemExit(f"model_specs.{model}: needs model_path (a local checkpoint, or 'dummy')")
    argv = ["--model-name", model, "--model-path", str(spec["model_path"])]
    for key, flag in (("tokenizer", "--tokenizer"), ("dtype", "--dtype"),
                      ("device", "--device"), ("attn_implementation", "--attn-implementation")):
        if spec.get(key):
            argv += [flag, str(spec[key])]
    return argv


def main() -> None:
    p = argparse.ArgumentParser(prog="oat.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--gt", default=None, choices=["with", "without"],
                   help="GT setting (default: the config's `gt`, else 'without' — "
                        "the vendored setting for this baseline). 'without' writes "
                        "under outputs-rb-nogt/.")
    p.add_argument("--stages", default=None,
                   help="Comma-separated stage subset passed through to predict.")
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
                *extractor_args(model, specs[model]),
                "--input", f"{cfg['data_dir']}/{subset}",
                "--output", f"{outputs_root}/{subset}/{model}",
                "--gt", gt,
            ]
            if cfg.get("seeds"):
                argv += ["--seeds", ",".join(str(s) for s in cfg["seeds"])]
            for key, flag in (
                ("layer", "--layer"),
                ("aggregation", "--aggregation"),
                ("latent_dim", "--latent-dim"),
                ("top_k", "--top-k"),
                ("alpha", "--alpha"),
                ("epochs", "--epochs"),
                ("patience", "--patience"),
                ("log_every", "--log-every"),
                ("train_device", "--train-device"),
                ("train_dir", "--train-dir"),
                ("train_root", "--train-root"),
                ("method_dir_prefix", "--method-dir-prefix"),
                ("start_idx", "--start_idx"),
                ("end_idx", "--end_idx"),
            ):
                if cfg.get(key) is not None:
                    argv += [flag, str(cfg[key])]
            if args.stages or cfg.get("stages"):
                argv += ["--stages", str(args.stages or cfg["stages"])]
            if cfg.get("overwrite"):
                argv += ["--overwrite"]
            run("oat.predict", argv, args.dry_run)


if __name__ == "__main__":
    main()
