"""Run StepFinder across encoders and subsets, one child process per pair.

Same shape as ``oat/sweep.py``: read a YAML config, resolve the output root
from the GT setting, and shell one ``stepfinder.predict`` per (encoder,
subset). The subprocess handling and the ``--set`` override parser come from
``baselines.prompting.sweep``, so every sweep in this repo builds and prints
commands the same way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.prompting.sweep import load_cfg, run
from baselines.shared.common import nogt_root


def encoder_args(model: str, spec: dict) -> list[str]:
    if "model_path" not in spec:
        raise SystemExit(f"model_specs[{model!r}] needs a `model_path`")
    argv = ["--model-name", model, "--model-path", str(spec["model_path"])]
    for key, flag in (("tokenizer", "--tokenizer"), ("dtype", "--dtype"),
                      ("device", "--device"), ("attn_implementation", "--attn-implementation")):
        if spec.get(key):
            argv += [flag, str(spec[key])]
    return argv


def main() -> None:
    p = argparse.ArgumentParser(prog="stepfinder.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--gt", default=None, choices=["with", "without"],
                   help="GT setting (default: the config's `gt`, else 'without' — "
                        "the vendored setting for this baseline). 'without' writes "
                        "under outputs-rb-nogt/.")
    p.add_argument("--protocol", default=None,
                   help="Comma-separated: regen, in-corpus (default: the config's).")
    p.add_argument("--stages", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    specs = cfg.get("model_specs", {})

    gt = args.gt or cfg.get("gt", "without")
    if gt not in ("with", "without"):
        raise SystemExit(f"gt must be 'with' or 'without', got {gt!r}")
    outputs_root = cfg["outputs_root"] if gt == "with" else nogt_root(cfg["outputs_root"])

    protocols = args.protocol or ",".join(cfg.get("protocols", ["regen"]))

    for model in cfg["models"]:
        if model not in specs:
            raise SystemExit(f"no model_specs entry for {model!r}")
        for subset in cfg["subsets"]:
            argv = [
                *encoder_args(model, specs[model]),
                "--input", f"{cfg['data_dir']}/{subset}",
                "--output", f"{outputs_root}/{subset}/{model}",
                "--gt", gt,
                "--protocol", protocols,
            ]
            if cfg.get("seeds"):
                argv += ["--seeds", ",".join(str(s) for s in cfg["seeds"])]
            if cfg.get("eval_seeds"):
                argv += ["--eval-seeds", ",".join(str(s) for s in cfg["eval_seeds"])]
            if cfg.get("splits"):
                sp = cfg["splits"]
                argv += ["--splits", f"{sp['train']},{sp['val']},{sp['test']}"]
            # Per-subset overrides win over the config-wide default.
            for key, flag in (("train_set", "--train-set"), ("preset", "--preset")):
                value = (cfg.get(f"{key}_overrides") or {}).get(subset, cfg.get(f"default_{key}"))
                if value is not None:
                    argv += [flag, str(value)]
            for key, flag in (
                ("train_data_root", "--train-data-root"),
                ("train_root", "--train-root"),
                ("model_selection", "--model-selection"),
                ("reduce", "--reduce"),
                ("val_frac", "--val-frac"),
                ("val_min_tasks", "--val-min-tasks"),
                ("val_min_size", "--val-min-size"),
                ("agent_normalize", "--agent-normalize"),
                ("epochs", "--epochs"),
                ("min_train_steps", "--min-train-steps"),
                ("batch_size", "--batch-size"),
                ("patience", "--patience"),
                ("top_k", "--top-k"),
                ("log_every", "--log-every"),
                ("train_device", "--train-device"),
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
            run("stepfinder.predict", argv, args.dry_run)


if __name__ == "__main__":
    main()
