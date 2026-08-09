"""Sweep driver for the CORRECT baseline — the whole 3-stage pipeline.

Per subset it runs, in order, each as a shelled-out idempotent child process:

  1. ``baselines.correct.schemagen``   — once, for ``schema_model`` (skipped
     when every trajectory already has a schema file);
  2. ``baselines.correct.similarity``  — once, model-independent (the child
     itself skips if the output exists);
  3. ``baselines.correct.predict``     — grid over models × methods.

    python -m baselines.correct.sweep --config <yaml> \\
        [--set k.sub=v ...] [--gt with|without] [--stages predict] [--dry-run]

Config schema = the prompting sweep's (``model_specs`` etc.) plus:

  schema_model: qwen3.5-9b           # who distills the schemata (all detectors share)
  embed_model:  ../hub/BAAI/bge-m3   # similarity encoder
  num_schemata: {hand-crafted: 10, algorithm-generated: 1}   # scalar or per-subset
  schema_gen:   {temperature: 0.7, ...}   # stage-1 sampling overlay; `params`
                                          # replaces the spec's params for stage 1

GT axis (GUIDE.md "GT settings"): the vendored cloud path never includes the
task answer, so this sweep defaults to ``--gt without`` — the paper setting —
mirroring detection outputs into ``outputs-nogt/``. ``--gt with`` inserts the
answer line and writes under ``outputs/``. Stage-1/2 artifacts are corpus-scoped
and GT-independent: they always live under the config's ``outputs_root`` and are
shared by both settings.

--dry-run prints every stage's command unconditionally (no completeness checks),
so it works on a clean checkout without models or artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from baselines.shared.common import nogt_root
from baselines.prompting.sweep import load_cfg, model_args, run

STAGES = ("schemagen", "similarity", "predict")

# Stage-1 sampling defaults = the vendored schema generator's SamplingParams;
# a config's schema_gen block overrides these, a model spec overrides both.
_SCHEMA_GEN_DEFAULTS = {"temperature": 0.7, "top_p": 0.95, "gen_max_tokens": 1024}


def _k_for(cfg: dict, subset: str) -> int:
    k = cfg.get("num_schemata", 1)
    if isinstance(k, dict):
        if subset not in k:
            raise SystemExit(f"num_schemata has no entry for subset {subset!r}")
        return int(k[subset])
    return int(k)


def _schemagen_complete(data_dir: Path, schemagen_dir: Path) -> bool:
    if not schemagen_dir.is_dir():
        return False
    done = {p.stem for p in schemagen_dir.glob("*.json") if p.stem.isdigit()}
    ids = {p.stem for p in data_dir.glob("*.json") if p.stem.isdigit()}
    return bool(ids) and ids <= done


def _common_argv(cfg: dict, extra_overwrite: bool = True) -> list[str]:
    argv = []
    if cfg.get("start_idx") is not None:
        argv += ["--start_idx", str(cfg["start_idx"])]
    if cfg.get("end_idx") is not None:
        argv += ["--end_idx", str(cfg["end_idx"])]
    if extra_overwrite and cfg.get("overwrite"):
        argv += ["--overwrite"]
    return argv


def main() -> None:
    p = argparse.ArgumentParser(prog="baselines.correct.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--gt", default=None, choices=["with", "without"],
                   help="GT setting (default: the config's `gt`, else 'without' — "
                        "the vendored/paper setting for this baseline). 'with' "
                        "inserts the answer line and writes under outputs/.")
    p.add_argument("--stages", default=",".join(STAGES),
                   help=f"Comma-separated subset of {STAGES} to run (default: all).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    specs = cfg.get("model_specs", {})
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s!r} (expected one of {STAGES})")

    gt = args.gt or cfg.get("gt", "without")
    if gt not in ("with", "without"):
        raise SystemExit(f"gt must be 'with' or 'without', got {gt!r}")
    outputs_root = cfg["outputs_root"]  # stage-1/2 artifacts: GT-independent
    detect_root = outputs_root if gt == "with" else nogt_root(outputs_root)

    schema_model = cfg.get("schema_model")
    embed_model = cfg.get("embed_model", "BAAI/bge-m3")
    embed_name = Path(str(embed_model)).name

    for subset in cfg["subsets"]:
        data_dir = f"{cfg['data_dir']}/{subset}"
        schemagen_dir = Path(outputs_root) / subset / str(schema_model) / "schemagen"
        sims_path = Path(outputs_root) / subset / "_similarities" / f"{embed_name}.json"

        if "schemagen" in stages:
            if not schema_model:
                raise SystemExit("config needs schema_model for the schemagen stage")
            if args.dry_run or cfg.get("overwrite") \
                    or not _schemagen_complete(Path(data_dir), schemagen_dir):
                if schema_model not in specs:
                    raise SystemExit(f"no model_specs entry for schema_model {schema_model!r}")
                sg = {**_SCHEMA_GEN_DEFAULTS, **(cfg.get("schema_gen") or {})}
                sg_spec = {**specs[schema_model],
                           **({"params": sg["params"]} if "params" in sg else {})}
                sg_cfg = {**cfg, **{k: v for k, v in sg.items() if k != "params"}}
                argv = [
                    *model_args(schema_model, sg_spec, sg_cfg),
                    "--model-name", schema_model,
                    "--input", data_dir,
                    "--output", f"{outputs_root}/{subset}/{schema_model}",
                    *_common_argv(cfg),
                ]
                run("baselines.correct.schemagen", argv, args.dry_run)
            else:
                print(f"  schemagen complete: {schemagen_dir}")

        if "similarity" in stages and (args.dry_run or not sims_path.exists()):
            argv = ["--input", data_dir, "--output", str(sims_path), "--model", str(embed_model)]
            if cfg.get("embed_batch_size") is not None:
                argv += ["--batch_size", str(cfg["embed_batch_size"])]
            if cfg.get("embed_max_length") is not None:
                argv += ["--max_length", str(cfg["embed_max_length"])]
            run("baselines.correct.similarity", argv, args.dry_run)

        if "predict" not in stages:
            continue
        for model in cfg["models"]:
            if model not in specs:
                raise SystemExit(f"no model_specs entry for {model!r}")
            spec = specs[model]
            for method in cfg["methods"]:
                argv = [
                    *model_args(model, spec, cfg),
                    "--model-name", model,
                    "--input", data_dir,
                    "--output", f"{detect_root}/{subset}/{model}",
                    "--method", method,
                    "--gt", gt,
                ]
                if method == "correct":
                    argv += [
                        "--schemata-dir", str(schemagen_dir),
                        "--similarities", str(sims_path),
                        "--num-schemata", str(_k_for(cfg, subset)),
                        "--schema-model", str(schema_model),
                    ]
                argv += _common_argv(cfg)
                run("baselines.correct.predict", argv, args.dry_run)


if __name__ == "__main__":
    main()
