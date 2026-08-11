"""Sweep driver for the CHIEF baseline — both stages.

Per subset it runs, in order, each as a shelled-out idempotent child process:

  1. ``baselines.chief.ragprep``  — once, detector-independent (the child skips
     if the artifact exists);
  2. ``baselines.chief.predict``  — grid over models.

    python -m baselines.chief.sweep --config <yaml> \\
        [--set k.sub=v ...] [--gt with|without] [--stages predict] [--dry-run]

Config schema = the prompting sweep's (``model_specs`` etc.) plus:

  rag:
    root:   vendored/CHIEF/rag       # index/ + kb/ of the committed knowledge base
    kb:     [gaia, assistantbench]   # which indices to search
    top_k:  2                        # vendored; the [1:top_k] slice yields 1 exemplar
    embed_model: sentence-transformers/all-MiniLM-L6-v2
  artifacts_root: artifacts/ww       # stage-1 artifacts (default: outputs_root with
                                     # its root component swapped to artifacts/)

``outputs_root`` holds predictions only. Retrieval writes to ``artifacts_root``,
stage first, then the encoder that produced it::

    artifacts/<ds>/<subset>/rag/<embed_model>.json

GT axis (GUIDE.md "GT settings"): every vendored CHIEF prompt carries the task
answer, so this sweep defaults to ``--gt with`` — the vendored setting — writing
under ``outputs/``. ``--gt without`` drops the answer sentence and mirrors into
``outputs-nogt/``. The RAG artifact is keyed by the question alone, so one
``artifacts/`` root serves both settings.

--dry-run prints every stage's command unconditionally (no completeness checks),
so it works on a clean checkout without models or artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from baselines.shared.common import artifacts_root, nogt_root
from baselines.prompting.sweep import load_cfg, model_args, run

from .rag import DEFAULT_RAG_ROOT, EMBED_MODEL

STAGES = ("ragprep", "predict")


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
    p = argparse.ArgumentParser(prog="baselines.chief.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--gt", default=None, choices=["with", "without"],
                   help="GT setting (default: the config's `gt`, else 'with' — the "
                        "vendored setting for this baseline, whose every stage prompt "
                        "carries the answer). 'without' writes under outputs-nogt/.")
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

    gt = args.gt or cfg.get("gt", "with")
    if gt not in ("with", "without"):
        raise SystemExit(f"gt must be 'with' or 'without', got {gt!r}")
    outputs_root = cfg["outputs_root"]          # predictions
    detect_root = outputs_root if gt == "with" else nogt_root(outputs_root)
    # Stage-1 artifacts live outside both GT trees (they are GT-independent).
    art_root = cfg.get("artifacts_root") or artifacts_root(outputs_root)

    rag = cfg.get("rag") or {}
    rag_kbs = rag.get("kb", ["gaia", "assistantbench"])
    embed_model = rag.get("embed_model", EMBED_MODEL)
    embed_name = Path(str(embed_model)).name

    for subset in cfg["subsets"]:
        data_dir = f"{cfg['data_dir']}/{subset}"
        rag_path = Path(art_root) / subset / "rag" / f"{embed_name}.json"

        if "ragprep" in stages and rag_kbs and (args.dry_run or not rag_path.exists()):
            argv = [
                "--input", data_dir,
                "--output", str(rag_path),
                "--rag-root", str(rag.get("root", DEFAULT_RAG_ROOT)),
                "--rag-kb", ",".join(rag_kbs),
                "--rag-top-k", str(rag.get("top_k", 2)),
                "--embed-model", str(embed_model),
            ]
            run("baselines.chief.ragprep", argv, args.dry_run)

        if "predict" not in stages:
            continue
        for model in cfg["models"]:
            if model not in specs:
                raise SystemExit(f"no model_specs entry for {model!r}")
            argv = [
                *model_args(model, specs[model], cfg),
                "--model-name", model,
                "--input", data_dir,
                "--output", f"{detect_root}/{subset}/{model}",
                "--gt", gt,
            ]
            if rag_kbs:
                argv += ["--rag-texts", str(rag_path)]
            argv += _common_argv(cfg)
            run("baselines.chief.predict", argv, args.dry_run)


if __name__ == "__main__":
    main()
