"""Sweep driver for the CORRECT baseline.

Grid over models × subsets; per combo, shells out up to three child processes
(each stage owns its heavy load and is idempotent — it exits early if its
output already exists, before touching the GPU):

  1. ``baselines.correct.schemagen``  — per-(model, subset) schema cache
  2. ``baselines.correct.similarity`` — per-subset ranked neighbours
     (model-independent: BGE-M3 embeddings; run once, shared by all models)
  3. ``baselines.correct.predict``    — schema-guided detection

Mirrors the shared sweep interface used across the repo:

    python -m baselines.correct.sweep --config <yaml> [--set k.sub=v ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml
from rich.console import Console

CONSOLE = Console()


def load_cfg(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts, node = key.split("."), cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def resolve_model(cfg: dict, model: str) -> str:
    return cfg.get("model_paths", {}).get(model, model)


def resolve_tokenizer(cfg: dict, model: str) -> str | None:
    return cfg.get("tokenizer_paths", {}).get(model)


def resolve_num_schemata(cfg: dict, subset: str) -> int:
    """``num_schemata`` may be a single int or a per-subset mapping."""
    k = cfg.get("num_schemata", 1)
    if isinstance(k, dict):
        return int(k[subset])
    return int(k)


def format_command(module: str, argv: list[str]) -> str:
    head = f"{sys.executable} -m {module}"
    if not argv:
        return head
    groups, current = [], []
    for token in argv:
        if token.startswith("--") and current:
            groups.append(current)
            current = []
        current.append(token)
    groups.append(current)
    args = " \\\n    ".join(" ".join(shlex.quote(t) for t in g) for g in groups)
    return f"{head} \\\n    {args}"


def run(module: str, argv: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, "-m", module, *argv]
    CONSOLE.print(format_command(module, argv), style="green")
    CONSOLE.rule()
    if not dry_run:
        subprocess.run(cmd, check=True)


def index_args(cfg: dict) -> list[str]:
    argv: list[str] = []
    if cfg.get("start_idx") is not None:
        argv += ["--start_idx", str(cfg["start_idx"])]
    if cfg.get("end_idx") is not None:
        argv += ["--end_idx", str(cfg["end_idx"])]
    return argv


def engine_args(cfg: dict) -> list[str]:
    """vLLM/engine knobs shared by schemagen and predict."""
    argv = [
        "--dtype", cfg.get("dtype", "bfloat16"),
        "--seed", str(cfg.get("seed", 0)),
        "--enable_thinking", str(cfg.get("enable_thinking", False)),
        "--gpu_memory_utilization", str(cfg.get("gpu_memory_utilization", 0.90)),
        "--tensor_parallel_size", str(cfg.get("tensor_parallel_size", 1)),
    ]
    if cfg.get("max_model_len") is not None:
        argv += ["--max_model_len", str(cfg["max_model_len"])]
    if cfg.get("truncate_prompt_tokens") is not None:
        argv += ["--truncate_prompt_tokens", str(cfg["truncate_prompt_tokens"])]
    return argv


def main() -> None:
    p = argparse.ArgumentParser(prog="baselines.correct.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    outputs_root = cfg["outputs_root"]
    schema_gen = cfg.get("schema_gen", {}) or {}
    overwrite = ["--overwrite"] if cfg.get("overwrite") else []

    for model in cfg["models"]:
        model_path = resolve_model(cfg, model)
        tokenizer_path = resolve_tokenizer(cfg, model)
        tokenizer = ["--tokenizer", tokenizer_path] if tokenizer_path else []
        for subset in cfg["subsets"]:
            num_schemata = resolve_num_schemata(cfg, subset)
            schemata_path = f"{outputs_root}/schemata/{model}/{subset}/error_schemata.txt"
            similarities_path = (f"{outputs_root}/similarities/"
                                 f"{subset}_trajectory_similarities.json")

            if num_schemata > 0:
                # 1. Offline schema cache (per model × subset).
                run("baselines.correct.schemagen", [
                    "--model", model_path,
                    *tokenizer,
                    "--input", f"{cfg['data_dir']}/{subset}",
                    "--output", f"{outputs_root}/schemata/{model}/{subset}",
                    "--temperature", str(schema_gen.get("temperature", 0.7)),
                    "--top_p", str(schema_gen.get("top_p", 0.95)),
                    "--gen_max_tokens", str(schema_gen.get("max_tokens", 1024)),
                    *engine_args(cfg),
                    *overwrite,
                ], args.dry_run)

                # 2. Ranked neighbours (per subset, model-independent).
                run("baselines.correct.similarity", [
                    "--input", f"{cfg['data_dir']}/{subset}",
                    "--output", similarities_path,
                    "--model", cfg.get("embed_model_path", "BAAI/bge-m3"),
                    *overwrite,
                ], args.dry_run)

            # 3. Schema-guided detection.
            run("baselines.correct.predict", [
                "--model", model_path,
                *tokenizer,
                "--input", f"{cfg['data_dir']}/{subset}",
                "--output", f"{outputs_root}/{model}/{subset}",
                *(["--schemata", schemata_path,
                   "--similarities", similarities_path] if num_schemata > 0 else []),
                "--num_schemata", str(num_schemata),
                "--scan_until_filled", str(cfg.get("scan_until_filled", False)),
                "--temperature", str(cfg.get("temperature", 0.0)),
                "--top_p", str(cfg.get("top_p", 1.0)),
                "--gen_max_tokens", str(cfg.get("gen_max_tokens", 1024)),
                *engine_args(cfg),
                *index_args(cfg),
                *overwrite,
            ], args.dry_run)


if __name__ == "__main__":
    main()
