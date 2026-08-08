"""Sweep driver for the CHIEF baseline.

Grid over models × subsets; shells out one child process per combo to
``baselines.chief.predict`` so each heavy vLLM load happens once. Mirrors the
shared sweep interface used across the repo:

    python -m baselines.chief.sweep --config <yaml> [--set k.sub=v ...] [--dry-run]
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


def rag_args(cfg: dict) -> list[str]:
    rag = cfg.get("rag", {}) or {}
    enabled = bool(rag.get("enabled", False))
    argv = ["--rag", str(enabled)]
    if enabled:
        kbs = rag.get("kb", ["gaia", "assistantbench"])
        argv += ["--rag-kb", ",".join(kbs)]
    if rag.get("root"):
        argv += ["--rag-root", str(rag["root"])]
    if rag.get("top_k") is not None:
        argv += ["--rag-top-k", str(rag["top_k"])]
    return argv


def main() -> None:
    p = argparse.ArgumentParser(prog="baselines.chief.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)

    for model in cfg["models"]:
        model_path = resolve_model(cfg, model)
        tokenizer_path = resolve_tokenizer(cfg, model)
        for subset in cfg["subsets"]:
            argv = [
                "--model", model_path,
                *(["--tokenizer", tokenizer_path] if tokenizer_path else []),
                "--input", f"{cfg['data_dir']}/{subset}",
                "--output", f"{cfg['outputs_root']}/{model}/{subset}",
                "--mode", str(cfg.get("mode", "batched")),
                *rag_args(cfg),
                "--dtype", cfg.get("dtype", "bfloat16"),
                "--seed", str(cfg.get("seed", 0)),
                "--temperature", str(cfg.get("temperature", 0.0)),
                "--top_p", str(cfg.get("top_p", 1.0)),
                "--gen_max_tokens", str(cfg.get("gen_max_tokens", 2048)),
                "--enable_thinking", str(cfg.get("enable_thinking", False)),
                "--gpu_memory_utilization", str(cfg.get("gpu_memory_utilization", 0.90)),
                "--tensor_parallel_size", str(cfg.get("tensor_parallel_size", 1)),
                *index_args(cfg),
            ]
            if cfg.get("max_model_len") is not None:
                argv += ["--max_model_len", str(cfg["max_model_len"])]
            if cfg.get("truncate_prompt_tokens") is not None:
                argv += ["--truncate_prompt_tokens", str(cfg["truncate_prompt_tokens"])]
            if cfg.get("overwrite"):
                argv += ["--overwrite"]
            run("baselines.chief.predict", argv, args.dry_run)


if __name__ == "__main__":
    main()
