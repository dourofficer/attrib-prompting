"""Sweep driver for the prompting baselines.

Grid over models × subsets × methods; shells out one child process per combo to
``baselines.prompting.predict`` so each heavy vLLM load happens once (API models
have no load cost — the child just fans out requests):

    python -m baselines.prompting.sweep --config <yaml> [--set k.sub=v ...] [--dry-run]

Model entries come from the config's ``model_specs`` map. A spec declares its
``backend`` and backend-specific fields; vLLM specs may also override any
top-level sampling/runtime knob (e.g. deepseek-8b bumps ``gen_max_tokens``):

    model_specs:
      qwen3.5-9b: {backend: vllm, model_path: ../hub/Qwen/Qwen3.5-9B}
      gpt-5:      {backend: openai, model: gpt-5, params: {max_completion_tokens: 4096}}

Runs are idempotent per trajectory: an existing ``<id>.json`` output is skipped
(``--set overwrite=true`` redoes everything).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml
from rich.console import Console

CONSOLE = Console()

# vLLM knobs a model spec may override; anything unset falls back to the
# top-level config value, then to this default.
_VLLM_KNOBS = {
    "dtype": "bfloat16",
    "seed": 0,
    "temperature": 0.6,
    "top_p": 0.95,
    "gen_max_tokens": 1024,
    "enable_thinking": False,
    "gpu_memory_utilization": 0.90,
    "tensor_parallel_size": 1,
    "max_model_len": None,
    "truncate_prompt_tokens": None,
}


def load_cfg(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts, node = key.split("."), cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


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


def _knob(spec: dict, cfg: dict, key: str):
    default = _VLLM_KNOBS[key]
    return spec.get(key, cfg.get(key, default))


def model_args(model: str, spec: dict, cfg: dict) -> list[str]:
    """Backend-specific predict argv for one model spec."""
    backend = spec.get("backend", "vllm")
    if backend == "vllm":
        if "model_path" not in spec:
            raise SystemExit(f"model_specs.{model}: vllm spec needs model_path")
        argv = [
            "--backend", "vllm",
            "--model", spec["model_path"],
            *(["--tokenizer", spec["tokenizer"]] if spec.get("tokenizer") else []),
            "--dtype", str(_knob(spec, cfg, "dtype")),
            "--seed", str(_knob(spec, cfg, "seed")),
            "--temperature", str(_knob(spec, cfg, "temperature")),
            "--top_p", str(_knob(spec, cfg, "top_p")),
            "--gen_max_tokens", str(_knob(spec, cfg, "gen_max_tokens")),
            "--enable_thinking", str(_knob(spec, cfg, "enable_thinking")),
            "--gpu_memory_utilization", str(_knob(spec, cfg, "gpu_memory_utilization")),
            "--tensor_parallel_size", str(_knob(spec, cfg, "tensor_parallel_size")),
        ]
        for key in ("max_model_len", "truncate_prompt_tokens"):
            if _knob(spec, cfg, key) is not None:
                argv += [f"--{key}", str(_knob(spec, cfg, key))]
        return argv
    if backend == "openai":
        argv = [
            "--backend", "openai",
            "--model", spec.get("model", model),
        ]
        if spec.get("base_url"):
            argv += ["--api-base-url", spec["base_url"]]
        if spec.get("api_key_env"):
            argv += ["--api-key-env", spec["api_key_env"]]
        for key, val in (spec.get("headers") or {}).items():
            argv += ["--api-header", f"{key}={val}"]
        if spec.get("concurrency") is not None:
            argv += ["--api-concurrency", str(spec["concurrency"])]
        if spec.get("max_retries") is not None:
            argv += ["--api-max-retries", str(spec["max_retries"])]
        for key, val in (spec.get("params") or {}).items():
            argv += ["--api-param", f"{key}={json.dumps(val)}"]  # YAML-parsed by predict
        return argv
    if backend == "dummy":
        return ["--backend", "dummy", "--model", spec.get("model", model)]
    raise SystemExit(f"model_specs.{model}: unknown backend {backend!r}")


def main() -> None:
    p = argparse.ArgumentParser(prog="baselines.prompting.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    specs = cfg.get("model_specs", {})

    for model in cfg["models"]:
        if model not in specs:
            raise SystemExit(f"no model_specs entry for {model!r}")
        spec = specs[model]
        for subset in cfg["subsets"]:
            for method in cfg["methods"]:
                argv = [
                    *model_args(model, spec, cfg),
                    "--model-name", model,
                    "--input", f"{cfg['data_dir']}/{subset}",
                    "--output", f"{cfg['outputs_root']}/{subset}/{model}",
                    "--method", method,
                ]
                if cfg.get("step_mode"):
                    argv += ["--step-mode", str(cfg["step_mode"])]
                if cfg.get("start_idx") is not None:
                    argv += ["--start_idx", str(cfg["start_idx"])]
                if cfg.get("end_idx") is not None:
                    argv += ["--end_idx", str(cfg["end_idx"])]
                if cfg.get("overwrite"):
                    argv += ["--overwrite"]
                run("baselines.prompting.predict", argv, args.dry_run)


if __name__ == "__main__":
    main()
