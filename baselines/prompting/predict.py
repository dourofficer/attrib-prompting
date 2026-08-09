"""Prompting-baseline runner — one (model, subset, method) per invocation.

Runs a Who&When attribution baseline over *all* trajectories in a subset and
writes one JSON file per trajectory (prediction + gold + full call log) under
``{output}/{method}/``. Inference is split-agnostic; evaluation on the per-seed
splits is deferred entirely to ``report.py``.

Backends
--------
``--backend vllm``   (default) local checkpoint, one giant lockstep batch.
``--backend openai`` any OpenAI-compatible chat-completions API; trajectories
                     run concurrently and each output file is written the
                     moment its trajectory completes.
``--backend dummy``  deterministic canned responses (keyless end-to-end check).

Resume
------
An existing ``{output}/{method}/<id>.json`` marks that trajectory done: rerun
the same command and only missing ids are executed (``--overwrite`` clears the
method directory instead). For API runs this bounds the cost of a crash to the
trajectories in flight.

Usage
-----
python -m baselines.prompting.predict \
    --model  ../hub/Qwen/Qwen3.5-9B --model-name qwen3.5-9b \
    --input  data/ww/hand-crafted \
    --output outputs/ww/hand-crafted/qwen3.5-9b \
    --method all_at_once

python -m baselines.prompting.predict \
    --backend openai --model gpt-4o \
    --api-param max_tokens=1024 --api-param temperature=0.6 \
    --input data/ww/hand-crafted --output outputs/ww/hand-crafted/gpt-4o \
    --method step_by_step

Output
------
{output}/{method}/<id>.json   one file per trajectory
{output}/{method}/_run.json   run snapshot (params actually sent, counts)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from baselines.shared.backends import get_backend
from baselines.shared.common import _get_sorted_json_files, _load_json_data
from baselines.shared.runner import OutputWriter, run_batched, run_streaming

from .methods import METHODS


def _bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def load_records(directory: str) -> list[dict]:
    """Load raw trajectory JSONs into method-ready records.

    Reads fields straight from JSON (mirroring the vendored baseline), which also
    gives us ``ground_truth`` and the gold labels carried into the output files.
    """
    records = []
    for fn in _get_sorted_json_files(directory):
        data = _load_json_data(Path(directory) / fn)
        if not data:
            continue
        history = data.get("history", [])
        if not history:
            continue
        records.append({
            "id": Path(fn).stem,
            "filename": fn,
            "question_id": data.get("question_ID") or data.get("question_id"),
            "history": history,
            "question": data.get("question", ""),
            "ground_truth": data.get("ground_truth", ""),
            "gold_agent": data.get("mistake_agent"),
            "gold_step": data.get("mistake_step"),
        })
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a prompting attribution baseline.")
    p.add_argument("--model", required=True,
                   help="Local checkpoint path (vllm) or API model name (openai).")
    p.add_argument("--model-name", default=None,
                   help="Short label recorded in outputs (default: --model).")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True,
                   help="Model-level output directory; files land in {output}/{method}/.")
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--backend", default="vllm", choices=["vllm", "openai", "dummy"])
    p.add_argument("--step-mode", default="auto", choices=["auto", "batch", "early_stop"],
                   help="step_by_step execution mode; auto = batch for vllm, "
                        "early_stop (the vendored control flow) otherwise.")
    p.add_argument("--gt", default="with", choices=["with", "without"],
                   help="'without' removes the 'The Answer for the problem is:' line "
                        "from every prompt. Affects prompts only — point --output at "
                        "an outputs-nogt/ root to keep the settings apart.")
    # vLLM-only knobs
    p.add_argument("--tokenizer", default=None,
                   help="Optional tokenizer path override (e.g. a corrected tokenizer dir).")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "auto"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--gen_max_tokens", type=int, default=1024)
    p.add_argument("--enable_thinking", type=_bool, default=False)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--truncate_prompt_tokens", type=int, default=None,
                   help="Safety net: keep only the last N prompt tokens instead of "
                        "erroring on over-length prompts. Off by default.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    # API-only knobs
    p.add_argument("--api-base-url", default=None,
                   help="OpenAI-compatible endpoint (default: the OpenAI API).")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY",
                   help="Env var holding the API key.")
    p.add_argument("--api-header", action="append", default=[], metavar="KEY=VALUE",
                   help="Extra HTTP header sent with every API request (repeatable), "
                        "e.g. X-Llmhub-Channel=1.")
    p.add_argument("--api-concurrency", type=int, default=8,
                   help="Max in-flight API requests.")
    p.add_argument("--api-max-retries", type=int, default=6)
    p.add_argument("--api-param", action="append", default=[], metavar="KEY=VALUE",
                   help="Request parameter sent verbatim (repeatable), e.g. "
                        "--api-param max_completion_tokens=4096. These are the ONLY "
                        "generation params an API model receives.")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="Clear {output}/{method}/ and redo every trajectory.")
    return p.parse_args()


def _api_params(pairs: list[str]) -> dict:
    params = {}
    for pair in pairs:
        key, sep, val = pair.partition("=")
        if not sep:
            raise SystemExit(f"--api-param expects KEY=VALUE, got {pair!r}")
        params[key.strip()] = yaml.safe_load(val)
    return params


def _api_headers(pairs: list[str]) -> dict[str, str]:
    headers = {}
    for pair in pairs:
        key, sep, val = pair.partition("=")
        if not sep:
            raise SystemExit(f"--api-header expects KEY=VALUE, got {pair!r}")
        headers[key.strip()] = val.strip()
    return headers


def build_backend(args: argparse.Namespace):
    if args.backend == "vllm":
        return get_backend(
            "vllm",
            model_path=args.model,
            tokenizer=args.tokenizer,
            dtype=args.dtype,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_gen_tokens=args.gen_max_tokens,
            enable_thinking=args.enable_thinking,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            truncate_prompt_tokens=args.truncate_prompt_tokens,
        )
    if args.backend == "openai":
        return get_backend(
            "openai",
            model=args.model,
            base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            headers=_api_headers(args.api_header),
            params=_api_params(args.api_param),
            concurrency=args.api_concurrency,
            max_retries=args.api_max_retries,
        )
    return get_backend("dummy")


def main() -> None:
    args = parse_args()
    model_name = args.model_name or args.model

    writer = OutputWriter(Path(args.output) / args.method, overwrite=args.overwrite)
    done = writer.done_ids()

    records = load_records(args.input)
    end_idx = args.end_idx if args.end_idx is not None else len(records)
    records = records[args.start_idx:end_idx]
    remaining = [r for r in records if r["id"] not in done]
    print(f"  {len(records)} trajectories [{args.start_idx}:{end_idx}] from {args.input}"
          f" — {len(done)} done, {len(remaining)} to run")
    if not remaining:
        print(f"  skip (complete): {writer.dir}")
        return

    backend = build_backend(args)
    step_mode = args.step_mode
    if step_mode == "auto":
        step_mode = "early_stop" if backend.prefers_streaming else "batch"
    include_gt = args.gt == "with"

    writer.write_run_config({
        "model": model_name,
        "model_arg": args.model,
        "method": args.method,
        "subset": Path(args.input).name,
        "backend": args.backend,
        "request_params": backend.request_params,
        "step_mode": step_mode if args.method == "step_by_step" else None,
        "gt_in_prompt": include_gt,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_total": len(records),
        "n_already_done": len(done),
        "n_remaining": len(remaining),
        "resumed": bool(done),
    })

    def on_done(record: dict, pred: dict) -> None:
        doc = {
            "id": record["id"],
            "filename": record["filename"],
            "question_id": record["question_id"],
            "method": args.method,
            "model": model_name,
            "backend": args.backend,
            "gt_in_prompt": include_gt,
            "predicted_agent": pred["predicted_agent"],
            "predicted_step": pred["predicted_step"],
            "gold_agent": record["gold_agent"],
            "gold_step": record["gold_step"],
            "raw": pred["raw"],
            "calls": pred["calls"],
        }
        writer.write(record["id"], doc)

    programs = [(r, METHODS[args.method](r, step_mode=step_mode, include_gt=include_gt))
                for r in remaining]

    t0 = time.perf_counter()
    if backend.prefers_streaming:
        run_streaming(programs, backend, on_done, max_workers=args.api_concurrency)
    else:
        run_batched(programs, backend, on_done)
    elapsed = time.perf_counter() - t0

    n_written = len(writer.done_ids())
    print(f"  wrote {writer.dir}  ({n_written}/{len(records)} files, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
