"""Prompting-baseline runner — one (model, subset, method) per invocation.

Runs a Who&When attribution baseline over *all* trajectories in a subset with
VLLM (batched) and records one prediction row per trajectory. Evaluation on the
CRR test splits is deferred to ``report.py`` — inference here is split-agnostic.

Usage
-----
python -m baselines.prompting.predict \
    --model  /data/hoang/resources/models/Qwen/Qwen3.5-9B \
    --input  data/ww/hand-crafted \
    --output outputs-ww/prompting/qwen3.5-9b/hand-crafted \
    --method all_at_once \
    --enable_thinking False

Output
------
{output}/predictions_method-{method}.jsonl   (one JSON object per trajectory)
{output}/config.json                          (run snapshot)
Idempotent: skips if the predictions file already exists unless --overwrite.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.utils.common import _get_sorted_json_files, _load_json_data

from .engine import PromptEngine
from .methods import METHODS


def _bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def load_records(directory: str) -> list[dict]:
    """Load raw trajectory JSONs into method-ready records.

    Reads fields straight from JSON (mirroring the vendored baseline), which also
    gives us ``ground_truth`` — not carried on the ``Trajectory`` dataclass.
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
    p = argparse.ArgumentParser(description="Run a prompting attribution baseline with VLLM.")
    p.add_argument("--model", required=True, help="HF model name or local path.")
    p.add_argument("--tokenizer", default=None,
                   help="Optional tokenizer path override (e.g. a corrected tokenizer dir).")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True, help="Output directory.")
    p.add_argument("--method", required=True, choices=list(METHODS))
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
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output)
    out_path = out_dir / f"predictions_method-{args.method}.jsonl"
    if out_path.exists() and not args.overwrite:
        print(f"skip (exists): {out_path}")
        return

    records = load_records(args.input)
    end_idx = args.end_idx if args.end_idx is not None else len(records)
    records = records[args.start_idx:end_idx]
    print(f"  {len(records)} trajectories [{args.start_idx}:{end_idx}] from {args.input}")
    if not records:
        print("  nothing to do")
        return

    engine = PromptEngine(
        args.model,
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

    t0 = time.perf_counter()
    preds = METHODS[args.method](records, engine)
    elapsed = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r, pred in zip(records, preds):
            row = {
                "id": r["id"],
                "filename": r["filename"],
                "question_id": r["question_id"],
                "predicted_agent": pred["predicted_agent"],
                "predicted_step": pred["predicted_step"],
                "gold_agent": r["gold_agent"],
                "gold_step": r["gold_step"],
                "raw": pred["raw"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "model": args.model,
        "method": args.method,
        "subset": Path(args.input).name,
        "dtype": args.dtype,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gen_max_tokens": args.gen_max_tokens,
        "enable_thinking": args.enable_thinking,
        "n_trajectories": len(records),
    }
    (out_dir / f"config_method-{args.method}.json").write_text(json.dumps(config, indent=2))

    print(f"  wrote {out_path}  ({len(preds)} rows, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
