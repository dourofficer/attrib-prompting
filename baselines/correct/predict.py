"""CORRECT-baseline runner — one (model, subset) per invocation.

Runs schema-guided all-at-once error detection over *all* trajectories in a
subset with VLLM (batched) and records one prediction row per trajectory, in the
house JSONL format the shared report consumes. Evaluation on the CRR val/test
splits is deferred to ``report.py`` — inference here is split-agnostic.

Requires the two offline artifacts (produced by ``schemagen`` / ``similarity``):
``--schemata`` (error_schemata.txt) and ``--similarities`` (ranked-neighbour
JSON). With ``--num_schemata 0`` it runs the vendored no-schema LLM-as-a-Judge
baseline instead (method name ``correct-base``; artifacts not needed).

Usage
-----
python -m baselines.correct.predict \
    --model  ../hub/Qwen/Qwen3.5-9B \
    --input  data/ww/hand-crafted \
    --output outputs-ww/correct/qwen3.5-9b/hand-crafted \
    --schemata     outputs-ww/correct/schemata/qwen3.5-9b/hand-crafted/error_schemata.txt \
    --similarities outputs-ww/correct/similarities/hand-crafted_trajectory_similarities.json \
    --num_schemata 10

Output
------
{output}/predictions_method-{method}.jsonl   (one JSON object per trajectory)
{output}/config_method-{method}.json          (run snapshot)
Idempotent: skips if the predictions file already exists unless --overwrite.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from baselines.prompting.predict import load_records

from .engine import PromptEngine
from .methods import build_all_at_once_prompt, inject_schemata, messages, parse_prediction
from .retrieval import SchemaAnalyzer, load_error_schemata, load_trajectory_similarities


def _bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the CORRECT baseline with VLLM.")
    p.add_argument("--model", required=True, help="HF model name or local path.")
    p.add_argument("--tokenizer", default=None,
                   help="Optional tokenizer path override (e.g. a corrected tokenizer dir).")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True, help="Output directory.")
    p.add_argument("--schemata", default=None, help="Path to error_schemata.txt.")
    p.add_argument("--similarities", default=None,
                   help="Path to {subset}_trajectory_similarities.json.")
    p.add_argument("--num_schemata", type=int, default=1,
                   help="Top-k schemata to retrieve; 0 = no-schema baseline mode.")
    p.add_argument("--scan_until_filled", type=_bool, default=False,
                   help="Retrieval variant: False = Who&When script (inspect only "
                        "top-k neighbours), True = CORRECT-Error script (scan until "
                        "k schemata found).")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "auto"])
    p.add_argument("--seed", type=int, default=0)
    # Vendored vLLM inference is greedy: temperature=0.0, top_p=1.0, max_tokens=1024.
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
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

    baseline_mode = args.num_schemata == 0
    method = "correct-base" if baseline_mode else "correct"

    out_dir = Path(args.output)
    out_path = out_dir / f"predictions_method-{method}.jsonl"
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

    analyzer = None
    if not baseline_mode:
        if not args.schemata or not args.similarities:
            raise SystemExit("--schemata and --similarities are required when num_schemata > 0")
        schemata = load_error_schemata(args.schemata)
        similarities = load_trajectory_similarities(args.similarities)
        analyzer = SchemaAnalyzer(schemata, similarities,
                                  scan_until_filled=args.scan_until_filled)
        print(f"  {len(schemata)} schemata, similarities for {len(similarities)} trajectories, "
              f"k={args.num_schemata}")

    # Build all prompts up front (mirrors the vendored single llm.generate call).
    message_lists = []
    schema_usage: dict[str, list[int]] = {}
    for r in records:
        prompt = build_all_at_once_prompt(r["history"], r["question"])
        if analyzer is not None:
            file_num = int("".join(filter(str.isdigit, r["filename"])) or 0)
            schema_keys, schema_contents = analyzer.get_similarity_based_schema(
                file_num, args.num_schemata)
            if schema_contents:
                prompt = inject_schemata(prompt, schema_contents)
                schema_usage[r["filename"]] = schema_keys
        message_lists.append(messages(prompt))

    if analyzer is not None:
        missing = len(records) - len(schema_usage)
        if missing:
            print(f"  WARNING: {missing} trajectories got no schema (base prompt only)")

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
    outputs = engine.generate(message_lists)
    elapsed = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r, raw in zip(records, outputs):
            agent, step, _trimmed = parse_prediction(raw)
            row = {
                "id": r["id"],
                "filename": r["filename"],
                "question_id": r["question_id"],
                "predicted_agent": agent,
                "predicted_step": step,
                "gold_agent": r["gold_agent"],
                "gold_step": r["gold_step"],
                "schema_keys": schema_usage.get(r["filename"]),
                "raw": raw,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "model": args.model,
        "method": method,
        "subset": Path(args.input).name,
        "num_schemata": args.num_schemata,
        "scan_until_filled": args.scan_until_filled,
        "schemata": args.schemata,
        "similarities": args.similarities,
        "dtype": args.dtype,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gen_max_tokens": args.gen_max_tokens,
        "enable_thinking": args.enable_thinking,
        "n_trajectories": len(records),
        "n_with_schemata": len(schema_usage) if analyzer is not None else 0,
    }
    (out_dir / f"config_method-{method}.json").write_text(json.dumps(config, indent=2))

    print(f"  wrote {out_path}  ({len(records)} rows, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
