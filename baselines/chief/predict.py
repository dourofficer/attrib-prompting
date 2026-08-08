"""CHIEF baseline runner — one (model, subset) per invocation.

Loads every trajectory in a subset, runs the CHIEF hierarchical-causal-graph
attribution pipeline with local vLLM, and records one prediction row per
trajectory in the SAME schema as ``baselines.prompting.predict`` — so the existing
report machinery tabulates CHIEF next to SVD/CRR/prompting on identical splits.

Two execution modes (identical algorithm, shared builders/parsers):
  --mode batched      columnar: one batched generate per stage over all trajectories (default, fast)
  --mode per_sample   faithful reference: six sequential calls per trajectory (correctness checks)

Usage
-----
python -m baselines.chief.predict \
    --model  /data/hoang/resources/models/Qwen/Qwen3.5-9B \
    --input  data/ww/hand-crafted \
    --output outputs-ww/chief/qwen3.5-9b/hand-crafted \
    --rag on --rag-kb gaia,assistantbench --rag-root baselines/CHIEF/rag

Output
------
{output}/predictions_method-chief.jsonl   (one JSON object per trajectory)
{output}/config_method-chief.json         (run snapshot)
Idempotent: skips if the predictions file already exists unless --overwrite.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from baselines.prompting.predict import load_records

from . import pipeline, reference
from .engine import PromptEngine
from .rag import build_retriever, rag_texts_for

METHOD = "chief"


def _bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t", "on"}


def _kb_list(x: str) -> list[str]:
    return [k.strip() for k in str(x).split(",") if k.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the CHIEF attribution baseline with local vLLM.")
    p.add_argument("--model", required=True, help="HF model name or local path.")
    p.add_argument("--tokenizer", default=None, help="Optional tokenizer path override.")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True, help="Output directory.")
    p.add_argument("--mode", default="batched", choices=["batched", "per_sample"])
    # RAG
    p.add_argument("--rag", type=_bool, default=False, help="Enable stage-1 RAG retrieval.")
    p.add_argument("--rag-kb", default="gaia,assistantbench",
                   help="Comma list of KBs to use when --rag on (gaia,assistantbench).")
    p.add_argument("--rag-root", default="baselines/CHIEF/rag",
                   help="Directory holding index/ and kb/ for RAG.")
    p.add_argument("--rag-top-k", type=int, default=2)
    # vLLM knobs (mirror baselines.prompting.predict). CHIEF is greedy → temp 0.0.
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "auto"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--gen_max_tokens", type=int, default=2048)
    p.add_argument("--enable_thinking", type=_bool, default=False)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--truncate_prompt_tokens", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output)
    out_path = out_dir / f"predictions_method-{METHOD}.jsonl"
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

    # Precompute RAG blocks (CPU) before loading the LLM.
    kbs = _kb_list(args.rag_kb) if args.rag else []
    retriever = build_retriever(args.rag_root, kbs)
    rag_texts = rag_texts_for(retriever, records, top_k=args.rag_top_k)

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

    runner = pipeline.run if args.mode == "batched" else reference.run

    t0 = time.perf_counter()
    preds = runner(records, engine, rag_texts)
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
        "method": METHOD,
        "mode": args.mode,
        "subset": Path(args.input).name,
        "rag": bool(args.rag),
        "rag_kb": kbs,
        "dtype": args.dtype,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gen_max_tokens": args.gen_max_tokens,
        "enable_thinking": args.enable_thinking,
        "n_trajectories": len(records),
    }
    (out_dir / f"config_method-{METHOD}.json").write_text(json.dumps(config, indent=2))

    print(f"  wrote {out_path}  ({len(preds)} rows, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
