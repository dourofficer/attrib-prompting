"""RAFFLES runner — one (model, subset) per invocation.

Runs the Judge-Evaluator loop over every trajectory in a subset and writes one
JSON file per trajectory under ``{output}/raffles/``, each carrying the full
call transcript in ``calls`` and a per-iteration audit in ``iterations``.

GT setting (GUIDE.md "GT settings"): the paper evaluates Who&When *without*
ground truth, so ``--gt without`` is the paper-faithful setting **and the
default** (as for correct; the inverse of prompting and chief). ``--gt with``
adds the answer line to the problem statement in every prompt. Point
``--output`` at the tree matching the setting (``outputs-nogt/`` for without,
``outputs/`` for with — the sweep does this mapping automatically).

Resume: an existing ``{output}/raffles/<id>.json`` marks that trajectory done;
rerun the same command and only missing ids execute (``--overwrite`` clears the
method directory instead). A trajectory whose calls exhaust their retries is
skipped without writing, so a rerun picks it up.

Usage
-----
python -m baselines.raffles.predict \\
    --model gpt-4o --model-name gpt-4o --backend openai \\
    --input  data/ww/hand-crafted \\
    --output outputs-nogt/ww/hand-crafted/gpt-4o
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from baselines.shared.runner import OutputWriter, run_batched, run_streaming
from baselines.prompting.predict import build_backend, load_records, _bool

from .methods import MAX_ITERS, METHOD, THRESHOLD, raffles_program


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the RAFFLES attribution baseline.")
    p.add_argument("--model", required=True,
                   help="Local checkpoint path (vllm) or API model name (openai).")
    p.add_argument("--model-name", default=None,
                   help="Short label recorded in outputs (default: --model).")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True,
                   help=f"Model-level output directory; files land in {{output}}/{METHOD}/.")
    p.add_argument("--method-dir", default=None,
                   help=f"Output directory name override (default: {METHOD}; "
                        "e.g. raffles.k5 for a side-by-side K=5 run).")
    p.add_argument("--backend", default="vllm", choices=["vllm", "openai", "dummy"])
    p.add_argument("--gt", default="without", choices=["with", "without"],
                   help="'without' matches the paper, which evaluates Who&When "
                        "without ground truth (default). 'with' adds the answer "
                        "line to the problem statement in every prompt.")
    p.add_argument("--max-iters", type=int, default=MAX_ITERS,
                   help=f"K, the extra Judge-Evaluator iterations after the first "
                        f"pass (default {MAX_ITERS}, the paper's main setting; "
                        "0 still runs one full pass).")
    p.add_argument("--threshold", type=int, default=THRESHOLD,
                   help=f"Stop once the summed confidence exceeds this "
                        f"(default {THRESHOLD} of 400, the paper's value).")
    # vLLM-only knobs. The paper decodes greedily, so temperature defaults to 0.
    p.add_argument("--tokenizer", default=None)
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
    # API-only knobs
    p.add_argument("--api-base-url", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--api-header", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--api-concurrency", type=int, default=8)
    p.add_argument("--api-max-retries", type=int, default=6)
    p.add_argument("--api-param", action="append", default=[], metavar="KEY=VALUE",
                   help="Request parameter sent verbatim (repeatable). These are the "
                        "ONLY generation params an API model receives.")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="Clear the method output directory and redo every trajectory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model_name or args.model
    method_dir = args.method_dir or METHOD
    include_gt = args.gt == "with"

    writer = OutputWriter(Path(args.output) / method_dir, overwrite=args.overwrite)
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
    writer.write_run_config({
        "model": model_name,
        "model_arg": args.model,
        "method": METHOD,
        "method_dir": method_dir,
        "subset": Path(args.input).name,
        "backend": args.backend,
        "request_params": backend.request_params,
        "gt_in_prompt": include_gt,
        "max_iters": args.max_iters,
        "threshold": args.threshold,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_total": len(records),
        "n_already_done": len(done),
        "n_remaining": len(remaining),
        "resumed": bool(done),
    })

    def on_done(record: dict, pred: dict) -> None:
        writer.write(record["id"], {
            "id": record["id"],
            "filename": record["filename"],
            "question_id": record["question_id"],
            "method": METHOD,
            "model": model_name,
            "backend": args.backend,
            "gt_in_prompt": include_gt,
            "max_iters": args.max_iters,
            "threshold": args.threshold,
            "predicted_agent": pred["predicted_agent"],
            "predicted_step": pred["predicted_step"],
            "gold_agent": record["gold_agent"],
            "gold_step": record["gold_step"],
            "raw": pred["raw"],
            "confidence": pred["confidence"],
            "n_iterations": pred["n_iterations"],
            "iterations": pred["iterations"],
            "calls": pred["calls"],
        })

    programs = [(r, raffles_program(r, max_iters=args.max_iters,
                                    threshold=args.threshold,
                                    include_gt=include_gt))
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
