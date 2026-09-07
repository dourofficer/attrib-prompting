"""ErrorProbe runner — one (model, subset, mode) per invocation.

Runs the Analyzer→Verifier diagnosis over every trajectory in a subset and
writes one JSON file per trajectory under ``{output}/<method dir>/``, each
carrying the full call transcript in ``calls``. ``--mode truncated`` (the
vendored default configuration) writes under ``errorprobe/``; ``--mode
backward`` (the vendored code's optional backward-tracing configuration)
writes under ``errorprobe_bt/``; ``--mode paper`` (the paper's own pipeline,
built from the paper since the vendored code omits it: MAST tagger, dependency
graph, Strategist/Investigator/Arbiter) writes under ``errorprobe_paper/``.

GT setting (GUIDE.md "GT settings"): the vendored prompts never carry the task
answer, so ``--gt without`` is the vendored-faithful setting **and the
default** (as for correct and raffles; the inverse of prompting and chief).
``--gt with`` appends the answer line to the task text in every prompt that
carries the task. Point ``--output`` at the tree matching the setting
(``outputs-nogt/`` for without, ``outputs/`` for with — the sweep does this
mapping automatically).

Resume: an existing ``{output}/<method dir>/<id>.json`` marks that trajectory
done; rerun the same command and only missing ids execute (``--overwrite``
clears the method directory instead). A trajectory whose calls exhaust their
retries is skipped without writing, so a rerun picks it up.

Usage
-----
python -m baselines.errorprobe.predict \\
    --model gpt-4o --model-name gpt-4o --backend openai \\
    --input  data/ww/hand-crafted \\
    --output outputs-nogt/ww/hand-crafted/gpt-4o
"""
from __future__ import annotations

import argparse
import functools
import time
from datetime import datetime, timezone
from pathlib import Path

from baselines.shared.runner import OutputWriter, run_batched, run_streaming
from baselines.prompting.predict import build_backend, load_records, _bool

from .methods import METHOD, METHOD_BT, errorprobe_bt_program, errorprobe_program
from .paper.program import METHOD_PAPER, PAPER_DEFAULTS, errorprobe_paper_program

MODES = {"truncated": (METHOD, errorprobe_program),
         "backward": (METHOD_BT, errorprobe_bt_program),
         "paper": (METHOD_PAPER, errorprobe_paper_program)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the ErrorProbe attribution baseline.")
    p.add_argument("--model", required=True,
                   help="Local checkpoint path (vllm) or API model name (openai).")
    p.add_argument("--model-name", default=None,
                   help="Short label recorded in outputs (default: --model).")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True,
                   help="Model-level output directory; files land in "
                        "{output}/errorprobe/, {output}/errorprobe_bt/ or "
                        "{output}/errorprobe_paper/.")
    p.add_argument("--mode", default="truncated", choices=sorted(MODES),
                   help="'truncated' is the vendored default configuration "
                        "(Analyzer over the last 15 turns); 'backward' is its "
                        "optional backward-tracing configuration (full trace, "
                        "many calls per trajectory); 'paper' is the paper's "
                        "pipeline (tagger, dependency graph, diagnosis team; "
                        "about 2 calls per 10 steps plus 5).")
    p.add_argument("--method-dir", default=None,
                   help="Output directory name override (default: the mode's "
                        f"method name, {METHOD}, {METHOD_BT} or {METHOD_PAPER}).")
    # Paper-mode knobs (ignored by the other modes).
    p.add_argument("--max-hypotheses", type=int, default=PAPER_DEFAULTS["max_hypotheses"],
                   help="paper mode: hypotheses the Strategist may propose, one "
                        "Investigator call each.")
    p.add_argument("--chunk-chars", type=int, default=PAPER_DEFAULTS["chunk_chars"],
                   help="paper mode: rendered characters per tagger/dependency chunk.")
    p.add_argument("--condensed-chars", type=int, default=PAPER_DEFAULTS["condensed_chars"],
                   help="paper mode: budget for the condensed trace the team reads.")
    p.add_argument("--include-mast-examples", type=_bool,
                   default=PAPER_DEFAULTS["include_mast_examples"],
                   help="paper mode: add the MAST authors' worked examples (~17k "
                        "tokens) to every tagger prompt.")
    p.add_argument("--sequential-edges", default=PAPER_DEFAULTS["sequential_edges"],
                   choices=["fallback", "always"],
                   help="paper mode: 'fallback' adds turn-to-turn edges only when "
                        "the dependency graph leaves too few steps; 'always' adds "
                        "them everywhere (no masking ever happens).")
    p.add_argument("--backend", default="vllm", choices=["vllm", "openai", "dummy"])
    p.add_argument("--gt", default="without", choices=["with", "without"],
                   help="'without' matches the vendored prompts, which never "
                        "carry the task answer (default). 'with' appends the "
                        "answer line to the task text in every prompt.")
    # vLLM-only knobs. The vendored config decodes at temperature 0.7.
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "auto"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
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
    method, program = MODES[args.mode]
    method_dir = args.method_dir or method
    include_gt = args.gt == "with"
    paper_params = None
    if args.mode == "paper":
        paper_params = {"max_hypotheses": args.max_hypotheses,
                        "chunk_chars": args.chunk_chars,
                        "condensed_chars": args.condensed_chars,
                        "include_mast_examples": args.include_mast_examples,
                        "sequential_edges": args.sequential_edges}
        program = functools.partial(program, **paper_params)

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
        "method": method,
        "method_dir": method_dir,
        "mode": args.mode,
        "subset": Path(args.input).name,
        "backend": args.backend,
        "request_params": backend.request_params,
        "gt_in_prompt": include_gt,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_total": len(records),
        "n_already_done": len(done),
        "n_remaining": len(remaining),
        "resumed": bool(done),
        **({"paper_params": paper_params} if paper_params else {}),
    })

    def on_done(record: dict, pred: dict) -> None:
        writer.write(record["id"], {
            "id": record["id"],
            "filename": record["filename"],
            "question_id": record["question_id"],
            "method": method,
            "mode": args.mode,
            "model": model_name,
            "backend": args.backend,
            "gt_in_prompt": include_gt,
            "predicted_agent": pred["predicted_agent"],
            "predicted_step": pred["predicted_step"],
            "gold_agent": record["gold_agent"],
            "gold_step": record["gold_step"],
            "raw": pred["raw"],
            "calls": pred["calls"],
            **{k: v for k, v in pred.items()
               if k not in ("predicted_agent", "predicted_step", "raw", "calls")},
        })

    programs = [(r, program(r, include_gt=include_gt)) for r in remaining]

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
