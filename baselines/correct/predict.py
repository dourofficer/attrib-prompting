"""CORRECT detection runner — one (model, subset, method) per invocation.

Runs schema-guided (``correct``) or k=0 baseline (``correct_baseline``)
all-at-once detection over *all* trajectories in a subset and writes one JSON
file per trajectory under ``{output}/{method}/``. ``correct`` needs the stage-1
schemata and stage-2 similarities (see ``schemagen.py`` / ``similarity.py``,
or run everything via ``sweep.py``); ``correct_baseline`` needs neither.

GT setting (GUIDE.md "GT settings"): for this baseline the vendored cloud path
never puts the task answer in the prompt, so ``--gt without`` is the
parity-tested paper setting **and the default** (the inverse of prompting).
``--gt with`` inserts the vendored answer line. Point ``--output`` at the tree
matching the setting (``outputs/`` for with, ``outputs-nogt/`` for without —
the sweep does this mapping automatically).

Resume: an existing ``{output}/{method}/<id>.json`` marks that trajectory done;
rerun the same command and only missing ids execute (``--overwrite`` clears the
method directory instead).

Usage
-----
python -m baselines.correct.predict \
    --model ../hub/Qwen/Qwen3.5-9B --model-name qwen3.5-9b \
    --input data/ww/hand-crafted \
    --output outputs-nogt/ww/hand-crafted/qwen3.5-9b \
    --method correct --num-schemata 10 --schema-model qwen3.5-9b \
    --schemata-dir outputs/ww/hand-crafted/qwen3.5-9b/schemagen \
    --similarities outputs/ww/hand-crafted/_similarities/bge-m3.json
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from baselines.shared.runner import OutputWriter, run_batched, run_streaming
from baselines.prompting.predict import build_backend, load_records, _bool

from .methods import METHODS, correct_baseline_program, correct_program
from .retrieval import SchemaAnalyzer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CORRECT schema-guided detection.")
    p.add_argument("--model", required=True,
                   help="Local checkpoint path (vllm) or API model name (openai).")
    p.add_argument("--model-name", default=None,
                   help="Short label recorded in outputs (default: --model).")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True,
                   help="Model-level output directory; files land in {output}/{method}/.")
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--method-dir", default=None,
                   help="Output directory name override (default: --method). For "
                        "side-by-side k experiments, e.g. correct.k3.")
    p.add_argument("--backend", default="vllm", choices=["vllm", "openai", "dummy"])
    p.add_argument("--gt", default="without", choices=["with", "without"],
                   help="'with' inserts the vendored 'The Answer for the problem is:' "
                        "line. Default 'without' — the vendored cloud path (and the "
                        "paper) never includes the answer.")
    # Retrieval artifacts (required for --method correct)
    p.add_argument("--schemata-dir", default=None,
                   help="Stage-1 output dir: .../<schema_model>/schemagen")
    p.add_argument("--similarities", default=None,
                   help="Stage-2 ranked-neighbour JSON: .../_similarities/bge-m3.json")
    p.add_argument("--num-schemata", type=int, default=1,
                   help="Top-k retrieved schemata (vendored default 1; paper: "
                        "alg-generated 1, hand-crafted 10, correct-error 5).")
    p.add_argument("--schema-model", default=None,
                   help="Label of the model that generated the schemata (recorded in outputs).")
    # vLLM-only knobs (defaults: greedy, as the vendored local CORRECT inference)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "auto"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--gen_max_tokens", type=int, default=1024)
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


def _build_analyzer(args: argparse.Namespace) -> SchemaAnalyzer:
    missing = []
    if not args.schemata_dir or not Path(args.schemata_dir).is_dir():
        missing.append(
            f"schemata dir {args.schemata_dir!r} — generate with:\n"
            f"    python -m baselines.correct.schemagen --input {args.input} "
            f"--output <...>/<schema_model> [--model ...]")
    if not args.similarities or not Path(args.similarities).is_file():
        missing.append(
            f"similarities file {args.similarities!r} — generate with:\n"
            f"    python -m baselines.correct.similarity --input {args.input} "
            f"--output <...>/_similarities/bge-m3.json")
    if missing:
        raise SystemExit("--method correct needs precomputed artifacts; missing:\n"
                         + "\n".join(missing)
                         + "\n(or run the full pipeline: python -m baselines.correct.sweep)")
    analyzer = SchemaAnalyzer.from_paths(args.schemata_dir, args.similarities)
    if not analyzer.schemata:
        raise SystemExit(f"no schemata found under {args.schemata_dir}")
    return analyzer


def main() -> None:
    args = parse_args()
    model_name = args.model_name or args.model
    method_dir = args.method_dir or args.method
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

    analyzer = None
    if args.method == "correct":
        analyzer = _build_analyzer(args)
        n_missing = sum(1 for r in remaining
                        if int(r["id"]) not in analyzer.similarities)
        if n_missing:
            print(f"  warning: {n_missing} trajectories have no similarity entry "
                  "(empty retrieval → vendored fallback prompt)")

    backend = build_backend(args)
    writer.write_run_config({
        "model": model_name,
        "model_arg": args.model,
        "method": args.method,
        "method_dir": method_dir,
        "subset": Path(args.input).name,
        "backend": args.backend,
        "request_params": backend.request_params,
        "gt_in_prompt": include_gt,
        "num_schemata": args.num_schemata if args.method == "correct" else 0,
        "schema_model": args.schema_model if args.method == "correct" else None,
        "schemata_dir": args.schemata_dir if args.method == "correct" else None,
        "similarities": args.similarities if args.method == "correct" else None,
        "n_with_schemata": len(analyzer.schemata) if analyzer else None,
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
            "method": args.method,
            "model": model_name,
            "backend": args.backend,
            "gt_in_prompt": include_gt,
            "schema_model": args.schema_model if args.method == "correct" else None,
            "num_schemata": pred["num_schemata"],
            "schema_cases": pred["schema_cases"],
            "predicted_agent": pred["predicted_agent"],
            "predicted_step": pred["predicted_step"],
            "gold_agent": record["gold_agent"],
            "gold_step": record["gold_step"],
            "raw": pred["raw"],
            "calls": pred["calls"],
        })

    if args.method == "correct":
        programs = [(r, correct_program(r, analyzer=analyzer,
                                        num_schemata=args.num_schemata,
                                        include_gt=include_gt))
                    for r in remaining]
    else:
        programs = [(r, correct_baseline_program(r, include_gt=include_gt))
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
