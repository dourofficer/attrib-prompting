"""CHIEF detection runner — one (model, subset) per invocation.

Runs the six-stage hierarchical-causal-graph attribution over every trajectory in
a subset and writes one JSON file per trajectory under ``{output}/chief/``, each
carrying the full six-call transcript in ``calls``. The stage-1 exemplar blocks
come from the precomputed artifact (see :mod:`baselines.chief.ragprep`), so this
runner performs no retrieval and needs neither faiss nor sentence-transformers.

GT setting (GUIDE.md "GT settings"): every vendored stage prompt carries
``The correct answer for the problem is: ...``, so ``--gt with`` is the
parity-tested vendored setting **and the default** (as for prompting; the inverse
of correct). ``--gt without`` drops that sentence from all six prompts. Point
``--output`` at the tree matching the setting (``outputs/`` for with,
``outputs-nogt/`` for without — the sweep does this mapping automatically).

Resume: an existing ``{output}/chief/<id>.json`` marks that trajectory done;
rerun the same command and only missing ids execute (``--overwrite`` clears the
method directory instead). A trajectory whose calls exhaust their retries is
skipped without writing, so a rerun picks it up.

Usage
-----
python -m baselines.chief.predict \\
    --model gpt-4o --model-name gpt-4o --backend openai \\
    --input  data/ww/hand-crafted \\
    --output outputs/ww/hand-crafted/gpt-4o \\
    --rag-texts artifacts/ww/hand-crafted/rag/all-MiniLM-L6-v2.json
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from baselines.shared.runner import OutputWriter, run_batched, run_streaming
from baselines.prompting.predict import build_backend, load_records, _bool

from .methods import METHOD, chief_program


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the CHIEF attribution baseline.")
    p.add_argument("--model", required=True,
                   help="Local checkpoint path (vllm) or API model name (openai).")
    p.add_argument("--model-name", default=None,
                   help="Short label recorded in outputs (default: --model).")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True,
                   help=f"Model-level output directory; files land in {{output}}/{METHOD}/.")
    p.add_argument("--method-dir", default=None,
                   help=f"Output directory name override (default: {METHOD}).")
    p.add_argument("--backend", default="vllm", choices=["vllm", "openai", "dummy"])
    p.add_argument("--gt", default="with", choices=["with", "without"],
                   help="'with' keeps the vendored 'The correct answer for the problem "
                        "is:' sentence in all six prompts (default — the vendored "
                        "bytes). 'without' drops it.")
    p.add_argument("--step-hint", default="on", choices=["on", "off"],
                   help="'on' (default) adds one sentence to each stage prompt "
                        "stating that steps are indexed from 0. 'off' restores the "
                        "vendored bytes, which leave the convention implicit and "
                        "make models answer 1-based. See IMPLEMENTATION.md.")
    p.add_argument("--rag-texts", default=None,
                   help="Stage-1 exemplar artifact: "
                        "artifacts/<ds>/<subset>/rag/<embed_model>.json. Omit to run "
                        "without retrieved examples (a documented prompt deviation).")
    # vLLM-only knobs. CHIEF is greedy — the vendored call_model pins temperature 0.
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


def _load_rag_texts(path: str | None) -> dict[str, str]:
    """Read the precomputed exemplar blocks; ``{}`` means no retrieval section."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"missing stage-1 RAG artifact {path!r} — generate it with:\n"
            f"    python -m baselines.chief.ragprep --input <subset> --output {path}\n"
            "(or run the full pipeline: python -m baselines.chief.sweep, or pass no "
            "--rag-texts to prompt without retrieved examples)")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    model_name = args.model_name or args.model
    method_dir = args.method_dir or METHOD
    include_gt = args.gt == "with"
    include_step_hint = args.step_hint == "on"

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

    rag_texts = _load_rag_texts(args.rag_texts)
    if rag_texts:
        n_missing = sum(1 for r in remaining if r["id"] not in rag_texts)
        if n_missing:
            print(f"  warning: {n_missing} trajectories have no RAG entry "
                  "(stage 1 omits the retrieved-example section for them)")

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
        "step_hint_in_prompt": include_step_hint,
        "rag_texts": args.rag_texts,
        "n_with_rag": len(rag_texts),
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
            "rag_in_prompt": record["id"] in rag_texts,
            "predicted_agent": pred["predicted_agent"],
            "predicted_step": pred["predicted_step"],
            "gold_agent": record["gold_agent"],
            "gold_step": record["gold_step"],
            "raw": pred["raw"],
            "calls": pred["calls"],
        })

    programs = [(r, chief_program(r, rag_text=rag_texts.get(r["id"]),
                                  include_gt=include_gt,
                                  include_step_hint=include_step_hint))
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
