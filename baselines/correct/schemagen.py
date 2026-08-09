"""CORRECT stage 1 — offline error-schema distillation, one call per trajectory.

For every annotated failed trajectory, an LLM distills an **error schema** from
the gold labels (mistake agent/step/reason + the conversation). The prompt is
copied **verbatim** from the vendored cloud generator
(``vendored/CORRECT/src/error_schema_generator_cloud.py::create_prompt``) — the
variant that produced the schemata behind the paper's Who&When results: history
serialized WITH step indices and a trailing format block that embeds the
trajectory's gold ``Agent Name`` / ``Step Number``. Retrieved schemata therefore
carry their source trajectory's gold labels into the detection prompt; that is
deliberate and part of the method.

Deliberate deviations (see README):
  * the agent key is fixed to ``role`` — the vendored autodetect ("name" if
    present in the first entry) targeted the *original* Who&When layout; this
    repo's algorithm-generated data swaps the fields (``role`` = agent name,
    ``name`` = user/assistant), so ``role`` reproduces what the vendored code
    yields on the original data;
  * schemata are stored as per-trajectory JSONs keyed by trajectory id
    (``{output}/schemagen/<id>.json``, resume for free) instead of one
    ``error_schemata.txt`` with fragile 1-based enumeration;
  * the ``schema`` field is ``strip_think``'d so local reasoning models work;
    the decisive ``raw`` is stored untouched.

Usage
-----
python -m baselines.correct.schemagen \
    --model ../hub/Qwen/Qwen3.5-9B --model-name qwen3.5-9b \
    --input data/ww/hand-crafted \
    --output outputs/ww/hand-crafted/qwen3.5-9b

Output: {output}/schemagen/<id>.json  (+ _run.json snapshot). File existence is
the resume ledger; ``--overwrite`` clears the directory.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from baselines.shared.common import _load_json_data
from baselines.shared.runner import OutputWriter, run_batched, run_streaming
from baselines.prompting.predict import build_backend, load_records, _bool

from .methods import Program, strip_think

SCHEMAGEN_SYSTEM = ("You are a helpful assistant skilled in analyzing conversations "
                    "and creating schemata for error detection.")

# The agent identity lives in the "role" field for every dataset in this repo.
AGENT_KEY = "role"


def load_schemagen_records(directory: str) -> list[dict]:
    """``load_records`` plus ``mistake_reason`` (the one field it drops).

    Same file walk and skips as the detection loader, so the schemagen id-set
    always equals the predict id-set.
    """
    records = load_records(directory)
    for r in records:
        data = _load_json_data(Path(directory) / r["filename"]) or {}
        r["mistake_reason"] = data.get("mistake_reason", "")
    return records


def build_schemagen_prompt(record: dict) -> str:
    """Verbatim ``error_schema_generator_cloud.py::create_prompt`` (agent key
    fixed to ``role``)."""
    chat_history = record["history"]
    question = record["question"]
    ground_truth = record["ground_truth"]
    mistake_agent = record["gold_agent"]
    mistake_step = record["gold_step"]
    mistake_reason = record["mistake_reason"]

    # Format chat history with step numbers
    chat_content = "\n".join([
        f"Step {idx}: {entry.get(AGENT_KEY, 'Unknown')}: {entry.get('content', '')}"
        for idx, entry in enumerate(chat_history)
    ])

    # Create focused prompt for error identification
    prompt_text = f"""Given an error analysis from a multi-agent conversation, create a error schema to help identify similar errors in the future.

Context:
Question: {question}
Ground Truth: {ground_truth}
Error Agent: {mistake_agent}
Error Step: {mistake_step}
Error Reason: {mistake_reason}

Conversation History:
{chat_content}

Based on this error case, please create a error schema that will help IDENTIFY similar errors in future conversations. Focus primarily on recognition patterns rather than mitigation strategies. The schema should include:

1. Error Signatures:
   - What distinctive patterns or signals indicate this type of error is occurring?
   - What are the telltale signs in the agent's behavior or responses?

2. Error Context Analysis:
   - What contextual conditions typically surround this type of error?
   - What sequence of interactions tends to precede this error?

3. Detection Heuristics:
   - What specific questions can be asked to determine if this error is present?
   - What analytical framework can help identify this error pattern?
   - What key phrases or conversation patterns serve as reliable indicators?

Please format your response as a structured schema that focuses specifically on ERROR IDENTIFICATION, not on how to improve agent behavior.

Provide a concise, actionable schema in the following format:

Agent Name: {mistake_agent}
Step Number: {mistake_step}
Reason for Mistake: [Your analysis of why this specific error occurred and how to identify similar patterns]
"""

    return prompt_text


def schemagen_messages(record: dict) -> list[dict]:
    # The vendored generator sends the prompt unscrubbed (generate_schema_gpt).
    return [
        {"role": "system", "content": SCHEMAGEN_SYSTEM},
        {"role": "user", "content": build_schemagen_prompt(record)},
    ]


def schemagen_program(record: dict) -> Program:
    raw = (yield [schemagen_messages(record)])[0]
    return {
        "predicted_agent": None,
        "predicted_step": None,
        "raw": raw,
        "schema": strip_think(raw).strip(),
        "calls": [{"response": raw}],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill CORRECT error schemata, one per trajectory.")
    p.add_argument("--model", required=True,
                   help="Local checkpoint path (vllm) or API model name (openai).")
    p.add_argument("--model-name", default=None,
                   help="Short label recorded in outputs (default: --model).")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True,
                   help="Model-level output directory; files land in {output}/schemagen/.")
    p.add_argument("--backend", default="vllm", choices=["vllm", "openai", "dummy"])
    # vLLM-only knobs (defaults = the vendored schema generator's SamplingParams).
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "auto"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
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
                   help="Clear {output}/schemagen/ and redo every trajectory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model_name or args.model

    writer = OutputWriter(Path(args.output) / "schemagen", overwrite=args.overwrite)
    done = writer.done_ids()

    records = load_schemagen_records(args.input)
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
        "method": "schemagen",
        "subset": Path(args.input).name,
        "backend": args.backend,
        "request_params": backend.request_params,
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
            "method": "schemagen",
            "model": model_name,
            "backend": args.backend,
            "predicted_agent": None,
            "predicted_step": None,
            "gold_agent": record["gold_agent"],
            "gold_step": record["gold_step"],
            "raw": pred["raw"],
            "schema": pred["schema"],
            "calls": pred["calls"],
        })

    programs = [(r, schemagen_program(r)) for r in remaining]

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
