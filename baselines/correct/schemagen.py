"""Offline error-schema generation — one (model, subset) per invocation.

Port of ``baselines/CORRECT/src/error_schema_generator.py``: for every annotated
trajectory, an LLM distills the gold error (agent / step / reason) plus the full
conversation into a reusable "error schema"; all schemata for a subset are
written to one vendored-format ``error_schemata.txt`` (block per trajectory,
keyed by the trajectory's numeric filename — see ``retrieval.py``).

Prompt and system prompt are verbatim; sampling defaults match the vendored
``SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)``. Deviations
(see README): vLLM via :class:`PromptEngine` (one batched call instead of the
vendored print-batching loop), no YaRN rope hack (native long-context models),
``ground_truth`` key (vendored reads the raw CORRECT-Error ``groundtruth`` key;
this repo's normalized data stores ``ground_truth``), and ``strip_think`` on
each generated schema so reasoning traces never leak into future retrieval
prompts.

Usage
-----
python -m baselines.correct.schemagen \
    --model  ../hub/Qwen/Qwen3.5-9B \
    --input  data/ww/hand-crafted \
    --output outputs-ww/correct/schemata/qwen3.5-9b/hand-crafted

Output: {output}/error_schemata.txt (+ config.json snapshot).
Idempotent: skips if the schemata file already exists unless --overwrite.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.utils.common import _get_sorted_json_files, _load_json_data

from .engine import PromptEngine, strip_think
from .retrieval import write_schemata_file

SYSTEM_PROMPT = (
    "You are a helpful assistant skilled in analyzing conversations and creating "
    "schemata for error detection."
)


def build_schema_prompt(error_log: dict) -> str:
    """Verbatim vendored ``create_prompt`` (modulo the ``ground_truth`` key)."""
    chat_history = error_log.get("history", [])
    question = error_log.get("question", "")
    ground_truth = error_log.get("ground_truth", "")
    mistake_agent = error_log.get("mistake_agent", "")
    mistake_step = error_log.get("mistake_step", "")
    mistake_reason = error_log.get("mistake_reason", "")

    chat_content = "\n".join([
        f"{entry.get('role', 'Unknown')}: {entry.get('content', '')}"
        for entry in chat_history
    ])

    return f"""Given an error analysis from a multi-agent conversation, create a error schema to help identify similar errors in the future.

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
"""


def _bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CORRECT error schemata with VLLM.")
    p.add_argument("--model", required=True, help="HF model name or local path.")
    p.add_argument("--tokenizer", default=None,
                   help="Optional tokenizer path override (e.g. a corrected tokenizer dir).")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True, help="Output directory for error_schemata.txt.")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "auto"])
    p.add_argument("--seed", type=int, default=0)
    # Vendored schema-gen sampling: temperature=0.7, top_p=0.95, max_tokens=1024.
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--gen_max_tokens", type=int, default=1024)
    p.add_argument("--enable_thinking", type=_bool, default=False)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--truncate_prompt_tokens", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output)
    out_path = out_dir / "error_schemata.txt"
    if out_path.exists() and not args.overwrite:
        print(f"skip (exists): {out_path}")
        return

    # Numeric filename sort — same ordering the vendored generator and the
    # similarity stage use, so schema #n always describes trajectory n.json.
    file_nums, error_logs = [], []
    for fn in _get_sorted_json_files(args.input):
        data = _load_json_data(Path(args.input) / fn)
        if not data or "history" not in data:
            print(f"  skipping {fn}: unexpected format")
            continue
        file_nums.append(int(Path(fn).stem))
        error_logs.append(data)
    print(f"  {len(error_logs)} trajectories from {args.input}")
    if not error_logs:
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

    message_lists = [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": build_schema_prompt(log)}]
        for log in error_logs
    ]

    t0 = time.perf_counter()
    outputs = engine.generate(message_lists)
    elapsed = time.perf_counter() - t0

    # strip_think: a schema with an embedded reasoning trace would pollute every
    # future retrieval prompt that cites it.
    schemata = {num: strip_think(raw) for num, raw in zip(file_nums, outputs)}

    out_dir.mkdir(parents=True, exist_ok=True)
    write_schemata_file(schemata, out_path)

    config = {
        "model": args.model,
        "subset": Path(args.input).name,
        "dtype": args.dtype,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gen_max_tokens": args.gen_max_tokens,
        "enable_thinking": args.enable_thinking,
        "n_trajectories": len(error_logs),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"  wrote {out_path}  ({len(schemata)} schemata, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
