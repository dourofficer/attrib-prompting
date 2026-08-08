# filename: baseline.py
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from openai import OpenAI

try:
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
except Exception:
    pass


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_agent(x: Any) -> Optional[str]:
    """Normalize agent name."""
    return re.sub(r"\s+", " ", str(x)).strip().lower() if x else None


def normalize_step(x: Any) -> Optional[int]:
    """Normalize step index."""
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


# ---------- OpenAI client ----------
def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Please set it before running:\n"
            "  export OPENAI_API_KEY=your_api_key"
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url and base_url.strip():
        return OpenAI(api_key=api_key.strip(), base_url=base_url.strip())

    return OpenAI(api_key=api_key.strip())


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.0,
) -> str:
    """Unified wrapper for OpenAI(-compatible) Chat Completions."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant skilled in analyzing multi-agent conversations.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


# ---------- Baseline logic ----------
def baseline_predict_error(
    client: OpenAI,
    model: str,
    history_text: List[Dict[str, Any]],
    question: str,
    ground_truth: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Provide the full trajectory + ground truth answer, and ask the model to locate the first mistake."""
    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation that failed to correctly solve a real-world problem.\n"
        f"The problem is: {question}\n"
        f"The correct answer for the problem should be: {ground_truth}\n\n"
        "Here is the full conversation log (in JSON format):\n"
        + str(history_text)
        + f"\n\nThere are total {len(history_text)} steps, each representing one agent's message.\n"
        "Your goal:\n"
        "1. Identify which agent made the reasoning mistake directly responsible for the incorrect outcome.\n"
        "2. Determine the exact step number where this mistake first occurred.\n"
        "3. Explain why this step represents a reasoning error.\n\n"
        "Please strictly follow this plain text format:\n"
        "Agent Name: (your prediction)\n"
        "Step Number: (your prediction)\n"
        "Reason for Mistake: (your explanation)\n"
        "No extra commentary or markdown symbols.\n"
    )

    result_text = call_llm(client, model, prompt, temperature=temperature).strip()

    agent_pattern = r"Agent Name:\s*([^\n*]+)"
    step_pattern = r"Step Number:\s*([0-9]+)"
    reason_pattern = r"Reason for Mistake:\s*(.*)"

    agent_m = re.search(agent_pattern, result_text)
    step_m = re.search(step_pattern, result_text)
    reason_m = re.search(reason_pattern, result_text, re.DOTALL)

    agent_name = agent_m.group(1).strip() if agent_m else None
    step_number = int(step_m.group(1)) if step_m else None
    reason_text = reason_m.group(1).strip() if reason_m else ""

    return {
        "final": {
            "mistake_agent": agent_name,
            "mistake_step": step_number,
            "reason": reason_text,
        },
        "raw": result_text,
    }


def process_sample(
    client: OpenAI,
    model: str,
    file_path: Path,
    idx: int,
    temperature: float = 0.0,
) -> Tuple[Dict[str, Any], int, int]:
    """Process a single sample file, return (result_dict, agent_acc, step_acc)."""
    data = load_json(file_path)
    history_text = data["history"]
    question = data.get("question", "")
    gt_agent = normalize_agent(data.get("mistake_agent"))
    gt_step = normalize_step(data.get("mistake_step"))
    ground_truth = data.get("ground_truth", "")

    error_pred = baseline_predict_error(
        client=client,
        model=model,
        history_text=history_text,
        question=question,
        ground_truth=ground_truth,
        temperature=temperature,
    )

    pred_agent = normalize_agent(error_pred["final"]["mistake_agent"])
    pred_step = normalize_step(error_pred["final"]["mistake_step"])

    acc_agent = 1 if (pred_agent and gt_agent and pred_agent == gt_agent) else 0
    acc_step = 1 if (pred_step is not None and gt_step is not None and pred_step == gt_step) else 0

    print("=" * 80)
    print(f"样本 {idx + 1}: {file_path.name}")
    print(f"问题: {question}")
    print(f"→ 预测: agent='{pred_agent}' | step={pred_step}")
    print(f"→ 标准: agent='{gt_agent}' | step={gt_step}")
    print(f"→ 准确性: agent={'✅' if acc_agent else '❌'} | step={'✅' if acc_step else '❌'}")

    return (
        {
            "file": str(file_path),
            "question": question,
            "gt": {"agent": gt_agent, "step": gt_step},
            "pred": {"agent": pred_agent, "step": pred_step},
            "acc_agent": acc_agent,
            "acc_step": acc_step,
            "step1_raw": None,
            "step2_raw": error_pred["raw"],
            "final_pred": error_pred["final"],
        },
        acc_agent,
        acc_step,
    )


def infer_dataset_tag(data_dir: Path) -> str:
    """Infer dataset tag from directory name (only for output naming)."""
    name = data_dir.name.lower()
    if "algorithm" in name or "generated" in name:
        return "AG"
    if "hand" in name or "crafted" in name:
        return "HC"
    return data_dir.name


def parse_args() -> argparse.Namespace:
    # <repo-root>/CHIEF/baseline.py -> <repo-root>
    repo_root = Path(__file__).resolve().parents[1]
    default_data_dir = repo_root / "data" / "Algorithm-Generated"
    default_output_dir = repo_root / "results" / "baseline"

    p = argparse.ArgumentParser(
        description="Who&When Baseline: predict the first mistake agent+step from a full conversation trajectory."
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for OpenAI(-compatible) Chat Completions (e.g., gpt-4o-mini).",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(default_data_dir),
        help="Directory containing *.json samples (e.g., data/Algorithm-Generated or data/Hand-Crafted).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(default_output_dir),
        help="Directory to save the result json.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model (default: 0.0).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only run the first N samples (for debugging).",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional: sleep seconds between requests to avoid rate limits.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    client = build_client()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir not found or not a directory: {data_dir}")

    json_files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() == ".json"])
    if args.limit is not None:
        json_files = json_files[: max(0, int(args.limit))]
        print(f"🧪 Limit mode: Running only {len(json_files)} samples.")

    results: List[Dict[str, Any]] = []
    acc_a = 0
    acc_s = 0

    for i, file_path in enumerate(tqdm(json_files, desc="Running Baseline Mode")):
        try:
            res, aa, ss = process_sample(
                client=client,
                model=args.model,
                file_path=file_path,
                idx=i,
                temperature=args.temperature,
            )
            results.append(res)
            acc_a += aa
            acc_s += ss
        except Exception as e:
            print(f"[❌ Error] {file_path}: {e}")

        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    total = len(json_files)
    dataset_tag = infer_dataset_tag(data_dir)

    summary = {
        "mode": "baseline",
        "model": args.model,
        "dataset": dataset_tag,
        "data_dir": str(data_dir),
        "total": total,
        "agent_acc": acc_a / total if total else 0,
        "step_acc": acc_s / total if total else 0,
        "details": results,
    }

    ts = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model).strip("-") or "model"
    out_path = output_dir / f"results_baseline_{safe_model}_{dataset_tag}_{ts}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Mode: Baseline")
    print(f"✅ Model: {args.model}")
    print(f"✅ Dataset: {dataset_tag}")
    print(f"✅ Sample Count: {total}")
    print(f"Agent Accuracy: {summary['agent_acc'] * 100:.2f}%")
    print(f"Step Accuracy: {summary['step_acc'] * 100:.2f}%")
    print(f"📂 Output file: {out_path}")
    print(f"Total time: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
