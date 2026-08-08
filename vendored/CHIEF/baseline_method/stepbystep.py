import os
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI

MAX_TOKENS = 1024
USE_GROUND_TRUTH_IN_PROMPT = True

DEBUG_MODE = True
DEBUG_SAMPLE_LIMIT = 1

try:
    from dotenv import load_dotenv

    REPO_ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    REPO_ROOT = Path(__file__).resolve().parents[1]


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Please set it before running (or put it into repo-root/.env)."
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url and base_url.strip():
        return OpenAI(api_key=api_key.strip(), base_url=base_url.strip())

    return OpenAI(api_key=api_key.strip())


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_json_files_sorted(directory: Path):
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".json"]

    def key_fn(p: Path):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return int(digits) if digits else 0

    return sorted(files, key=key_fn)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def llm_chat(client: OpenAI, model: str, messages, max_tokens: int, temperature: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    msg = resp.choices[0].message
    content = getattr(msg, "content", None)

    if content is None or (isinstance(content, str) and content.strip() == ""):
        reasoning = getattr(msg, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            content = reasoning

    return (content or "").strip()


YES_RE = re.compile(r"^\s*(?:1\.)?\s*yes\b", re.IGNORECASE)
NO_RE  = re.compile(r"^\s*(?:1\.)?\s*no\b", re.IGNORECASE)


def parse_yes_no(text: str):
    if not text:
        return "unknown"
    t = text.strip()
    if YES_RE.search(t):
        return "yes"
    if NO_RE.search(t):
        return "no"
    return "unknown"


def extract_gt(data: dict):
    gt_agent = data.get("mistake_agent", None)
    gt_step = data.get("mistake_step", None)

    if (gt_agent is None or gt_step is None) and isinstance(data.get("gt"), dict):
        gt_agent = gt_agent if gt_agent is not None else data["gt"].get("agent")
        gt_step = gt_step if gt_step is not None else data["gt"].get("step")

    gt_answer = data.get("ground_truth", "") or data.get("answer", "") or ""
    return gt_agent, gt_step, gt_answer


def infer_dataset_and_agent_key(data_dir: Path):
    name = data_dir.name.lower()
    if "hand" in name or "crafted" in name:
        return "HC", "role"
    if "algorithm" in name or "generated" in name:
        return "AG", "name"
    return data_dir.name, "name"


def build_step_prompt(problem: str, ground_truth: str, convo_so_far: str, idx: int, agent_name: str):
    gt_part = f"The Answer for the problem is: {ground_truth}\n" if USE_GROUND_TRUTH_IN_PROMPT else ""
    return (
        "You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent "
        "conversation aimed at solving a real-world problem.\n"
        f"The problem being addressed is: {problem}\n"
        f"{gt_part}"
        f"Here is the conversation history up to the current step:\n{convo_so_far}\n"
        f"The most recent step ({idx}) was by '{agent_name}'.\n"
        f"Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could "
        "hinder the problem-solving process or lead to an incorrect solution.\n"
        "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment.\n"
        "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process.\n"
        "Respond ONLY in the format:\n"
        "1. Yes/No.\n"
        "2. Reason: [Your explanation here]\n"
    )


def run_step_by_step_on_file(client: OpenAI, model: str, data: dict, index_agent: str, temperature: float):
    history = data.get("history", [])
    problem = data.get("question", "")

    gt_agent, gt_step, gt_answer = extract_gt(data)

    per_step_logs = []
    convo_so_far = ""

    if not history:
        return None, 0, per_step_logs, gt_agent, gt_step, problem

    for idx, entry in enumerate(history):
        agent_name = entry.get(index_agent, "Unknown Agent")
        content = entry.get("content", "")
        convo_so_far += f"Step {idx} - {agent_name}: {content}\n"

        prompt = build_step_prompt(problem, gt_answer, convo_so_far, idx, agent_name)
        messages = [
            {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
            {"role": "user", "content": prompt},
        ]

        ans = llm_chat(client, model, messages, max_tokens=MAX_TOKENS, temperature=temperature)
        verdict = parse_yes_no(ans)

        per_step_logs.append({
            "step": idx,
            "agent": agent_name,
            "verdict": verdict,
            "raw": ans
        })

        if verdict == "yes":
            return agent_name, idx, per_step_logs, gt_agent, gt_step, problem

    last_idx = len(history) - 1
    last_agent = history[last_idx].get(index_agent, "Unknown Agent")
    return last_agent, last_idx, per_step_logs, gt_agent, gt_step, problem

def parse_args():
    default_data_dir = REPO_ROOT / "data" / "Algorithm-Generated"
    default_output_dir = REPO_ROOT / "results" / "stepbystep"

    p = argparse.ArgumentParser(description="Step-by-step (Who&When) baseline.")
    p.add_argument("--model", type=str, required=True, help="Model name for OpenAI(-compatible) Chat Completions.")
    p.add_argument("--data_dir", type=str, default=str(default_data_dir), help="Directory containing *.json samples.")
    p.add_argument("--output_dir", type=str, default=str(default_output_dir), help="Directory to save output json.")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default 0.0).")
    p.add_argument("--limit", type=int, default=None, help="Only run first N samples (debug).")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise RuntimeError(f"Directory not found: {data_dir}")

    dataset, index_agent = infer_dataset_and_agent_key(data_dir)

    client = build_client()

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    json_files = list_json_files_sorted(data_dir)
    if DEBUG_MODE:
        json_files = json_files[:DEBUG_SAMPLE_LIMIT]
    if args.limit is not None:
        json_files = json_files[: max(0, int(args.limit))]

    details = []
    correct_agent = 0
    correct_step = 0
    total = 0

    for fp in tqdm(json_files, desc=f"Step-by-step ({dataset}+withGT)"):
        try:
            data = load_json(fp)
        except Exception as e:
            details.append({"file": fp.name, "error": f"load_json failed: {e}"})
            continue

        pred_agent, pred_step, step_logs, gt_agent, gt_step, problem = run_step_by_step_on_file(
            client=client,
            model=args.model,
            data=data,
            index_agent=index_agent,
            temperature=args.temperature
        )

        total += 1
        acc_agent = int(gt_agent is not None and pred_agent == gt_agent)
        acc_step = int(gt_step is not None and pred_step == gt_step)
        correct_agent += acc_agent
        correct_step += acc_step

        details.append({
            "file": fp.name,
            "question": problem,
            "gt": {"agent": gt_agent, "step": gt_step},
            "pred": {"agent": pred_agent, "step": pred_step},
            "acc_agent": acc_agent,
            "acc_step": acc_step,
            "per_step": step_logs
        })

        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    ts = datetime.now().strftime("%m%d_%H%M")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model).strip("-") or "model"
    out_path = output_dir / f"results_stepbystep_{safe_model}_{dataset}_withGT_{ts}.json"

    summary = {
        "mode": "V8_STEP_BY_STEP",
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "directory_path": str(data_dir),
        "dataset": dataset,
        "setting": "withGT",
        "debug_mode": DEBUG_MODE,
        "debug_limit": DEBUG_SAMPLE_LIMIT if DEBUG_MODE else 0,
        "total": total,
        "agent_acc": (correct_agent / total) if total else 0.0,
        "step_acc": (correct_step / total) if total else 0.0,
        "details": details
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved summary to: {out_path}")
    print(f"Agent Acc: {summary['agent_acc']:.4f} | Step Acc: {summary['step_acc']:.4f} | Total: {total}")


if __name__ == "__main__":
    main()
