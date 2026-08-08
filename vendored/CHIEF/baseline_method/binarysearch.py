import os
import json
import random
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI


MAX_TOKENS = 1024
RANDOM_SEED = 2025

DEBUG_MODE = False
DEBUG_SAMPLE_LIMIT = 1

USE_GROUND_TRUTH_IN_PROMPT = True


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
    # fallback
    return data_dir.name, "name"


def build_segment_text(history, start: int, end: int, index_agent: str):
    seg = history[start:end + 1]
    lines = []
    for entry in seg:
        agent_name = entry.get(index_agent, "Unknown Agent")
        content = entry.get("content", "")
        lines.append(f"{agent_name}: {content}")
    return "\n".join(lines)


def construct_binary_search_prompt(problem: str,
                                  answer: str,
                                  chat_segment_content: str,
                                  range_description: str,
                                  upper_half_desc: str,
                                  lower_half_desc: str):
    gt_part = f"The Answer for the problem is: {answer}\n" if USE_GROUND_TRUTH_IN_PROMPT else ""
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are "
        "collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. "
        "Determine which half of the segment contains the single step where this crucial error occurs, ultimately "
        "leading to the failure in resolving the user’s query.\n"
        f"The problem to address is as follows: {problem}\n"
        f"{gt_part}"
        f"Review the following conversation segment {range_description}:\n\n{chat_segment_content}\n\n"
        f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half "
        f"({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
        "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer "
        "should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. "
        "If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing "
        "for subjective judgment, and base your answer on that."
    )


def parse_half(text: str):
    if not text:
        return "unknown"
    t = text.strip().lower()
    if "upper half" in t:
        return "upper"
    if "lower half" in t:
        return "lower"
    return "unknown"


def run_binary_search_on_file(client: OpenAI, model: str, data: dict, index_agent: str, temperature: float):
    history = data.get("history", [])
    problem = data.get("question", "")

    gt_agent, gt_step, gt_answer = extract_gt(data)

    if not history:
        return None, 0, [], gt_agent, gt_step, problem

    rounds = []
    start, end = 0, len(history) - 1

    while True:
        if start > end:
            chosen = max(0, min(end, len(history) - 1))
            agent = history[chosen].get(index_agent, "Unknown Agent")
            rounds.append({
                "range": [start, end],
                "mid": None,
                "decision": "invalid_range_fallback",
                "raw": "",
            })
            return agent, chosen, rounds, gt_agent, gt_step, problem

        if start == end:
            agent = history[start].get(index_agent, "Unknown Agent")
            rounds.append({
                "range": [start, end],
                "mid": start,
                "decision": "done",
                "raw": "",
            })
            return agent, start, rounds, gt_agent, gt_step, problem

        mid = start + (end - start) // 2

        range_desc = f"from step {start} to step {end}"
        upper_desc = f"from step {start} to step {mid}"
        lower_desc = f"from step {mid + 1} to step {end}"

        chat_segment_content = build_segment_text(history, start, end, index_agent)
        prompt = construct_binary_search_prompt(
            problem=problem,
            answer=gt_answer,
            chat_segment_content=chat_segment_content,
            range_description=range_desc,
            upper_half_desc=upper_desc,
            lower_half_desc=lower_desc
        )

        messages = [
            {"role": "system",
             "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
            {"role": "user", "content": prompt},
        ]

        raw = llm_chat(client, model, messages, max_tokens=MAX_TOKENS, temperature=temperature)
        decision = parse_half(raw)

        rounds.append({
            "range": [start, end],
            "mid": mid,
            "upper": [start, mid],
            "lower": [mid + 1, end],
            "decision": decision,
            "raw": raw
        })

        if decision == "upper":
            end = mid
        elif decision == "lower":
            start = min(mid + 1, end)
        else:
            if random.randint(0, 1) == 0:
                rounds[-1]["decision"] = "upper_random"
                end = mid
            else:
                rounds[-1]["decision"] = "lower_random"
                start = min(mid + 1, end)


def parse_args():
    default_data_dir = REPO_ROOT / "data" / "Algorithm-Generated"
    default_output_dir = REPO_ROOT / "results" / "binarysearch"

    p = argparse.ArgumentParser(description="Binary Search (Who&When) baseline.")
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
    random.seed(RANDOM_SEED)

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

    for fp in tqdm(json_files, desc=f"Binary Search ({dataset}+withGT)"):
        try:
            data = load_json(fp)
        except Exception as e:
            details.append({"file": fp.name, "error": f"load_json failed: {e}"})
            continue

        pred_agent, pred_step, rounds, gt_agent, gt_step, problem = run_binary_search_on_file(
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
            "per_round": rounds
        })

        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    ts = datetime.now().strftime("%m%d_%H%M")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model).strip("-") or "model"
    out_path = output_dir / f"results_binarysearch_{safe_model}_{dataset}_withGT_{ts}.json"

    summary = {
        "mode": "V8_BINARY_SEARCH",
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "directory_path": str(data_dir),
        "dataset": dataset,
        "setting": "withGT",
        "debug_mode": DEBUG_MODE,
        "debug_limit": DEBUG_SAMPLE_LIMIT if DEBUG_MODE else 0,
        "random_seed": RANDOM_SEED,
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
