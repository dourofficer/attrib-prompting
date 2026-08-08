# filename: v1.py (step1 划分子任务 step2 错误定位)
import os
import json
import re
import time
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
import argparse

try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(".env")
except Exception:
    pass

# ===== 全局配置 =====
MODEL_NAME = None
client = None
folder = None
dataset = None

# ===== 调试模式 =====
DEBUG_MODE = False
DEBUG_SAMPLE_LIMIT = 3


# ===== 工具函数 =====
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_agent(x):
    """标准化代理名称"""
    return re.sub(r"\s+", " ", str(x)).strip().lower() if x else None


def normalize_step(x):
    """标准化步骤"""
    try:
        return int(float(str(x).strip()))
    except:
        return None


# ---------- OpenAI client (same style as CHIEF) ----------
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


def call_model(prompt):
    return call_llm(client=client, model=MODEL_NAME, prompt=prompt, temperature=0.0)


# ===== STEP 1 =====
def step1_generate_subtasks(history_text, question, ground_truth):
    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real-world problem.\n"
        f"The problem is: {question}\n"
        f"The correct answer for the problem is: {ground_truth}\n\n"
        "Here is the conversation in JSON format:\n"
        + str(history_text)
        + f"\n\nThere are total {len(history_text)} steps, each entry provides the input of the agent and its role.\n"
        "Based on this conversation, please:\n"
        "1. Decompose the reasoning into semantic subtasks.\n"
        "2. Predict the correct output (oracle) for each subtask.\n\n"
        "Please answer in the format below, strictly following the plain text structure:\n\n"
        "The Subtask Name: (your prediction)\n"
        "Step Range: (start-end)\n"
        "The Oracle: (your prediction)\n\n"
        "No overlaps between step ranges are allowed.\n"
        "Example:\n"
        "The Subtask Name: Understand the task\n"
        "Step Range: 0-2\n"
        "The Oracle: Identify time, location, and goal from the user input.\n\n"
        "Now generate your output below:\n"
    )

    response = call_model(prompt)
    result_text = response.strip()

    name_pattern = r"The Subtask Name:\s*([^\n*]+)"
    range_pattern = r"Step Range:\s*([0-9]+-[0-9]+)"
    oracle_pattern = r"The Oracle:\s*([^\n*]+)"

    names = re.findall(name_pattern, result_text)
    ranges = re.findall(range_pattern, result_text)
    oracles = re.findall(oracle_pattern, result_text)

    subtasks = []
    for subtask_name, step_range, oracle in zip(names, ranges, oracles):
        subtasks.append(
            {
                "Subtask Name": subtask_name.strip(),
                "Step Range": step_range.strip(),
                "The Oracle": oracle.strip(),
            }
        )

    return subtasks, result_text


# ===== STEP 2 =====
def step2_predict_error(history_text, question, ground_truth, subtasks):
    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation solving a real-world problem.\n"
        f"The problem is: {question}\n"
        f"The correct answer for the problem is: {ground_truth}\n\n"
        "Here is the conversation (in JSON format):\n"
        + str(history_text)
        + f"\n\nThere are total {len(history_text)} steps, each entry provides an agent's input.\n"
        "Below are the subtasks, their step ranges, and the expected oracle outputs:\n"
        + str(subtasks)
        + "\n\nYour job:\n"
        "1. Identify which agent made a reasoning mistake directly responsible for the wrong final result.\n"
        "2. Determine the exact step number where this mistake first occurred.\n"
        "3. Explain why that step is the reasoning slip.\n\n"
        "Please answer in this exact plain text format:\n"
        "Agent Name: (your prediction)\n"
        "Step Number: (your prediction)\n"
        "Reason for Mistake: (your explanation)\n"
        "No special symbols, no extra commentary.\n"
    )

    response = call_model(prompt)
    result_text = response.strip()

    agent_pattern = r"Agent Name:\s*([^\n*]+)"
    step_pattern = r"Step Number:\s*([0-9]+)"
    reason_pattern = r"Reason for Mistake:\s*(.*)"

    agent = re.search(agent_pattern, result_text)
    step = re.search(step_pattern, result_text)
    reason = re.search(reason_pattern, result_text, re.S)

    agent_name = agent.group(1).strip() if agent else None

    if agent_name:
        agent_name = agent_name.split("(", 1)[0].strip()

    step_number = int(step.group(1)) if step else None
    reason_text = reason.group(1).strip() if reason else ""

    return {
        "final": {
            "mistake_agent": agent_name,
            "mistake_step": step_number,
            "reason": reason_text,
        },
        "raw": result_text,
    }


def process_sample(file_path, idx):
    data = load_json(file_path)
    history_text = data["history"]
    question = data.get("question", "")
    gt_agent = normalize_agent(data.get("mistake_agent"))
    gt_step = normalize_step(data.get("mistake_step"))
    ground_truth = data.get("ground_truth", "")

    subtasks, subtask_text = step1_generate_subtasks(history_text, question, ground_truth)
    error_pred = step2_predict_error(history_text, question, ground_truth, subtasks)

    pred_agent = normalize_agent(error_pred["final"]["mistake_agent"])
    pred_step = normalize_step(error_pred["final"]["mistake_step"])

    acc_agent = 1 if (pred_agent and gt_agent and pred_agent == gt_agent) else 0
    acc_step = 1 if (pred_step is not None and gt_step is not None and pred_step == gt_step) else 0

    print("=" * 80)
    print(f"Sample {idx+1}: {os.path.basename(file_path)}")
    print(f"Question: {question}")
    print(f"→ Prediction: agent='{pred_agent}' | step={pred_step}")
    print(f"→ Answer: agent='{gt_agent}' | step={gt_step}")
    print(f"→ Accuracy: agent={'✅' if acc_agent else '❌'} | step={'✅' if acc_step else '❌'}")

    return {
        "file": file_path,
        "question": question,
        "gt": {"agent": gt_agent, "step": gt_step},
        "pred": {"agent": pred_agent, "step": pred_step},
        "acc_agent": acc_agent,
        "acc_step": acc_step,
        "step1_raw": subtask_text,
        "step2_raw": error_pred["raw"],
        "final_pred": error_pred["final"],
    }, acc_agent, acc_step


# ===== 主函数 =====
def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="e.g., data/Algorithm-Generated")
    parser.add_argument("--model", required=True, help="e.g., deepseek-chat / gpt-4o-mini / ...")
    parser.add_argument("--limit", type=int, default=None, help="Optional: only run first N samples")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional: sleep seconds between requests")
    args = parser.parse_args()

    global folder, dataset, MODEL_NAME, client

    folder = args.data_dir
    MODEL_NAME = args.model

    if not os.path.isdir(folder):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
        if os.path.isdir(alt):
            folder = alt
        else:
            raise RuntimeError(f"Directory not found: {folder}")

    folder_name = os.path.basename(os.path.normpath(folder))
    if folder_name == "Algorithm-Generated":
        dataset = "AG"
    elif folder_name == "Hand-Crafted":
        dataset = "HC"
    else:
        dataset = folder_name

    client = build_client()

    start = time.time()
    json_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
    json_files = sorted(json_files)

    if args.limit is not None:
        json_files = json_files[: max(0, int(args.limit))]
        print(f"🧪 Limit mode: only {len(json_files)} samples")

    if DEBUG_MODE:
        json_files = json_files[:DEBUG_SAMPLE_LIMIT]
        print(f"🧪 Debug: only {len(json_files)} samples")

    results = []
    acc_a = acc_s = 0

    for i, file in enumerate(tqdm(json_files, desc="Running V1 Mode")):
        try:
            res, aa, ss = process_sample(file, i)
            results.append(res)
            acc_a += aa
            acc_s += ss
        except Exception as e:
            print(f"[❌] {file}: {e}")

        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    total = len(json_files)
    summary = {
        "mode": "V1",
        "model": MODEL_NAME,
        "data_dir": folder,
        "dataset": dataset,
        "total": total,
        "agent_acc": acc_a / total if total else 0,
        "step_acc": acc_s / total if total else 0,
        "details": results,
    }

    ts = datetime.now().strftime("%m%d_%H%M")
    os.makedirs("./results/v1", exist_ok=True)
    prefix = f"./results/v1/results_v1_{MODEL_NAME}_{dataset}_{ts}.json"
    with open(prefix, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ mode: V1")
    print(f"✅ Model: {MODEL_NAME}")
    print(f"✅ Dataset: {dataset}")
    print(f"✅ num of samples: {total}")
    print(f"Agent accuracy: {summary['agent_acc']*100:.2f}%")
    print(f"Step accuracy: {summary['step_acc']*100:.2f}%")
    print(f"📂 Output file: {prefix}")
    print(f"Time cost: {(time.time()-start)/60:.2f} min")


if __name__ == "__main__":
    main()
