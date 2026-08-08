"""The three Who&When attribution methods — batched over trajectories via VLLM.

Prompts and control-flow are copied **verbatim** from the vendored local baseline
``baselines/Agents_Failure_Attribution/Automated_FA/Lib/local_model.py`` (the
open-model path). The only deliberate deviations (see the plan) are:

  * the agent-identity field is ``history[t]["role"]`` for every dataset (this
    repo's data stores the agent name in ``role`` and ``mistake_agent`` matches
    it; the vendored ``name``/``role`` ``is_handcrafted`` switch targeted the
    *original* Who&When layout), and
  * outputs are passed through :func:`~baselines.prompting.engine.strip_think`
    before parsing, so reasoning backbones are handled, and
  * ``step_by_step`` evaluates every step in one batch instead of stopping at the
    first "Yes" — the per-step judgment depends only on the deterministic
    accumulated history, so the *prediction* (earliest "Yes") is identical.

Each method takes ``records`` (list of dicts with keys ``history``, ``question``,
``ground_truth``) and a :class:`PromptEngine`, and returns a list of prediction
dicts aligned to ``records``: ``{"predicted_agent", "predicted_step", "raw"}``
(step is an ``int`` or ``None``; agent is a ``str`` or ``None``).
"""
from __future__ import annotations

import re

from .engine import PromptEngine, strip_think

# The agent identity lives in the "role" field for every dataset in this repo.
AGENT_KEY = "role"

SYSTEM_PROMPT = "You are a helpful assistant skilled in analyzing conversations."

# Parsing regexes — identical to the vendored evaluate.py.
AGENT_RE = re.compile(r"Agent Name:\s*([\w_]+)", re.IGNORECASE)
STEP_RE = re.compile(r"Step Number:\s*(\d+)", re.IGNORECASE)

# Markdown decoration to drop before applying the vendored regexes. Reasoning
# models (e.g. DeepSeek-R1) bold the labels — `**Agent Name:** WebSurfer` — which
# defeats `Agent Name:\s*([\w_]+)` (the `*` right after the colon blocks `[\w_]+`).
# Stripping `* ` ` `# ` is a no-op on the GPT/Qwen-style outputs the vendored code
# targeted, so the regexes stay faithful and just become markdown-tolerant.
_MARKDOWN_RE = re.compile(r"[*`#]")


def _strip_markdown(text: str) -> str:
    return _MARKDOWN_RE.sub("", text)


def parse_all_at_once(raw: str) -> tuple[str | None, int | None]:
    """Extract (predicted_agent, predicted_step) from an all-at-once generation.

    Faithful to the vendored evaluate.py regexes; only strips <think> and cosmetic
    markdown first. Pure function of `raw` so it can re-parse stored predictions.
    """
    answer = _strip_markdown(strip_think(raw))
    agent_m = AGENT_RE.search(answer)
    step_m = STEP_RE.search(answer)
    return (agent_m.group(1) if agent_m else None,
            int(step_m.group(1)) if step_m else None)


def _agent_of(entry: dict) -> str:
    return entry.get(AGENT_KEY, "Unknown Agent")


def _messages(user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders (verbatim from local_model.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_all_at_once_prompt(history: list[dict], problem: str, ground_truth: str) -> str:
    chat_content = "\n".join(
        f"{_agent_of(entry)}: {entry.get('content', '')}" for entry in history
    )
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {problem} \n"
        f"The Answer for the problem is: {ground_truth}\n"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n" + chat_content +
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        '{\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n"agent a": "xxxxxxx"\n},\n'
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n."
    )


def build_step_by_step_prompt(
    problem: str, ground_truth: str, current_conversation_history: str, idx: int, agent_name: str
) -> str:
    return (
        f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
        f"The Answer for the problem is: {ground_truth}\n"
        f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
        f"The most recent step ({idx}) was by '{agent_name}'.\n"
        # NOTE: this line is a plain (non-f) string in the source, so the literal
        # "{idx}" is intentionally left unformatted — reproduced verbatim.
        "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
        "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
        "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
        "Attention: Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
    )


def build_binary_search_prompt(
    problem: str,
    answer: str,
    chat_segment_content: str,
    range_description: str,
    upper_half_desc: str,
    lower_half_desc: str,
) -> str:
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
        f"The problem to address is as follows: {problem}\n"
        f"The Answer for the problem is: {answer}\n"
        f"Review the following conversation segment {range_description}:\n\n{chat_segment_content}\n\n"
        f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
        "Please simply output either 'upper half' or 'lower half'. You should not output anything else."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Methods
# ─────────────────────────────────────────────────────────────────────────────

def all_at_once(records: list[dict], engine: PromptEngine) -> list[dict]:
    prompts = [
        _messages(build_all_at_once_prompt(r["history"], r["question"], r["ground_truth"]))
        for r in records
    ]
    outputs = engine.generate(prompts)

    preds = []
    for raw in outputs:
        agent, step = parse_all_at_once(raw)
        preds.append({
            "predicted_agent": agent,
            "predicted_step": step,
            "raw": raw,
        })
    return preds


def step_by_step(records: list[dict], engine: PromptEngine) -> list[dict]:
    # Flatten all (record, step) pairs into one batch. The judgment at step idx
    # depends only on the deterministic accumulated history, so batching is exact.
    prompts: list[list[dict]] = []
    owners: list[tuple[int, int, str]] = []  # (record_idx, step_idx, agent_name)

    for ri, r in enumerate(records):
        history = r["history"]
        acc = ""
        for idx, entry in enumerate(history):
            agent_name = _agent_of(entry)
            content = entry.get("content", "")
            acc += f"Step {idx} - {agent_name}: {content}\n"
            prompts.append(_messages(
                build_step_by_step_prompt(r["question"], r["ground_truth"], acc, idx, agent_name)
            ))
            owners.append((ri, idx, agent_name))

    outputs = engine.generate(prompts)

    # Per record, pick the earliest step whose answer starts with "1. yes".
    preds: list[dict] = [
        {"predicted_agent": None, "predicted_step": None, "raw": None}
        for _ in records
    ]
    for (ri, idx, agent_name), raw in zip(owners, outputs):
        if preds[ri]["predicted_step"] is not None:
            continue  # already found an earlier decisive step for this record
        answer = strip_think(raw)
        if answer.lower().strip().startswith("1. yes"):
            preds[ri] = {
                "predicted_agent": agent_name,
                "predicted_step": idx,
                "raw": raw,
            }
    return preds


def binary_search(records: list[dict], engine: PromptEngine) -> list[dict]:
    # Round-by-round, batched across trajectories at the same recursion depth.
    # State per record: [start, end]; active while start < end.
    states = [[0, len(r["history"]) - 1] for r in records]
    # Records with empty history are already filtered upstream; guard anyway.
    preds: list[dict] = [{"predicted_agent": None, "predicted_step": None, "raw": None}
                         for _ in records]

    while True:
        active = [i for i, (s, e) in enumerate(states) if s < e]
        if not active:
            break

        prompts: list[list[dict]] = []
        meta: list[tuple[int, int]] = []  # (record_idx, mid)
        for i in active:
            start, end = states[i]
            history = records[i]["history"]
            mid = start + (end - start) // 2
            segment = history[start:end + 1]
            chat_content = "\n".join(
                f"{_agent_of(entry)}: {entry.get('content', '')}" for entry in segment
            )
            prompt = build_binary_search_prompt(
                records[i]["question"],
                records[i]["ground_truth"],
                chat_content,
                range_description=f"from step {start} to step {end}",
                upper_half_desc=f"from step {start} to step {mid}",
                lower_half_desc=f"from step {mid + 1} to step {end}",
            )
            prompts.append(_messages(prompt))
            meta.append((i, mid))

        outputs = engine.generate(prompts)

        for (i, mid), raw in zip(meta, outputs):
            start, end = states[i]
            result_lower = strip_think(raw).lower().strip()
            if "upper half" in result_lower:
                states[i] = [start, mid]
            elif "lower half" in result_lower:
                states[i] = [min(mid + 1, end), end]
            else:
                # Ambiguous → default to upper half (matches local variant).
                states[i] = [start, mid]
            preds[i]["raw"] = raw  # keep the last segment's response

    for i, (start, end) in enumerate(states):
        history = records[i]["history"]
        step = start if history else 0
        if history:
            preds[i]["predicted_agent"] = _agent_of(history[step])
            preds[i]["predicted_step"] = step
    return preds


METHODS = {
    "all_at_once": all_at_once,
    "step_by_step": step_by_step,
    "binary_search": binary_search,
}
