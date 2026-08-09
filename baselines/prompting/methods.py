"""The three Who&When attribution methods, as per-trajectory programs.

Prompts and control-flow are copied **verbatim** from the vendored baseline
``vendored/Agents_Failure_Attribution/Automated_FA/Lib/local_model.py`` (the
open-model path). The only deliberate deviations:

  * the agent-identity field is ``history[t]["role"]`` for every dataset (this
    repo's data stores the agent name in ``role`` and ``mistake_agent`` matches
    it; the vendored ``name``/``role`` ``is_handcrafted`` switch targeted the
    *original* Who&When layout), and
  * outputs are passed through :func:`strip_think` before parsing, so reasoning
    backbones are handled, and
  * ``step_by_step`` has two execution modes (see below); the vendored code
    early-stops at the first "Yes", and both modes produce that exact
    prediction because the per-step judgment depends only on the deterministic
    accumulated history.

Each method is a **generator program** over a single trajectory: it yields one
round's prompts (a list of chat message lists), receives that round's decoded
responses (``list[str]``, same order), and finally *returns* the prediction

    {"predicted_agent": str|None, "predicted_step": int|None,
     "raw": str|None, "calls": list[dict]}

``raw`` keeps the legacy semantics (all_at_once: the one response;
step_by_step: the firing step's response or None; binary_search: the last
round's response); ``calls`` is the full response log in issue order with
method-specific metadata. A driver in :mod:`.runner` decides how programs
execute: lockstep giant batches for vLLM, independent per-trajectory
completion (with incremental writes/resume) for API backends.

step_by_step modes
------------------
``"batch"`` (vLLM default) yields every step's prompt as one round and scans
for the earliest response starting with "1. yes". ``"early_stop"`` (API
default, and the vendored control flow) yields one step per round and returns
at the first "Yes", saving roughly half the paid calls. Prompts are
byte-identical between modes; only the number of issued calls differs.
"""
from __future__ import annotations

import re
from typing import Generator

# The agent identity lives in the "role" field for every dataset in this repo.
AGENT_KEY = "role"

SYSTEM_PROMPT = "You are a helpful assistant skilled in analyzing conversations."

# Parsing regexes — identical to the vendored evaluate.py.
AGENT_RE = re.compile(r"Agent Name:\s*\(?\s*([\w_]+)\s*\)?", re.IGNORECASE)
STEP_RE = re.compile(r"Step Number:\s*\(?\s*(\d+)\s*\)?", re.IGNORECASE)

# Markdown decoration to drop before applying the vendored regexes. Reasoning
# models (e.g. DeepSeek-R1) bold the labels — `**Agent Name:** WebSurfer` — which
# defeats `Agent Name:\s*([\w_]+)` (the `*` right after the colon blocks `[\w_]+`).
# Stripping `* ` ` `# ` is a no-op on the GPT/Qwen-style outputs the vendored code
# targeted, so the regexes stay faithful and just become markdown-tolerant.
_MARKDOWN_RE = re.compile(r"[*`#]")

# ``<think> ... </think>`` (DOTALL). Also handles a dangling ``</think>`` with no
# opening tag (some templates inject the opener into the prompt, so the model only
# emits the closer).
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_DANGLING_CLOSE = re.compile(r"^.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove reasoning traces so the baseline parsers see only the answer."""
    if text is None:
        return ""
    out = _THINK_BLOCK.sub("", text)
    if "</think>" in out:
        out = _DANGLING_CLOSE.sub("", out)
    return out.strip()


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
# Method programs
# ─────────────────────────────────────────────────────────────────────────────

Program = Generator[list[list[dict]], list[str], dict]


def _empty_pred() -> dict:
    return {"predicted_agent": None, "predicted_step": None, "raw": None, "calls": []}


def all_at_once_program(record: dict, *, step_mode: str = "batch") -> Program:
    prompt = _messages(
        build_all_at_once_prompt(record["history"], record["question"], record["ground_truth"])
    )
    raw = (yield [prompt])[0]
    agent, step = parse_all_at_once(raw)
    return {
        "predicted_agent": agent,
        "predicted_step": step,
        "raw": raw,
        "calls": [{"response": raw}],
    }


def _fires(raw: str) -> bool:
    """The vendored step_by_step decision: response starts with '1. yes'."""
    return strip_think(raw).lower().strip().startswith("1. yes")


def step_by_step_program(record: dict, *, step_mode: str = "batch") -> Program:
    # Prompt construction is shared by both modes: the judgment at step idx
    # depends only on the deterministic accumulated history, so the prompt
    # bytes are identical whether or not later steps are ever issued.
    prompts: list[list[dict]] = []
    metas: list[tuple[int, str]] = []  # (step_idx, agent_name)
    acc = ""
    for idx, entry in enumerate(record["history"]):
        agent_name = _agent_of(entry)
        content = entry.get("content", "")
        acc += f"Step {idx} - {agent_name}: {content}\n"
        prompts.append(_messages(
            build_step_by_step_prompt(record["question"], record["ground_truth"], acc, idx, agent_name)
        ))
        metas.append((idx, agent_name))

    pred = _empty_pred()
    if step_mode == "batch":
        # One round with every step's prompt; earliest "1. yes" wins.
        outputs = yield prompts
        pred["calls"] = [
            {"step": idx, "agent": agent, "response": raw}
            for (idx, agent), raw in zip(metas, outputs)
        ]
        for (idx, agent_name), raw in zip(metas, outputs):
            if _fires(raw):
                pred.update(predicted_agent=agent_name, predicted_step=idx, raw=raw)
                break
    elif step_mode == "early_stop":
        # One step per round, stop at the first "Yes" — the vendored control flow.
        for (idx, agent_name), prompt in zip(metas, prompts):
            raw = (yield [prompt])[0]
            pred["calls"].append({"step": idx, "agent": agent_name, "response": raw})
            if _fires(raw):
                pred.update(predicted_agent=agent_name, predicted_step=idx, raw=raw)
                break
    else:
        raise ValueError(f"unknown step_mode {step_mode!r} (expected batch | early_stop)")
    return pred


def binary_search_program(record: dict, *, step_mode: str = "batch") -> Program:
    history = record["history"]
    pred = _empty_pred()
    start, end = 0, len(history) - 1  # empty history → start > end → no rounds

    round_no = 0
    while start < end:
        mid = start + (end - start) // 2
        segment = history[start:end + 1]
        chat_content = "\n".join(
            f"{_agent_of(entry)}: {entry.get('content', '')}" for entry in segment
        )
        prompt = build_binary_search_prompt(
            record["question"],
            record["ground_truth"],
            chat_content,
            range_description=f"from step {start} to step {end}",
            upper_half_desc=f"from step {start} to step {mid}",
            lower_half_desc=f"from step {mid + 1} to step {end}",
        )
        raw = (yield [_messages(prompt)])[0]
        pred["calls"].append(
            {"round": round_no, "start": start, "end": end, "mid": mid, "response": raw}
        )
        pred["raw"] = raw  # keep the last segment's response
        result_lower = strip_think(raw).lower().strip()
        if "upper half" in result_lower:
            start, end = start, mid
        elif "lower half" in result_lower:
            start, end = min(mid + 1, end), end
        else:
            # Ambiguous → default to upper half (matches local variant).
            start, end = start, mid
        round_no += 1

    if history:
        step = start
        pred["predicted_agent"] = _agent_of(history[step])
        pred["predicted_step"] = step
    return pred


METHODS = {
    "all_at_once": all_at_once_program,
    "step_by_step": step_by_step_program,
    "binary_search": binary_search_program,
}
