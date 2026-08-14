"""Simplified RAFFLES: same Judge-Evaluator loop, deliberately weaker prompts.

Keeps the core mechanics of the paper — a Judge proposes an (agent, step)
candidate with three criterion rationales; Evaluators 1-3 each score one
rationale 0-100; a rule-based Evaluator 4 checks the candidate against the
log; all feedback is appended to memory and fed back to the Judge on the
next iteration — but replaces the paper's prompts (role-play framing,
three-criteria contract, fallback procedure, scoring rubric) with short,
plain-language versions. Same public API and data shapes as raffles.py,
so it can slot into the pipeline as a prompt-strength ablation.
"""
from __future__ import annotations

import json
import re

AGENT_KEY = "role"

CRITERIA = {
    1: "the agent really made a mistake at that step",
    2: "it is the first mistake that led to the wrong outcome",
    3: "the mistake was never corrected later",
}
CRITERION_FIELDS = {1: "mistake_reason", 2: "first_mistake", 3: "mistake_not_corrected"}

# ── Prompts (weak versions) ──────────────────────────────────────────────────

JUDGE_TEMPLATE = """Here is a failed multi-agent conversation.

{task_log}

Which agent caused the failure, and at which step? Pick the first uncorrected mistake that led to the wrong final outcome. Always name one agent and one step number.

Answer with only this JSON:
```json
{{
    "agent_name": "",
    "step_number": 0,
    "mistake_reason": "why the agent made a mistake at that step",
    "first_mistake": "why it is the first mistake behind the wrong outcome",
    "mistake_not_corrected": "why it was never corrected later"
}}
```"""

EVALUATOR_TEMPLATE = """Here is a failed multi-agent conversation and a claim about which step went wrong.

## Log ##
{task_log}

## Claim ##
{error_step}

Check the claim's argument that {criterion}. Answer with only this JSON:
```json
{{
    "reason": "why the argument holds or does not",
    "confidence": 0
}}
```
"confidence" is an integer 0-100 for how well the argument is supported by the log."""


# ── Prompt assembly ──────────────────────────────────────────────────────────

def messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def _agent_of(entry: dict) -> str:
    return entry.get(AGENT_KEY, "Unknown Agent")


def _log_text(history: list[dict]) -> str:
    return "\n".join(
        f"Step {i} - {_agent_of(e)}: {e.get('content', '')}"
        for i, e in enumerate(history)
    )


def build_task_log(record: dict, include_gt: bool = False) -> str:
    gt_line = (f"The Answer for the problem is: {record.get('ground_truth', '')}\n"
               if include_gt else "")
    return (f"The problem is: {record.get('question', '')}\n"
            + gt_line
            + "The conversation is as follows:\n"
            + _log_text(record["history"]))


def _feedback_text(feedback: list[dict]) -> str:
    lines = ["Feedback on your previous answers:"]
    for entry in feedback:
        lines.append(f"Answer {entry['iteration']}: {entry['answer']}")
        for ev in entry.get("evals") or []:
            name = CRITERIA.get(ev["criterion"], "consistency with the log")
            lines.append(f"- {name}: {ev['confidence']}/100. {ev['reason']}")
    lines.append("Propose a better candidate, or keep one the feedback supports.")
    return "\n".join(lines)


def build_judge_prompt(record: dict, feedback: list[dict] | None = None,
                       include_gt: bool = False) -> str:
    task_log = build_task_log(record, include_gt=include_gt)
    if feedback:
        task_log += "\n\n" + _feedback_text(feedback)
    return JUDGE_TEMPLATE.format(task_log=task_log)


def build_evaluator_prompt(criterion: int, record: dict, candidate: dict,
                           include_gt: bool = False) -> str:
    field = CRITERION_FIELDS[criterion]
    error_step = json.dumps({
        "agent_name": candidate.get("agent_name"),
        "step_number": candidate.get("step_number"),
        field: candidate.get(field, ""),
    }, indent=4, ensure_ascii=False)
    return EVALUATOR_TEMPLATE.format(
        criterion=CRITERIA[criterion],
        task_log=build_task_log(record, include_gt=include_gt),
        error_step=error_step,
    )


# ── Parsers ──────────────────────────────────────────────────────────────────

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(raw: str) -> dict:
    text = _THINK.sub("", raw)
    m = _FENCE.search(text)
    candidates = [m.group(1)] if m else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in response")


def parse_judge(raw: str) -> dict:
    obj = extract_json(raw)
    if obj.get("agent_name") in (None, "") or obj.get("step_number") in (None, ""):
        raise ValueError("judge response lacks agent_name/step_number")
    return {
        "agent_name": str(obj["agent_name"]),
        "step_number": obj["step_number"],
        **{f: str(obj.get(f, "")) for f in CRITERION_FIELDS.values()},
    }


def parse_evaluator(raw: str) -> tuple[str, int]:
    try:
        obj = extract_json(raw)
    except ValueError:
        return "[unparseable evaluator response]", 0
    reason = str(obj.get("reason", ""))
    try:
        conf = int(float(str(obj.get("confidence")).strip().rstrip("%")))
    except (TypeError, ValueError):
        return reason, 0
    return reason, max(0, min(100, conf))


def normalize_step(x) -> int | None:
    try:
        return int(float(str(x).strip()))
    except (TypeError, ValueError):
        return None


def _norm_agent(x) -> str:
    return re.sub(r"\s+", " ", str(x)).strip().lower()


def rule_check(history: list[dict], candidate: dict) -> tuple[str, int]:
    """Evaluator 4 — rule-based consistency check (unchanged logic)."""
    step = normalize_step(candidate.get("step_number"))
    if step is None:
        return "The candidate step number is not a number.", 0
    if not 0 <= step < len(history):
        return (f"The log has {len(history)} steps (0 to {len(history) - 1}), "
                f"so step {step} does not exist.", 0)
    actual = _agent_of(history[step])
    if _norm_agent(actual) != _norm_agent(candidate.get("agent_name")):
        return (f"The agent at step {step} is '{actual}', not the candidate "
                f"'{candidate.get('agent_name')}'.", 0)
    return f"Step {step} exists in the log and its agent is '{actual}'.", 100