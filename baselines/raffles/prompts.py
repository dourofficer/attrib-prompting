"""RAFFLES prompts and parsers, transcribed from the paper (Appendix F.3).

RAFFLES ships no code, so there is no ``vendored/`` source to copy bytes from.
The source of truth is the paper itself (``baselines/raffles/raffles.pdf``,
EACL 2026, pages 7678-7686): the Judge and Evaluator texts below reproduce its
Appendix F.3 word for word, and the surrounding template is the Appendix F.1
"Prompt Template" the paper says the Judge reuses. Because a PDF wraps lines at
the width of its figure boxes, the original line breaks are unrecoverable; the
transcription keeps every word and paragraph break and joins the box-width
wraps into logical lines. ``tests/test_raffles_pipeline.py`` locks the
resulting bytes with golden fixtures. IMPLEMENTATION.md lists every choice the
paper leaves open (marked "glue" below).

Two structural readings of the paper are baked in here:

* The three Evaluator prompts differ only in the criterion phrase they verify
  and are otherwise identical, so they are one template parameterized by
  criterion (the per-box wording in the PDF is identical; only the incidental
  line wrapping differs).
* Evaluator ``p`` receives the candidate plus **only its own** rationale
  (Section 4: "given the proposed step t, τ, and the rationale r_j^p"; the
  Figure 6 caption sends each rationale block to its specific Evaluator).

GT setting: the paper evaluates Who&When *without* ground truth, so every
builder defaults to ``include_gt=False`` — the paper-faithful bytes.
``include_gt=True`` adds the prompting baselines' answer line to the problem
statement, an extension this repo supports for the paired comparison
(GUIDE.md "GT settings").
"""
from __future__ import annotations

import json
import re

from baselines.prompting.methods import strip_think

# The agent identity lives in the "role" field for every dataset in this repo.
AGENT_KEY = "role"

# ─────────────────────────────────────────────────────────────────────────────
# Appendix F.1 — the shared prompt template the Judge reuses
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are an intelligent assistant that takes in a task description, and task output and complete based on requirements.

Task Description
{task_description}

Input Metadata
{input_metadata}

Task Output
{task_output}

Remember, that your output should only be a json and nothing else."""


# ─────────────────────────────────────────────────────────────────────────────
# Appendix F.3 — Judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_TASK_DESCRIPTION = """You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. Identify which agent made an error, at which step, and explain the reason for the error.
Based on this conversation, please predict the following:
1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.
2. In which step the mistake agent first made mistake.
**You must always output an agent name and a step number.** Null, None, or empty values are strictly forbidden for the "agent_name" and "step_number" fields.
Notice that you should point out the agent and the step such that all three of the following criteria are satisfied:
1. The agent made a mistake at that step.
2. It is the first mistake step that relates to the final wrong outcome.
3. The mistake was not corrected by or correctable by later agents.
## Handling Ambiguity (Fallback Procedure)
In cases where no single agent or step perfectly meets all three criteria (for example, if the error was collaborative or no obvious mistake exists), **you must apply the following logic to make a determination:**
Identify the agent whose contribution was the **most pivotal in setting the final, incorrect direction.** This could be the agent who introduced the flawed method, provided the key piece of wrong information, or signed off on the solution without a final critical review. Select the corresponding step. This ensures you always provide a "best guess" even in unclear situations."""

# The ".**." after "at that step" is in the paper — preserved verbatim.
JUDGE_TASK_OUTPUT = """Please answer in the format:
```json
{
    "agent_name": "The name of the faulty agent you identified, satisfying all the three criteria.",
    "step_number": "The step number where the chosen agent made the mistake, satisfying all the three criteria.",
    "mistake_reason": "Briefly explain **why the agent made a mistake at that step.**. Reference the log as needed for clarity.",
    "first_mistake": "Briefly explain **why it is the first mistake step that relates to the final wrong outcome.** Reference the log as needed for clarity.",
    "mistake_not_corrected": "Briefly explain **how the mistake was not corrected by or correctable by later agents.** Reference the log as needed for clarity."
}
```"""


# ─────────────────────────────────────────────────────────────────────────────
# Appendix F.3 — Evaluators 1-3 (one template, three criterion phrases)
# ─────────────────────────────────────────────────────────────────────────────

# Criterion p → the quoted phrase Evaluator p verifies. The rationale field
# names mirror the Judge's output keys.
CRITERIA = {
    1: "correctly pointing out a faulty agent and step number",
    2: "finding the first mistake in the pipeline",
    3: "how this mistake was never corrected afterwards",
}
CRITERION_FIELDS = {1: "mistake_reason", 2: "first_mistake", 3: "mistake_not_corrected"}

EVALUATOR_TEMPLATE = """You are a rigorous and meticulous logic verifier, serving as a critical component within a reasoning system dedicated to fault attribution in complex system logs. Your specific assigned task is to verify the reasoning logic provided by your partner. Your sole purpose is to identify flaws, inconsistencies, and leaps in logic, and you must not be swayed by your partner's conclusion, but only by the soundness of their argument.
Your partner will identify the agent and the step such that all three of the following criteria are satisfied:
1. The agent made a mistake at that step.
2. It is the first mistake step that relates to the final wrong outcome.
3. The mistake was not corrected by or correctable by later agents.
Hence, you are provided with the following inputs:
- Task Log: A multi-agent conversation log
- Error Step: Output from your partner with candidate point of fault and their associated reasoning.
Your task is **ONLY** to think whether the argument provided by your partner for '{criterion}' is logical. You will try to verify the argument from the task log and give your reasons about whether this argument is logical or not. Then, you will give a confidence score between 0 to 100 indicating your confidence in the soundness of your partner's argument.
For example, general or non-specific reasoning that cannot be verified by a non-expert is less logical than specific reasoning that can be easily verified. Further, if you are unable to verify the correctness of the argument from the task log, you should give a low confidence score.
## Task Log ##
{task_log}
## Error Step ##
{error_step}
## Your output format ##
You should directly output a json in the following format:
```json
{{
    "reason": "Briefly explain why the given argument for '{criterion}' is sound or unsound. If unsound, identify the specific flaw.",
    "confidence": "Assign an integer score between 0 to 100 indicating your confidence in the **soundness and logical consistency of the partner's argument**. 100 means the argument is logical, specific, and fully supported by the log. 0 means the argument is illogical, non-specific, or contradicts the log."
}}
```"""


# ─────────────────────────────────────────────────────────────────────────────
# Glue: log serialization and slot assembly (choices, not paper text)
# ─────────────────────────────────────────────────────────────────────────────

def messages(prompt: str) -> list[dict]:
    """One user message; the paper gives no system prompt, so we add none."""
    return [{"role": "user", "content": prompt}]


def _agent_of(entry: dict) -> str:
    return entry.get(AGENT_KEY, "Unknown Agent")


def _log_text(history: list[dict]) -> str:
    """Step-labelled log, one line per turn — the step_by_step serialization,
    so the 0-based indices the Judge must answer with are visible in the log."""
    return "\n".join(
        f"Step {idx} - {_agent_of(entry)}: {entry.get('content', '')}"
        for idx, entry in enumerate(history)
    )


def build_task_log(record: dict, include_gt: bool = False) -> str:
    """Problem statement + conversation — the {task_log} / Input Metadata core.

    The answer line (with-GT extension) uses the prompting baselines' exact
    sentence so the paired with/without comparison differs by that line only.
    """
    gt_line = (f"The Answer for the problem is: {record.get('ground_truth', '')}\n"
               if include_gt else "")
    return (f"The problem is: {record.get('question', '')}\n"
            + gt_line
            + "The conversation is as follows:\n"
            + _log_text(record["history"]))


def _feedback_text(feedback: list[dict]) -> str:
    """Serialize the memory H for the Judge's next round.

    The paper says only that the Evaluators' output is "fed back to the Judge";
    this block is the glue. Each prior iteration contributes its candidate and
    the four verifier verdicts (criterion phrase, confidence, reason).
    """
    lines = ["## Feedback on your previous answers ##",
             "Independent verifiers reviewed each previous answer below. Use their "
             "feedback to propose a better candidate, or keep a previous one if "
             "the feedback supports it."]
    for entry in feedback:
        lines.append(f"### Iteration {entry['iteration']} answer ###")
        lines.append(entry["answer"])
        if entry.get("evals") is None:
            continue
        lines.append(f"### Iteration {entry['iteration']} feedback ###")
        for ev in entry["evals"]:
            name = CRITERIA.get(ev["criterion"], "candidate is consistent with the log")
            lines.append(f"- {name} (confidence {ev['confidence']}/100): {ev['reason']}")
    return "\n".join(lines)


def build_judge_prompt(record: dict, feedback: list[dict] | None = None,
                       include_gt: bool = False) -> str:
    """Assemble the Judge prompt from the F.1 template and the F.3 texts.

    Slot assembly is glue: the paper names the template and the texts but not
    the fill. ``task_description`` = the Judge text, ``input_metadata`` = the
    task log (+ prior-iteration feedback after the first round),
    ``task_output`` = the Judge's format block.
    """
    input_metadata = build_task_log(record, include_gt=include_gt)
    if feedback:
        input_metadata += "\n\n" + _feedback_text(feedback)
    return PROMPT_TEMPLATE.format(
        task_description=JUDGE_TASK_DESCRIPTION,
        input_metadata=input_metadata,
        task_output=JUDGE_TASK_OUTPUT,
    )


def build_evaluator_prompt(criterion: int, record: dict, candidate: dict,
                           include_gt: bool = False) -> str:
    """Assemble Evaluator ``criterion``'s prompt.

    The Error Step carries the candidate plus only this criterion's rationale
    (Section 4's r_j^p; see the module docstring).
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(raw: str) -> dict:
    """Pull the one JSON object out of a model response.

    Prefers a fenced ```json block (the format every prompt demands); falls
    back to the outermost braces for models that skip the fence. Raises
    ``ValueError`` when nothing parses — callers decide what a failure means.
    """
    text = strip_think(raw)
    m = _JSON_FENCE.search(text)
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
    """Parse a Judge response into a candidate dict.

    Requires ``agent_name`` and ``step_number`` (the prompt forbids empty
    values); the three rationale fields default to "" so a terse response
    still yields verifiable Evaluator prompts.
    """
    obj = extract_json(raw)
    if obj.get("agent_name") in (None, "") or obj.get("step_number") in (None, ""):
        raise ValueError("judge response lacks agent_name/step_number")
    return {
        "agent_name": str(obj["agent_name"]),
        "step_number": obj["step_number"],
        **{f: str(obj.get(f, "")) for f in CRITERION_FIELDS.values()},
    }


def parse_evaluator(raw: str) -> tuple[str, int]:
    """Parse an Evaluator response into (reason, confidence).

    Confidence is coerced to an int and clamped to [0, 100]; anything
    unparseable scores 0 — an unverifiable rationale earns no confidence,
    exactly what the Evaluator prompt tells the model.
    """
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
    """Coerce the Judge's step_number (int, "3", "3.0") to an int, else None."""
    try:
        return int(float(str(x).strip()))
    except (TypeError, ValueError):
        return None


def _norm_agent(x) -> str:
    return re.sub(r"\s+", " ", str(x)).strip().lower()


def rule_check(history: list[dict], candidate: dict) -> tuple[str, int]:
    """Evaluator 4 — the paper's rule-based consistency check, our rules.

    The paper says only that it "validate[s] whether the proposed candidate
    step t is consistent with the log": here that means the step exists and
    the agent speaking at it is the candidate agent. All or nothing (100/0),
    matching the check's yes/no nature.
    """
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