"""ErrorProbe Analyzer/Verifier prompts and parsers, from the vendored code.

Everything the model sees or that decides a prediction is transcribed from
``vendored/ERRORPROBE`` byte for byte and parity-tested against it
(``tests/test_errorprobe_pipeline.py`` drives the vendored classes themselves):

- :func:`build_analyzer_prompt` ← ``AnalyzerAgent._build_analysis_prompt``
  (``simplified_mas/llm_agents.py``): the task, the last 15 turns at 500
  characters each, the agent-options list, the 14-mode MAST taxonomy, and a
  JSON answer contract asking for an error turn span, family, agent and reason.
- :func:`parse_analysis` ← ``AnalyzerAgent._parse_llm_response``: fenced-JSON
  extraction, the ``error_step`` / ``error_step_start`` / ``error_start_turn``
  key cascade, and the vendored parse-failure sentinel
  (agent ``"unknown"``, span ``(0, 0)``, confidence 0).
- :func:`build_verifier_prompt` ← ``VerifierAgent._build_verification_prompt``:
  the proposed analysis plus a ±2-turn context slice at 300 characters each.
- :func:`parse_verification` ← ``VerifierAgent._parse_verification_response``:
  the evidence dict (verified flag, impact estimate, suggested fix).

The vendored code reads the agent from a turn's ``name`` field and falls back
to ``role`` (split before any parenthesis, generic chat roles → ``Unknown``).
This repo stores the agent in ``role`` (GUIDE.md), so callers hand these
builders a history whose turns carry no ``name`` key and the fallback branch —
written for exactly this format — is the one exercised (see
``methods.strip_names``). Deliberate deviations beyond that are listed in
``IMPLEMENTATION.md``; the main ones are the answer line ``--gt with`` appends
to the task text (:func:`task_text`) and coercing numeric step values where
the vendored process would crash on a non-integer.
"""
from __future__ import annotations

import json

# The MAST taxonomy exactly as the Analyzer prompt renders it: the 14 failure
# modes of ``get_all_mast_families()`` in FC1 → FC2 → FC3 order, each with its
# ``ERROR_FAMILIES`` description (``core/epm_schema.py``). The vendored
# legacy/extra families are excluded there and hence here.
MAST_FAMILIES = {
    # FC1: Design and Specification Failures
    "fail_task_spec": "Fails to follow task requirements/instructions",
    "fail_role_spec": "Fails to follow assigned agent role specifications",
    "step_repetition": "Repeats same steps due to rigid turn configurations",
    "context_loss": "Loses conversation history or critical context information",
    "unaware_stopping": "Fails to recognize task completion conditions",
    # FC2: Coordination and Communication Failures
    "conversation_reset": "Unexpected conversation restart losing prior context",
    "fail_clarification": "Proceeds with wrong assumptions instead of asking clarification",
    "task_derailment": "Deviates from original task objective",
    "info_withholding": "Withholds crucial information from other agents",
    "ignored_input": "Ignores inputs or feedback from other agents",
    "reasoning_action_mismatch": "Mismatch between stated reasoning and actual actions taken",
    # FC3: Verification and Termination Failures
    "premature_termination": "Terminates task before completion",
    "incomplete_verification": "No verification or only superficial checks performed",
    "incorrect_verification": "Verification process produces incorrect assessment",
}

# The vendored parse-failure result (``_parse_llm_response``'s except branch):
# span (0, 0), family/agent "unknown", confidence 0. Kept verbatim — it counts
# as a wrong prediction under the shared metrics, like every other baseline's
# parse failure.
ANALYSIS_FAILURE = {
    "error_step": (0, 0),
    "error_family": "unknown",
    "error_reason": "Failed to parse LLM response",
    "confidence": 0.0,
    "tool_used": "unknown",
    "context": {},
}


def messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def task_text(record: dict, include_gt: bool = False) -> str:
    """The task statement; ``--gt with`` appends the repo's standard answer line."""
    question = record.get("question", "Unknown task")
    if include_gt:
        return f"{question}\nThe Answer for the problem is: {record.get('ground_truth', '')}"
    return question


def agent_from_turn(turn: dict) -> str:
    """The vendored per-turn agent extraction (conversation-formatting branch)."""
    agent = turn.get("name")
    if not agent:
        role_str = turn.get("role", "")
        agent = role_str.split("(")[0].strip() if role_str else "Unknown"
        if agent.lower() in ["human", "assistant", "user", "system"]:
            agent = "Unknown"
    if not agent or agent == "":
        agent = "Unknown"
    return agent


def collect_agent_names(history: list[dict]) -> list[str]:
    """The vendored agent-options set: every turn, generic chat roles excluded.

    Order note: the vendored code lists a plain ``set``, so the ordering
    depends on the process's string hashing (PYTHONHASHSEED). Preserved as-is;
    the parity test runs both sides in one process.
    """
    agent_names = set()
    for turn in history:
        agent = turn.get("name")
        if not agent:
            role_str = turn.get("role", "")
            agent = role_str.split("(")[0].strip() if role_str else None
            if agent and agent.lower() not in ["human", "assistant", "user", "system", ""]:
                agent_names.add(agent)
        elif agent and agent != "Unknown":
            agent_names.add(agent)
    return list(agent_names) if agent_names else ["Unknown"]


def format_agent_options(agent_names: list[str]) -> str:
    """Verbatim from ``utils/trace_utils.py``: a numbered classification list."""
    if not agent_names:
        return "No agents identified in trace"
    lines = ["Available agents in this trace:"]
    for i, name in enumerate(agent_names, 1):
        lines.append(f"  {i}. {name}")
    return "\n".join(lines)


def build_analyzer_prompt(record: dict, include_gt: bool = False) -> str:
    """The Analyzer prompt: last 15 turns, 500-char contents, MAST taxonomy."""
    history = record.get("history", [])
    task_goal = task_text(record, include_gt)

    agent_names = collect_agent_names(history)
    agent_options = format_agent_options(agent_names)

    conversation = []
    for idx, turn in enumerate(history[-15:]):  # the vendored 15-turn window
        agent = agent_from_turn(turn)
        content = turn.get("content", "")[:500]
        conversation.append(f"""Turn {idx}:
  Agent: {agent}
  Content: {content}
---""")
    conv_text = "\n".join(conversation)

    error_types_text = "\n".join(
        f"- {family}: {desc}" for family, desc in MAST_FAMILIES.items()
    )

    return f"""You are an expert at analyzing failed multi-agent execution traces.

TASK:
{task_goal}

CONVERSATION HISTORY (last 15 turns):
{conv_text}

{agent_options}

Based on the above trace, identify the error:
1. Which turn(s) contain the error?
2. Which agent caused the error?
3. What type of error occurred?
4. Why did it happen?

ERROR TAXONOMY:
{error_types_text}

RESPOND IN JSON FORMAT:
{{
  "error_start_turn": <number>,
  "error_end_turn": <number>,
  "error_family": "<type from taxonomy above>",
  "error_agent": "<agent name - must be from available agents list>",
  "error_reason": "<explanation>",
  "confidence": <0.0 to 1.0>
}}

Only output valid JSON, no other text."""


def _extract_json_block(response_text: str) -> str:
    """The vendored fence extraction shared by both parsers."""
    if "```json" in response_text:
        return response_text.split("```json")[1].split("```")[0].strip()
    if "```" in response_text:
        return response_text.split("```")[1].strip()
    return response_text.strip()


def coerce_step(value) -> int:
    """Coerce a JSON step value to int; raises ValueError on garbage.

    Deviation: the vendored code passes raw JSON values through, and a string
    step later crashes the process at the Verifier's ``range(...)``. We coerce
    numeric strings/floats here; anything else raises inside the parser's
    ``try`` and degrades to the vendored parse-failure sentinel.
    """
    return int(float(str(value).strip()))


def parse_analysis(response_text: str) -> dict:
    """Verbatim ``_parse_llm_response`` decision rules + step coercion."""
    try:
        data = json.loads(_extract_json_block(response_text))
        agent = data.get("error_agent") or data.get("tool_or_component", "unknown")
        if "error_step" in data:
            step = coerce_step(data["error_step"])
            error_step = (step, step)
        elif "error_step_start" in data and "error_step_end" in data:
            error_step = (coerce_step(data["error_step_start"]),
                          coerce_step(data["error_step_end"]))
        elif "error_start_turn" in data or "error_end_turn" in data:
            error_step = (coerce_step(data.get("error_start_turn", 0)),
                          coerce_step(data.get("error_end_turn", 0)))
        else:
            error_step = (0, 0)
        return {
            "error_step": error_step,
            "error_family": data.get("error_family", "unknown"),
            "error_reason": data.get("error_reason", "No reason provided"),
            "confidence": float(data.get("confidence", 0.5)),
            "tool_used": agent,
            "context": data,
        }
    except Exception:
        return dict(ANALYSIS_FAILURE)


def build_verifier_prompt(record: dict, analysis: dict) -> str:
    """The Verifier prompt: the analysis plus ±2 turns of 300-char context.

    Faithfulness note: ``analysis["error_step"]`` is the Analyzer's *raw* span
    (window-local indices), and the slice below indexes the absolute history
    with it — exactly what the vendored code does. The absolute remap happens
    only in the output document, never here.
    """
    history = record.get("history", [])
    error_start, error_end = analysis["error_step"]

    context_turns = []
    for idx in range(max(0, error_start - 2), min(len(history), error_end + 3)):
        turn = history[idx]
        role = turn.get("role", "unknown")
        content = turn.get("content", "")[:300]
        marker = " <<<ERROR TURN" if error_start <= idx <= error_end else ""
        context_turns.append(f"Turn {idx} [{role}]{marker}: {content}")
    context_text = "\n".join(context_turns)

    return f"""You are verifying an error analysis.

PROPOSED ERROR ANALYSIS:
- Error turns: {error_start} to {error_end}
- Error type: {analysis['error_family']}
- Reason: {analysis['error_reason']}
- Confidence: {analysis['confidence']}

TRACE CONTEXT:
{context_text}

TASK: Verify if this analysis is correct by:
1. Checking if the error turns make sense
2. Assessing if the error type is appropriate
3. Evaluating the reasoning quality
4. Estimating impact if this error were fixed (0.0 to 1.0)
5. Suggesting a fix (guard, rewrite, or prompt hint)

RESPOND IN JSON:
{{
  "verified": <true/false>,
  "verification_confidence": <0.0 to 1.0>,
  "impact_if_fixed": <0.0 to 1.0>,
  "evidence_count": <number of supporting evidence pieces>,
  "suggested_fix": {{
    "type": "guard|rewrite|hint",
    "fix_text": "<actual fix>"
  }},
  "verification_notes": "<why verified or not>"
}}

Only output valid JSON."""


def parse_verification(response_text: str) -> dict:
    """Verbatim ``_parse_verification_response``, as a plain evidence dict."""
    try:
        data = json.loads(_extract_json_block(response_text))
        return {
            "ablation_gain": float(data.get("impact_if_fixed", 0.0)),
            "static_hits": int(data.get("evidence_count", 0)),
            "replays": 1,
            "verified": bool(data.get("verified", False)),
            "probe_details": {
                "verification_confidence": data.get("verification_confidence", 0.0),
                "suggested_fix": data.get("suggested_fix", {}),
                "notes": data.get("verification_notes", ""),
            },
        }
    except Exception as exc:
        return {
            "ablation_gain": 0.0,
            "static_hits": 0,
            "replays": 0,
            "verified": False,
            "probe_details": {"parse_error": str(exc)},
        }
