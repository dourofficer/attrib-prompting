"""Turn a trajectory into the one flat string OAT feeds its frozen LLM.

OAT never writes a prompt. It concatenates the question and every step of the
log into a single document, runs one forward pass over it, and reads the
model's internal vectors back out. This module builds that document and records
where each step starts and ends, so ``extract.py`` can tell which tokens belong
to which step.

The text format is vendored: ``vendored/OAT/data_pipeline.py:33-57`` for the
steps and ``vendored/OAT/extract_states.py:118-126`` for the wrapper. Two
things differ here, both forced by this repo's data and both documented in
IMPLEMENTATION.md:

- The vendored loader picks its step header from a ``name`` field on Who&When's
  algorithm-generated records. No corpus in ``data/`` has that field — agent
  identity is ``history[t]["role"]`` everywhere (GUIDE.md) — so every corpus
  goes through the vendored *hand-crafted* branch, whose header reads exactly
  the same field.
- ``--gt with`` adds the answer to the question. The vendored text never
  carries it, so ``--gt without`` is this baseline's default.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

# vendored/OAT/config.py:16 — tool returns are capped before serialization.
MAX_TOOL_CONTENT_LENGTH = 4096

# Turns the agent did not take. The vendored code drops ``human`` turns on
# Who&When hand-crafted (data_pipeline.py:16-19) and ``Computer_terminal``
# turns on algorithm-generated, the latter keyed on ``name``; this repo's
# records carry both identities in ``role``, and ``user`` is what the
# CORRECT-Error corpus calls the human. Environment turns whose role names a
# tool — ``bash`` and ``str_replace_editor`` in traceelephant/swe — stay in:
# there the tool call *is* the agent's action, and a gold label can land on it.
NON_MODEL_ROLES = frozenset({"human", "user", "Computer_terminal", "ComputerTerminal"})

_GT_LINE = "The Answer for the problem is: {ground_truth}"


def is_model_step(entry: dict) -> bool:
    """Did an agent take this turn, or was it the human or the environment?"""
    return str(entry.get("role", "")) not in NON_MODEL_ROLES


def agent_of(entry: dict) -> str:
    """The agent label for a step — vendored ``_agent_from_who_when_step``.

    ``vendored/OAT/data_pipeline.py:22-30``, hand-crafted branch: the human
    stays ``human``, every Magentic orchestrator variant
    (``Orchestrator (thought)``, ``Orchestrator (-> WebSurfer)``) collapses to
    ``Orchestrator``, and anything else is its own role.
    """
    role = str(entry.get("role", ""))
    if role == "human":
        return "human"
    if role.startswith("Orchestrator"):
        return "Orchestrator"
    return role


def serialize_trajectory(history: Sequence[dict]) -> tuple[str, list[tuple[int, int]], list[str], list[bool]]:
    """Serialize a log; return its text, per-step char bounds, agents and flags.

    Verbatim ``serialize_who_and_when_trajectory(history, "hand_crafted")``
    from ``vendored/OAT/data_pipeline.py:33-57``. Each step is a two-line
    record — a header naming the step index, the role and the agent, then the
    content — and the char bounds are half-inclusive ``(start, end)`` pairs
    that ``extract.py`` maps onto token spans.
    """
    parts: list[str] = []
    char_bounds: list[tuple[int, int]] = []
    agents: list[str] = []
    is_model: list[bool] = []
    cursor = 0

    for t, entry in enumerate(history):
        role = entry.get("role", "unknown")
        agent = agent_of(entry)
        header = f"[STEP {t}] [ROLE: {role}] [AGENT: {agent}]\n"
        step_text = header + str(entry.get("content", "")) + "\n"
        start = cursor
        cursor += len(step_text)
        parts.append(step_text)
        char_bounds.append((start, cursor - 1))
        agents.append(agent)
        is_model.append(is_model_step(entry))

    return "".join(parts), char_bounds, agents, is_model


def _serialize_content(content: Any) -> str:
    """Vendored ``_serialize_content`` (``data_pipeline.py:60-73``)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def serialize_mcp_atlas_trajectory(
    history: Sequence[dict],
    max_tool_content_length: int = MAX_TOOL_CONTENT_LENGTH,
) -> tuple[str, list[tuple[int, int]], list[str], list[bool]]:
    """Vendored ``serialize_mcp_atlas_trajectory`` (``data_pipeline.py:76-116``).

    The training corpus is tool-use logs, not multi-agent chat, so its steps
    carry reasoning traces and tool calls rather than an agent name. Kept
    verbatim: the training text must look to the extractor exactly as it did
    to the authors'.
    """
    parts: list[str] = []
    char_bounds: list[tuple[int, int]] = []
    agents: list[str] = []
    is_model_flags: list[bool] = []
    cursor = 0

    for t, entry in enumerate(history):
        role = entry.get("role", "unknown")
        is_model = role == "assistant"
        header = f"[STEP {t}] [ROLE: {role}]\n"
        body_parts: list[str] = []

        if is_model:
            reasoning = (entry.get("original_message") or {}).get("reasoning_content", "")
            if reasoning:
                body_parts.append(f"[REASONING]\n{reasoning}")

        content = _serialize_content(entry.get("content"))
        if not is_model and len(content) > max_tool_content_length:
            content = content[:max_tool_content_length] + "\n...[truncated]"
        if content:
            body_parts.append(content)

        for tc in entry.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            body_parts.append(f"[TOOL_CALL: {func.get('name', 'unknown')}({func.get('arguments', '')})]")

        step_text = header + ("\n".join(body_parts) + "\n" if body_parts else "\n")
        start = cursor
        cursor += len(step_text)
        parts.append(step_text)
        char_bounds.append((start, cursor - 1))
        agents.append(str(role))
        is_model_flags.append(is_model)

    return "".join(parts), char_bounds, agents, is_model_flags


def question_text(record: dict, include_gt: bool = False) -> str:
    """The question as it enters the document.

    ``include_gt=True`` appends the repo's standard answer line, the same
    sentence every prompting baseline interpolates
    (``baselines/prompting/methods.py:127``). The vendored text has no such
    line, which is why ``without`` is the default.
    """
    question = str(record.get("question", "") or "")
    if not include_gt:
        return question
    return f"{question}\n" + _GT_LINE.format(ground_truth=record.get("ground_truth", "") or "")


def build_input_text(
    question: str,
    text: str,
    step_char_boundaries: Iterable[tuple[int, int]],
) -> tuple[str, list[tuple[int, int]]]:
    """Wrap the serialized steps in the vendored question prefix.

    Verbatim ``_build_input_text`` (``vendored/OAT/extract_states.py:118-126``).
    The question gets a span of its own at the front — row 0 of the
    representation sequence, indexed ``-1``, which the model uses to seed its
    hidden state and which never receives an anomaly score.
    """
    prefix = f"Question: {question} Trajectory: "
    offset = len(prefix)
    question_start = len("Question: ")
    question_end = max(question_start, question_start + len(question) - 1)
    bounds = [(question_start, question_end)]
    bounds.extend((start + offset, end + offset) for start, end in step_char_boundaries)
    return prefix + text, bounds


def _parse_scope(scope: Any) -> Optional[int]:
    """Vendored ``_parse_scope`` (``data_pipeline.py:119-125``)."""
    if isinstance(scope, int):
        return scope
    if isinstance(scope, str) and scope.lstrip("-").isdigit():
        return int(scope)
    return None


def load_mcp_atlas_records(data_dir, include_success: bool = True) -> list[dict]:
    """Read the training corpus that ships with the vendored repo.

    MCP-Atlas is a tool-use benchmark; the vendored copy under
    ``vendored/OAT/dataset/MCP-atlas/`` is the only source of *successful*
    trajectories available to this repo — every corpus in ``data/`` is failures
    only, and OAT trains on successes alone. A record counts as successful when
    its ``errors`` list is empty; error scopes are 1-based in the file and
    become 0-based step indices here, exactly as
    ``vendored/OAT/data_pipeline.py:165-207`` does it.
    """
    import json
    from pathlib import Path

    records: list[dict] = []
    for path in sorted(Path(data_dir).glob("*.json")):
        data = json.loads(path.read_text())
        errors = data.get("errors")
        history = data.get("raw_conversation_history", [])
        if not isinstance(errors, list) or not history:
            continue

        error_steps: list[int] = []
        for err in errors:
            scope = _parse_scope(err.get("scope"))
            if scope is not None and 0 <= scope - 1 < len(history):
                error_steps.append(scope - 1)
        error_steps = sorted(set(error_steps))
        is_success = len(errors) == 0
        if is_success and not include_success:
            continue
        if not is_success and not error_steps:
            continue

        text, bounds, agents, is_model = serialize_mcp_atlas_trajectory(history)
        records.append(
            {
                "id": path.stem,
                "is_success": is_success,
                "text": text,
                "step_char_boundaries": bounds,
                "step_agents": agents,
                "step_is_model": is_model,
                "error_steps": error_steps,
                "num_steps": len(history),
                "question": data.get("PROMPT", ""),
            }
        )
    return records
