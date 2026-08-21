"""ErrorProbe as generator programs — one per vendored prediction mode.

Both programs reproduce ``SimplifiedMAS.analyze_trace`` (vendored
``simplified_mas/llm_agents.py``) for a single trajectory:

- ``errorprobe_program`` — the vendored default (``backward_tracing.enabled:
  false``): one Analyzer round over the last 15 turns, then one Verifier round
  over the proposed span. The Analyzer alone decides the prediction; the
  Verifier contributes the evidence block and the combined confidence, exactly
  as in the vendored flow.
- ``errorprobe_bt_program`` — the config-gated backward-tracing mode: the
  ``backward.backward_trace`` walk (one relevance round per turn, root-cause
  rounds for relevant turns, memory-maintenance rounds, a synthesis round),
  then the same Verifier round on the converted result.

Memory retrieval is deliberately absent: the vendored baseline prompt accepts
``similar_patterns`` and never interpolates them, so the Error Pattern Memory
cannot change a prediction — see README.md.

Every response is ``strip_think``-ed (repo-wide infrastructure) and logged into
``calls`` with its role (and turn index for the backward walk); prompts are
never stored (GUIDE.md). The truncated mode's window-local span is remapped to
absolute indices only in the output document (``predicted_step``); the raw
span is kept in ``predicted_span_raw`` and is what the Verifier saw. Backend
errors never reach these programs — the runner skips the trajectory without
writing, and a rerun resumes it.
"""
from __future__ import annotations

from typing import Generator

from baselines.prompting.methods import strip_think

from .backward import TraceIndex, backward_trace
from .prompts import (
    build_analyzer_prompt,
    build_verifier_prompt,
    coerce_step,
    messages,
    parse_analysis,
    parse_verification,
    task_text,
)

METHOD = "errorprobe"
METHOD_BT = "errorprobe_bt"

# The vendored Analyzer window (history[-15:]) — also the remap offset base.
WINDOW = 15

Program = Generator[list[list[dict]], list[str], dict]


def strip_names(record: dict) -> dict:
    """A record view whose turns carry no ``name`` key.

    This repo stores the agent in ``role`` (GUIDE.md), while the upstream
    alg-gen format kept it in ``name`` — swapped here, so a turn's ``name`` is
    the generic ``"assistant"``. Dropping the key routes the vendored
    name-first extraction onto its role-fallback branch, the one written for
    exactly this data shape.
    """
    return {**record,
            "history": [{k: v for k, v in t.items() if k != "name"}
                        for t in record["history"]]}


def _combined_confidence(analysis: dict, evidence: dict):
    """The vendored hypothesis confidence: analysis × verification confidence."""
    try:
        return float(analysis["confidence"]) * float(
            evidence["probe_details"].get("verification_confidence", 0.5))
    except (TypeError, ValueError, KeyError):
        return None


def _verify(record: dict, analysis: dict, calls: list[dict]):
    """Sub-generator: the Verifier round shared by both modes."""
    raw = strip_think((yield [messages(build_verifier_prompt(record, analysis))])[0])
    calls.append({"role": "verifier", "response": raw})
    return parse_verification(raw)


def errorprobe_program(record: dict, *, include_gt: bool = False) -> Program:
    """Truncated-history mode: Analyzer round, Verifier round, output doc."""
    record = strip_names(record)
    calls: list[dict] = []

    raw_analysis = strip_think(
        (yield [messages(build_analyzer_prompt(record, include_gt=include_gt))])[0])
    calls.append({"role": "analyzer", "response": raw_analysis})
    analysis = parse_analysis(raw_analysis)

    evidence = yield from _verify(record, analysis, calls)

    start, end = analysis["error_step"]
    offset = max(0, len(record["history"]) - WINDOW)
    return {
        "predicted_agent": str(analysis["tool_used"]),
        "predicted_step": start + offset,
        "raw": raw_analysis,
        "calls": calls,
        "error_family": analysis["error_family"],
        "error_reason": analysis["error_reason"],
        "predicted_span_raw": [start, end],
        "predicted_span": [start + offset, end + offset],
        "analysis_confidence": analysis["confidence"],
        "evidence": evidence,
        "hypothesis_confidence": _combined_confidence(analysis, evidence),
    }


def errorprobe_bt_program(record: dict, *, include_gt: bool = False) -> Program:
    """Backward-tracing mode: the backward walk, then the Verifier round."""
    record = strip_names(record)
    calls: list[dict] = []

    trace_index = TraceIndex(record)
    question = task_text(record, include_gt)

    # Relay the walk's (meta, prompts) rounds to the runner, logging responses.
    walk = backward_trace(trace_index, question)
    try:
        meta, prompts = next(walk)
        while True:
            raws = yield prompts
            stripped = [strip_think(r) for r in raws]
            for s in stripped:
                calls.append({**meta, "response": s})
            meta, prompts = walk.send(stripped)
    except StopIteration as done:
        prediction = done.value

    # The vendored ``_analyze_with_backward_tracing`` conversion, plus int
    # coercion where the vendored process would crash on a non-integer step.
    try:
        step = coerce_step(prediction.get("mistake_step", 0))
    except ValueError:
        step = None
    analysis = {
        "error_step": (step or 0, step or 0),
        "error_family": prediction.get("mistake_type", "unknown"),
        "error_reason": prediction.get("mistake_reason", "No reason provided"),
        "confidence": prediction.get("confidence", 0.5),
        "tool_used": prediction.get("mistake_agent", "unknown"),
    }

    evidence = yield from _verify(record, analysis, calls)

    raw_synthesis = next((c["response"] for c in reversed(calls)
                          if c["role"] == "synthesize"), None)
    return {
        "predicted_agent": str(analysis["tool_used"]),
        "predicted_step": step,
        "raw": raw_synthesis,
        "calls": calls,
        "error_family": analysis["error_family"],
        "error_reason": analysis["error_reason"],
        "analysis_confidence": analysis["confidence"],
        "evidence": evidence,
        "hypothesis_confidence": _combined_confidence(analysis, evidence),
        "bt_method": prediction.get("method"),
        "turns_examined": prediction.get("turns_examined"),
        "total_turns": prediction.get("total_turns"),
        "memory_summary": prediction.get("memory_summary"),
    }


METHODS = {METHOD: errorprobe_program, METHOD_BT: errorprobe_bt_program}
