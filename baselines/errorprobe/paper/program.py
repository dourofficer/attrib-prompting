"""The paper pipeline as one generator program: four rounds per trajectory.

Round 1 sends every tagger chunk and every dependency chunk at once. Between
rounds the graph is assembled, searched backward from the symptom, and
condensed. Round 2 is the Strategist; round 3 one Investigator call per
hypothesis; round 4 the Arbiter. All prompts of a round go out together, so
``run_batched`` batches them across trajectories and ``run_streaming`` runs
trajectories concurrently, as for the other modes.

Failure semantics follow the repo's convention that an unreadable answer is a
wrong answer, with one exception borrowed from the vendored backward mode's
synthesis step: when the Arbiter's answer cannot be read, the best-evidenced
hypothesis stands as the prediction (``failure_stage: "arbiter"``). Only a
Strategist that proposes nothing yields a null prediction
(``failure_stage: "strategist"``). Backend errors never reach this module; the
runner skips the trajectory and a rerun resumes it.

``calls`` logs every response with its role and, for chunked and per-hypothesis
calls, which chunk or hypothesis it served. Prompts are never stored.
"""
from __future__ import annotations

from typing import Generator

from baselines.prompting.methods import strip_think

from ..prompts import messages
from .graph import (
    build_dependency_prompt, condense, effective_receptive_field, merge_edges,
    parse_edges, sequential_edges, structural_edges,
)
from .structure import agent_list, agent_roles, build_steps, system_description, task_step_index
from .tagger import build_tagger_prompt, chunk_steps, parse_tags
from .team import (
    build_arbiter_prompt, build_investigator_prompt, build_strategist_prompt,
    memory_candidate, parse_evidence, parse_hypotheses, parse_verdict, select_fallback,
)

METHOD_PAPER = "errorprobe_paper"

PAPER_DEFAULTS = {
    "max_hypotheses": 3,
    "chunk_chars": 12000,
    "condensed_chars": 20000,
    "include_mast_examples": False,
    "sequential_edges": "fallback",
}
STEP_CHARS = 1200          # per-step clip in the tagger, dependency and evidence prompts
MEMORY_TAU = 0.7           # the paper's tau for the (absent) write gate

Program = Generator[list[list[dict]], list[str], dict]


def _null_doc(record: dict, calls: list[dict], raw: str | None, stage: str, **extra) -> dict:
    return {"predicted_agent": None, "predicted_step": None, "raw": raw, "calls": calls,
            "error_family": None, "error_reason": None, "confidence": None,
            "failure_stage": stage, **extra}


def errorprobe_paper_program(record: dict, *, include_gt: bool = False,
                             max_hypotheses: int = PAPER_DEFAULTS["max_hypotheses"],
                             chunk_chars: int = PAPER_DEFAULTS["chunk_chars"],
                             condensed_chars: int = PAPER_DEFAULTS["condensed_chars"],
                             include_mast_examples: bool = PAPER_DEFAULTS["include_mast_examples"],
                             sequential_edges: str = PAPER_DEFAULTS["sequential_edges"],
                             retrieved_patterns=()) -> Program:
    params = {"max_hypotheses": max_hypotheses, "chunk_chars": chunk_chars,
              "condensed_chars": condensed_chars, "include_mast_examples": include_mast_examples,
              "sequential_edges": sequential_edges}
    calls: list[dict] = []

    steps = build_steps(record)
    n = len(steps)
    agents = agent_list(steps)
    task_step = task_step_index(steps, record.get("question", ""))
    chunks = chunk_steps(steps, chunk_chars, STEP_CHARS)

    # ── round 1: MAST tags and information-flow edges, one call per chunk each ──
    prompts = [messages(build_tagger_prompt(record, steps, c, include_gt=include_gt,
                                            include_examples=include_mast_examples,
                                            step_chars=STEP_CHARS)) for c in chunks]
    prompts += [messages(build_dependency_prompt(record, steps, c, include_gt=include_gt,
                                                 step_chars=STEP_CHARS)) for c in chunks]
    raws = [strip_think(r) for r in (yield prompts)]

    tags: list[dict] = []
    llm_edges: list[dict] = []
    for k, (lo, hi) in enumerate(chunks):
        parsed = parse_tags(raws[k], lo, hi)
        calls.append({"role": "tagger", "chunk": k, "steps": [lo, hi],
                      "parsed": parsed is not None, "response": raws[k]})
        for t in parsed or []:
            tags.append({**t, "chunk": k})
    for k, (lo, hi) in enumerate(chunks):
        raw = raws[len(chunks) + k]
        parsed = parse_edges(raw, lo, hi, n)
        calls.append({"role": "dependency", "chunk": k, "steps": [lo, hi],
                      "parsed": parsed is not None, "response": raw})
        llm_edges += parsed or []

    # ── backward tracing (deterministic) ──
    groups = [llm_edges, structural_edges(steps)]
    if sequential_edges == "always":
        groups.append(sequential_edges_for(n))
    edges = merge_edges(*groups)
    erf, depth, floor_applied = effective_receptive_field(edges, n, task_step=task_step)
    condensed = condense(steps, erf, depth, tags, task_step=task_step,
                         condensed_chars=condensed_chars)
    erf_doc = {"steps": condensed["kept"], "depth": {str(k): v for k, v in sorted(depth.items())},
               "floor_applied": floor_applied, "masked": condensed["masked"],
               "chars": condensed["chars"], "step_chars": condensed["step_chars"]}
    base = {"structure": [s.to_doc() for s in steps], "tags": tags,
            "graph": {"edges": edges, "n_steps": n}, "erf": erf_doc, "paper_params": params}

    # ── round 2: Strategist ──
    prompt = build_strategist_prompt(
        record, steps, condensed["text"], tags, agents,
        retrieved_patterns=retrieved_patterns, include_gt=include_gt,
        max_hypotheses=max_hypotheses, system_text=system_description(record),
        roles=agent_roles(record, steps))
    raw_strategist = strip_think((yield [messages(prompt)])[0])
    hypotheses = parse_hypotheses(raw_strategist, steps, agents, max_hypotheses)
    calls.append({"role": "strategist", "parsed": hypotheses is not None,
                  "response": raw_strategist})
    if not hypotheses:
        return _null_doc(record, calls, raw_strategist, "strategist",
                         hypotheses=[], evidence=[], verdict=None,
                         arbiter_saw_only_inconclusive=None, memory_candidate=None, **base)

    # ── round 3: one Investigator call per hypothesis ──
    prompts = [messages(build_investigator_prompt(record, steps, edges, h, include_gt=include_gt,
                                                  step_chars=STEP_CHARS))
               for h in hypotheses]
    raws = [strip_think(r) for r in (yield prompts)]
    evidence = []
    for j, (h, raw) in enumerate(zip(hypotheses, raws)):
        ev = parse_evidence(raw, h["tool"])
        evidence.append(ev)
        calls.append({"role": "investigator", "hypothesis": j, "tool": h["tool"],
                      "parsed": not ev["parse_error"], "response": raw})

    # ── round 4: Arbiter ──
    prompt, only_inconclusive = build_arbiter_prompt(record, steps, agents, hypotheses, evidence,
                                                     include_gt=include_gt)
    raw_arbiter = strip_think((yield [messages(prompt)])[0])
    verdict = parse_verdict(raw_arbiter, steps, agents)
    calls.append({"role": "arbiter", "parsed": verdict is not None, "response": raw_arbiter})

    failure_stage = None
    if verdict is None:
        j = select_fallback(hypotheses, evidence)
        h = hypotheses[j]
        verdict = {"agent": h["agent"], "agent_fixed": h["agent_fixed"], "step": h["step"],
                   "mode": h["mode"], "confidence": evidence[j]["confidence"],
                   "reason": h["rationale"], "chosen_hypothesis": j, "novel_and_robust": False,
                   "signature": {"tool": "unknown", "api": "default", "arg_schema": [],
                                 "context_slots": [], "err_family": h["mode"]},
                   "guard": "", "fallback": True}
        failure_stage = "arbiter"

    chosen = verdict["chosen_hypothesis"]
    chosen_evidence = evidence[chosen] if 0 <= chosen < len(evidence) else None
    return {
        "predicted_agent": verdict["agent"],
        "predicted_step": verdict["step"],
        "raw": raw_arbiter,
        "calls": calls,
        "error_family": verdict["mode"],
        "error_reason": verdict["reason"],
        "confidence": verdict["confidence"],
        "hypotheses": hypotheses,
        "evidence": evidence,
        "verdict": verdict,
        "arbiter_saw_only_inconclusive": only_inconclusive,
        "memory_candidate": memory_candidate(verdict, chosen_evidence, MEMORY_TAU),
        "failure_stage": failure_stage,
        **base,
    }


def sequential_edges_for(n: int) -> list[dict]:
    return sequential_edges(n)


METHODS = {METHOD_PAPER: errorprobe_paper_program}
