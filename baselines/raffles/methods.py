"""RAFFLES as a generator program — a Judge-Evaluator loop per trajectory.

One iteration is two rounds: the Judge proposes a candidate (agent, step) with
one rationale per decisive-fault criterion, then the three Evaluators verify
those rationales in a single yielded round — three prompts at once, which the
API backend fans out concurrently (the parallelization the paper's Appendix D
uses to halve latency). A rule-based fourth check (no LLM call) validates the
candidate against the log, and the four confidences sum to C out of 400.

The loop is Algorithm 1: iterate while C stays at or below the threshold and
the iteration budget ``K = max_iters`` is not exhausted (K=0 still runs one
full Judge+Evaluator pass — the paper's "RAFFLES K=0" rows), feeding every
prior candidate and critique back to the Judge. The prediction is the
candidate with the highest C; on ties the latest wins, since the Judge saw
strictly more feedback when it produced it.

Failure handling follows chief: an unparseable Judge response costs that
iteration (no Evaluator round; the Judge is told its output did not parse and
tries again), and if every iteration fails the trajectory ends with a null
prediction. Backend errors never reach this module — the runner calls
``generate`` outside the program and skips the trajectory without writing, so
a rerun resumes it (GUIDE.md "Resume").
"""
from __future__ import annotations

from typing import Generator

from baselines.prompting.methods import strip_think

from .prompts import (
    CRITERIA,
    build_evaluator_prompt,
    build_judge_prompt,
    messages,
    normalize_step,
    parse_evaluator,
    parse_judge,
    rule_check,
)

METHOD = "raffles"

# Paper defaults: K=2 (main tables) and the 350/400 termination threshold.
MAX_ITERS = 2
THRESHOLD = 350

Program = Generator[list[list[dict]], list[str], dict]


def raffles_program(record: dict, *, max_iters: int = MAX_ITERS,
                    threshold: int = THRESHOLD,
                    include_gt: bool = False) -> Program:
    """Yield Judge and Evaluator rounds in turn; return the output doc."""
    history = record["history"]
    calls: list[dict] = []       # every LLM response, in issue order
    iterations: list[dict] = []  # per-iteration audit: candidate, scores, C
    feedback: list[dict] = []    # memory H, serialized into the next Judge prompt
    best: tuple[int, dict, str] | None = None  # (C, candidate, judge raw)

    for k in range(max_iters + 1):
        raw_judge = strip_think(
            (yield [messages(build_judge_prompt(record, feedback,
                                                include_gt=include_gt))])[0])
        calls.append({"iteration": k, "role": "judge", "response": raw_judge})

        try:
            candidate = parse_judge(raw_judge)
        except ValueError as exc:
            iterations.append({"iteration": k, "candidate": None,
                               "total_confidence": None, "error": str(exc)})
            feedback.append({"iteration": k, "answer": raw_judge, "evals": [
                {"criterion": 0, "confidence": 0,
                 "reason": "This answer was not in the required json format. "
                           "Answer strictly in the Task Output format."}]})
            continue

        eval_raws = yield [messages(build_evaluator_prompt(p, record, candidate,
                                                           include_gt=include_gt))
                           for p in sorted(CRITERIA)]
        evals = []
        for p, raw in zip(sorted(CRITERIA), eval_raws):
            raw = strip_think(raw)
            calls.append({"iteration": k, "role": "evaluator", "criterion": p,
                          "response": raw})
            reason, confidence = parse_evaluator(raw)
            evals.append({"criterion": p, "reason": reason, "confidence": confidence})
        reason, confidence = rule_check(history, candidate)
        evals.append({"criterion": 4, "reason": reason, "confidence": confidence})

        total = sum(e["confidence"] for e in evals)
        iterations.append({"iteration": k, "candidate": candidate,
                           "confidences": evals, "total_confidence": total})
        if best is None or total >= best[0]:
            best = (total, candidate, raw_judge)
        if total > threshold:
            break
        feedback.append({"iteration": k, "answer": raw_judge, "evals": evals})

    if best is None:  # every Judge round failed to parse
        return {"predicted_agent": None, "predicted_step": None, "raw": None,
                "calls": calls, "confidence": None,
                "n_iterations": len(iterations), "iterations": iterations}

    total, candidate, raw_judge = best
    return {"predicted_agent": candidate["agent_name"],
            "predicted_step": normalize_step(candidate["step_number"]),
            "raw": raw_judge,
            "calls": calls,
            "confidence": total,
            "n_iterations": len(iterations),
            "iterations": iterations}


METHODS = {"raffles": raffles_program}
