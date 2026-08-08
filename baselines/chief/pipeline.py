"""Columnar (batched) CHIEF pipeline — the fast path.

Because CHIEF is exactly six sequential LLM calls per trajectory (one per stage),
we run it *columnar*: batch stage 1 across ALL trajectories in one
``engine.generate`` call, parse each output, then batch stage 2, and so on. Every
trajectory's own stage-N prompt still depends only on its own parsed stage-(N-1)
result, so this is identical in effect to the per-sample :mod:`reference` path —
it just lets vLLM batch across trajectories (mirroring how
``baselines.prompting.methods`` batch their per-step calls).

Builders/parsers are shared with :mod:`reference` via :mod:`baselines.chief.stages`,
which is what guarantees the two paths agree.
"""
from __future__ import annotations

from .engine import PromptEngine, strip_think
from .stages import (
    build_dag_graph,
    build_step1, build_step2, build_step3, build_step4, build_step5, build_step6,
    messages,
    parse_step1, parse_step2, parse_step3, parse_step4, parse_step5, parse_step6,
)


def _fail(state: dict, err: Exception) -> None:
    state["alive"] = False
    state["pred"]["raw"] = f"[chief-error] {type(err).__name__}: {err}"


def _batch(engine: PromptEngine, states: list[dict], build_fn) -> dict[int, str]:
    """Build prompts for every alive state, run one batched generate, return {idx: text}."""
    idxs = [i for i, s in enumerate(states) if s["alive"]]
    prompts = []
    keep = []
    for i in idxs:
        try:
            prompts.append(messages(build_fn(states[i])))
            keep.append(i)
        except Exception as e:  # a builder should never throw, but be safe
            _fail(states[i], e)
    raws = engine.generate(prompts) if prompts else []
    return {i: strip_think(raw) for i, raw in zip(keep, raws)}


def run(records: list[dict], engine: PromptEngine, rag_texts=None) -> list[dict]:
    if rag_texts is None:
        rag_texts = [None] * len(records)

    states = [{
        "rec": r,
        "rag": rag_texts[i],
        "alive": True,
        "pred": {"predicted_agent": None, "predicted_step": None, "raw": None},
    } for i, r in enumerate(records)]

    def h(s):  return s["rec"]["history"]
    def q(s):  return s["rec"].get("question", "")
    def gt(s): return s["rec"].get("ground_truth", "")

    # Stage 1 — subtasks
    for i, raw in _batch(engine, states, lambda s: build_step1(h(s), q(s), gt(s), s["rag"])).items():
        try:
            states[i]["subtasks"] = parse_step1(raw)
        except Exception as e:
            _fail(states[i], e)

    # Stage 2 — subtask edges
    for i, raw in _batch(engine, states, lambda s: build_step2(h(s), q(s), gt(s), s["subtasks"])).items():
        try:
            states[i]["edges"] = parse_step2(raw)
        except Exception as e:
            _fail(states[i], e)

    # Stage 3 — agents within subtasks
    for i, raw in _batch(engine, states, lambda s: build_step3(h(s), q(s), gt(s), s["subtasks"])).items():
        try:
            states[i]["subtasks_agents"] = parse_step3(raw, states[i]["subtasks"])
        except Exception as e:
            _fail(states[i], e)

    # Stage 4 — agent edges within subtasks
    for i, raw in _batch(engine, states, lambda s: build_step4(h(s), q(s), gt(s), s["subtasks_agents"])).items():
        try:
            states[i]["agent_edges"] = parse_step4(raw)
        except Exception as e:
            _fail(states[i], e)

    # DAG assembly (no LLM call)
    for s in states:
        if s["alive"]:
            s["dag"] = build_dag_graph(
                s["subtasks_agents"], s["edges"]["subtasks_edges"], s["agent_edges"])

    # Stage 5 — candidate error set
    for i, raw in _batch(engine, states, lambda s: build_step5(h(s), q(s), gt(s), s["dag"])).items():
        try:
            states[i]["candidate_set"] = parse_step5(raw)
        except Exception as e:
            _fail(states[i], e)

    # Stage 6 — final single-step attribution
    for i, raw in _batch(engine, states, lambda s: build_step6(h(s), q(s), gt(s), s["candidate_set"], s["dag"])).items():
        try:
            final = parse_step6(raw)
            states[i]["pred"] = {
                "predicted_agent": final["final"]["mistake_agent"],
                "predicted_step": final["final"]["mistake_step"],
                "raw": final["raw"],
            }
        except Exception as e:
            _fail(states[i], e)

    return [s["pred"] for s in states]
