"""Per-sample CHIEF pipeline — the faithful reference path.

This mirrors the vendored ``process_sample`` (CHIEF.py:1007-1057) one trajectory at
a time, six sequential LLM calls per trajectory, using the shared builders/parsers
in :mod:`baselines.chief.stages`. The only substantive change vs. the original is
that the LLM call goes through the local vLLM :class:`PromptEngine` (with
``strip_think``) instead of the OpenAI API.

Use this path for correctness checks against the batched :mod:`pipeline` and as the
literal, easy-to-audit reproduction of CHIEF. It is *not* meant for full sweeps
(one prompt per ``generate`` call underutilizes the GPU).
"""
from __future__ import annotations

from .engine import PromptEngine, strip_think
from .stages import (
    build_dag_graph,
    build_step1, build_step2, build_step3, build_step4, build_step5, build_step6,
    messages,
    parse_step1, parse_step2, parse_step3, parse_step4, parse_step5, parse_step6,
)


def _call(engine: PromptEngine, prompt: str) -> str:
    return strip_think(engine.generate([messages(prompt)])[0])


def run_one(record: dict, engine: PromptEngine, rag_text=None) -> dict:
    """Run all six stages for one trajectory; return a prediction dict."""
    history = record["history"]
    question = record.get("question", "")
    ground_truth = record.get("ground_truth", "")

    subtasks = parse_step1(_call(engine, build_step1(history, question, ground_truth, rag_text)))
    edges = parse_step2(_call(engine, build_step2(history, question, ground_truth, subtasks)))
    subtasks_agents = parse_step3(_call(engine, build_step3(history, question, ground_truth, subtasks)), subtasks)
    agent_edges = parse_step4(_call(engine, build_step4(history, question, ground_truth, subtasks_agents)))

    dag_graph = build_dag_graph(subtasks_agents, edges["subtasks_edges"], agent_edges)

    candidate_set = parse_step5(_call(engine, build_step5(history, question, ground_truth, dag_graph)))
    final = parse_step6(_call(engine, build_step6(history, question, ground_truth, candidate_set, dag_graph)))

    return {
        "predicted_agent": final["final"]["mistake_agent"],
        "predicted_step": final["final"]["mistake_step"],
        "raw": final["raw"],
    }


def run(records: list[dict], engine: PromptEngine, rag_texts=None) -> list[dict]:
    """Sequentially attribute every record. Errors degrade to a null prediction."""
    if rag_texts is None:
        rag_texts = [None] * len(records)
    preds = []
    for i, record in enumerate(records):
        try:
            preds.append(run_one(record, engine, rag_texts[i]))
        except Exception as e:  # never let one trajectory kill the run
            preds.append({"predicted_agent": None, "predicted_step": None,
                          "raw": f"[chief-error] {type(e).__name__}: {e}"})
    return preds
