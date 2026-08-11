"""CHIEF as a generator program — six LLM calls per trajectory, one per stage.

The vendored ``process_sample`` (``vendored/CHIEF/CHIEF.py:1007``) runs the six
stages as a straight-line sequence of blocking calls. Expressing the same
sequence as a generator that *yields* each stage's prompt and resumes with the
response buys both execution shapes for free (GUIDE.md "Execution"):
:func:`baselines.shared.runner.run_batched` gathers stage N across every live
trajectory into one vLLM call — the columnar batching CHIEF wants on a local
GPU — while :func:`~baselines.shared.runner.run_streaming` drives whole
trajectories independently on an API backend, writing each as it finishes.

The stage order is the vendored one, and stage 3 consumes stage 1 (not stage 2),
exactly as in ``CHIEF.py:1015-1027``::

    1 subtasks (+oracles, +loop info; RAG exemplar injected)
    2 subtask edges
    3 agents per subtask (OTAR + data flow)
    4 agent edges                    → build_dag_graph, no LLM call
    5 candidate error set
    6 counterfactual attribution     → Agent Name / Step Number / Reason

Every response is kept in ``calls`` as ``{"stage": n, "response": ...}``, so a
finished trajectory carries the full six-call transcript. The vendored code
stored stages 1-5 and dropped stage 6's text; we keep all six.

A parse failure ends the trajectory with a null prediction and the transcript so
far — the vendored behaviour, which counts such a sample as wrong rather than
dropping it. Backend failures are different: they raise, and the runner skips
the trajectory *without writing*, so a rerun picks it up (GUIDE.md "Resume").
"""
from __future__ import annotations

import re
from typing import Generator

from baselines.prompting.methods import strip_think as _strip_think

from .stages import (
    build_dag_graph,
    build_step1, build_step2, build_step3, build_step4, build_step5, build_step6,
    messages,
    parse_step1, parse_step2, parse_step3, parse_step4, parse_step5, parse_step6,
)

METHOD = "chief"

# Extra safety net on top of prompting's strip_think: some reasoning backbones
# emit an opener with no closer, or leftover bare tags. CHIEF's stage parsers are
# strict regexes over plain text and must never see a reasoning block.
_DANGLING_OPEN = re.compile(r"<think>(?!.*</think>).*\Z", re.DOTALL)
_BARE_TAGS = re.compile(r"</?think>")


def strip_think(text: str) -> str:
    """Remove reasoning traces so CHIEF's regex parsers see only the answer."""
    out = _strip_think(text)
    out = _DANGLING_OPEN.sub("", out)
    out = _BARE_TAGS.sub("", out)
    return out.strip()


Program = Generator[list[list[dict]], list[str], dict]


def chief_program(record: dict, *, rag_text: str | None = None,
                  include_gt: bool = True) -> Program:
    """Yield each stage's messages in turn; return the output doc.

    ``rag_text`` is the precomputed stage-1 exemplar block (see
    :mod:`baselines.chief.ragprep`); ``None`` omits the retrieved-example
    section. Retrieval itself never happens here — the six calls read a
    prepared string, so no FAISS index or embedding model is loaded at
    detection time.
    """
    history = record["history"]
    question = record.get("question", "")
    ground_truth = record.get("ground_truth", "")
    calls: list[dict] = []

    def ask(stage: int, prompt: str):
        """Yield one stage's prompt, log the response, hand it back."""
        raw = strip_think((yield [messages(prompt)])[0])
        calls.append({"stage": stage, "response": raw})
        return raw

    def done(agent, step, raw):
        return {
            "predicted_agent": agent,
            "predicted_step": step,
            "raw": raw,
            "calls": calls,
        }

    try:
        raw1 = yield from ask(1, build_step1(history, question, ground_truth,
                                             rag_text, include_gt))
        subtasks = parse_step1(raw1)

        raw2 = yield from ask(2, build_step2(history, question, ground_truth,
                                             subtasks, include_gt))
        edges = parse_step2(raw2)

        raw3 = yield from ask(3, build_step3(history, question, ground_truth,
                                             subtasks, include_gt))
        subtasks_agents = parse_step3(raw3, subtasks)

        raw4 = yield from ask(4, build_step4(history, question, ground_truth,
                                             subtasks_agents, include_gt))
        agent_edges = parse_step4(raw4)

        dag = build_dag_graph(subtasks_agents, edges["subtasks_edges"], agent_edges)

        raw5 = yield from ask(5, build_step5(history, question, ground_truth,
                                             dag, include_gt))
        candidate_set = parse_step5(raw5)

        raw6 = yield from ask(6, build_step6(history, question, ground_truth,
                                             candidate_set, dag, include_gt))
        final = parse_step6(raw6)
    except Exception as exc:  # noqa: BLE001 — a malformed stage output, not a backend error
        # Backend errors never land here: the runner calls generate() outside the
        # program, so a failed call skips the trajectory instead of writing it.
        return done(None, None, f"[chief-error] after stage {len(calls)}: "
                                f"{type(exc).__name__}: {exc}")

    return done(final["final"]["mistake_agent"],
                final["final"]["mistake_step"],
                final["raw"])


METHODS = {"chief": chief_program}
