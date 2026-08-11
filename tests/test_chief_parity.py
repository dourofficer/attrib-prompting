"""Execution-level parity between the CHIEF stages and the vendored pipeline.

Loads ``vendored/CHIEF/CHIEF.py`` (stubbing the dependencies it imports at module
scope — it builds a ``RAGRetriever`` on import), monkeypatches its ``call_model``
to capture the prompt and return a canned response, then drives all six vendored
``stepN_*`` functions. Each one's prompt must be byte-identical to ours, and each
one's parsed structure deep-equal to ours.

This is the test ``baselines/chief/README.md`` promises: the prompts and regexes
in ``stages.py`` are copies, and a copy is only trustworthy while something keeps
checking it against the original.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from baselines.chief import stages

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_CHIEF = REPO_ROOT / "vendored/CHIEF/CHIEF.py"

# Hits the stubbed retriever returns — one per source, so format_rag_blocks
# exercises both the GAIA branch and the AssistantBench branch.
HITS = [
    {"source": "GAIA", "score": 0.9, "question": "How many albums?",
     "steps": "1. Search.\n2. Count."},
    {"source": "AssistantBench", "score": 0.8,
     "text": "Task: Find a gym.\nExplanation: Use Maps — then filter…"},
]

HISTORY = [
    {"role": "Orchestrator (thought)", "content": "Plan: “search” the web — then compute…"},
    {"role": "WebSurfer", "content": "I found • the population is 8,336,000\xa0people 中文"},
    {"role": "Assistant", "content": "Final answer: ‘16,672,000’"},
]
QUESTION = "What is twice the “population” of the city?"
# Deliberately absent from HISTORY, so the without-GT test can assert the answer
# is nowhere in the prompt rather than merely absent from one sentence.
GROUND_TRUTH = "sixteen-million-six-hundred-seventy-two-thousand"

# Canned stage outputs, shaped so every parser has real structure to chew on.
OUT1 = """The Subtask Name: Locate the population
Step Range: 0-1
The Oracle: The population figure is retrieved from a reliable source
Evidence: WebSurfer reports 8,336,000
Loop Info:
{
  is_loop_related: false,
  loop_role: none,
  loop_group_id: null,
  reversibility: reversible,
  loop_risk_score: 0.1
}

The Subtask Name: Double the figure
Step Range: 2-2
The Oracle: The figure is multiplied by two
Evidence: Assistant outputs the final answer
Loop Info:
{
  is_loop_related: true,
  loop_role: entry,
  loop_group_id: L1,
  reversibility: irreversible,
  loop_risk_score: 0.7
}
"""

OUT2 = """From: S1
To: S2
Type: data_dependency
Strength: 0.9
Explanation: The doubling needs the retrieved population.
Data_Transfer: {
  upstream_output:
    - data_item: "population"
      data_type: "numeric"
      steps: [1]
  downstream_usage:
    - data_item: "population"
      data_type: "numeric"
      steps: [2]
  consistency_score: 0.95
}
Failure Modes:
  - type: data_issue
    description: A wrong population propagates into the product.
    severity: high
    if_from_wrong: the product is wrong
    likely_effect_on_to: incorrect final answer
"""

OUT3 = """The Subtask Name: Locate the population
Agents:
- Agent: WebSurfer
-- Action: Searched the web
-- Observation: A page with the figure
-- Thought: This source looks reliable
-- Result: 8,336,000
Data_Flow:
- from_step: 1
  to_step: 2
  source_agent: WebSurfer
  target_agent: Assistant
  data_item: "population"
  data_type: "numeric"
  transformation: "passed through"
  correctness: correct
  confidence: 0.9

The Subtask Name: Double the figure
Agents:
- Agent: Assistant
-- Action: Multiplied by two
-- Observation: The population figure
-- Thought: Twice the population is the answer
-- Result: 16,672,000
Data_Flow: []
"""

OUT4 = """The Subtask Name: Locate the population
Agents_edges:
  - From_agent: WebSurfer
    To_agent: Assistant
    agent_dependency_type: obs_dependency
    agent_strength: 0.8
    agent_explanation: The assistant consumes the surfer's figure.
    agent_failure_modes:
      - type: data_issue
        description: A mis-read figure is passed on.
        severity: high
        if_from_agent_wrong: the product is wrong
        likely_effect_on_to_agent: incorrect final answer
"""

OUT5 = """Candidate Error Subtasks: [S1, S2]
Candidate Error Agents: [WebSurfer, Assistant]
Candidate Error Steps:
- step_id: 1
  agent_in_step: [WebSurfer]
  loop_issue: {is_in_loop: false, loop_role: none, loop_group_id: null}
  data_issue: {has_issue: true, data_item: "population", source_step: 1, consistency_score: 0.4, explanation: misread the page}
  irrecoverability_issue: {is_irrecoverable: true, reason: the figure is never rechecked}
  impact: {affected_steps: [2], impact_score: 0.9}
  information: {input: search results, output: 8,336,000}
  confidence: 0.8
- step_id: 2
  agent_in_step: [Assistant]
  loop_issue: {is_in_loop: true, loop_role: entry, loop_group_id: L1}
  data_issue: {has_issue: false, data_item: "product", source_step: 2, consistency_score: 0.9, explanation: none}
  irrecoverability_issue: {is_irrecoverable: false, reason: could be recomputed}
  impact: {affected_steps: [], impact_score: 0.2}
  information: {input: population, output: 16,672,000}
  confidence: 0.5
"""

OUT6 = """Agent Name: WebSurfer (thought)
Step Number: 1
Reason for Mistake: The surfer mis-read the population, and every later step
inherited the wrong figure.
"""


def _stub(name: str, **attrs) -> types.ModuleType | None:
    """Register a stand-in module unless the real one imports."""
    if name in sys.modules:
        return sys.modules[name]
    try:
        __import__(name)
        return sys.modules[name]
    except ImportError:
        pass
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _StubRetriever:
    """Stands in for the vendored RAGRetriever, which is built at import time."""

    def search(self, query, top_k=2):
        return HITS


@pytest.fixture(scope="module")
def vendored():
    """Import vendored/CHIEF/CHIEF.py with its module-scope side effects neutered."""
    _stub("tqdm", tqdm=lambda it, **kw: it)
    _stub("openai", OpenAI=lambda *a, **kw: None)
    _stub("dotenv", load_dotenv=lambda *a, **kw: None)
    rag_pkg = types.ModuleType("rag")
    rag_search = types.ModuleType("rag.rag_search")
    rag_search.RAGRetriever = _StubRetriever
    rag_pkg.rag_search = rag_search
    sys.modules.setdefault("rag", rag_pkg)
    sys.modules.setdefault("rag.rag_search", rag_search)

    spec = importlib.util.spec_from_file_location("vendored_chief", VENDORED_CHIEF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def drive(vendored, monkeypatch):
    """Run one vendored stage, returning (prompt_it_built, value_it_parsed)."""
    def _drive(fn, response, *args):
        seen = {}

        def fake_call_model(prompt):
            seen["prompt"] = prompt
            return response

        monkeypatch.setattr(vendored, "call_model", fake_call_model)
        parsed, raw = fn(*args)
        assert raw == response.strip()  # every vendored stage strips before parsing
        return seen["prompt"], parsed

    return _drive


@pytest.fixture()
def rag_text():
    return stages.format_rag_blocks(HITS)


def test_system_prompt_matches(vendored):
    # The vendored system prompt is a literal inside call_llm's messages list.
    import inspect
    assert stages.SYSTEM_PROMPT in inspect.getsource(vendored.call_llm)


def test_rag_block_formatting_matches(vendored, drive, rag_text):
    """format_rag_blocks reproduces the block step 1 builds inline."""
    prompt, _ = drive(vendored.step1_generate_subtasks, OUT1,
                      HISTORY, QUESTION, GROUND_TRUTH)
    assert rag_text in prompt
    assert "[RAG Example 1]" in rag_text and "[RAG Example 2]" in rag_text


def test_step1_parity(vendored, drive, rag_text):
    prompt, parsed = drive(vendored.step1_generate_subtasks, OUT1,
                           HISTORY, QUESTION, GROUND_TRUTH)
    assert stages.build_step1(HISTORY, QUESTION, GROUND_TRUTH, rag_text) == prompt
    assert stages.parse_step1(OUT1) == parsed
    assert len(parsed) == 2  # the canned output really did parse


def test_step2_parity(vendored, drive):
    subtasks = stages.parse_step1(OUT1)
    prompt, parsed = drive(vendored.step2_generate_subtasks_edges, OUT2,
                           HISTORY, QUESTION, GROUND_TRUTH, subtasks)
    assert stages.build_step2(HISTORY, QUESTION, GROUND_TRUTH, subtasks) == prompt
    assert stages.parse_step2(OUT2) == parsed
    assert parsed["subtasks_edges"]


def test_step3_parity(vendored, drive):
    subtasks = stages.parse_step1(OUT1)
    prompt, parsed = drive(vendored.step3_generate_agents, OUT3,
                           HISTORY, QUESTION, GROUND_TRUTH, subtasks)
    assert stages.build_step3(HISTORY, QUESTION, GROUND_TRUTH, subtasks) == prompt
    assert stages.parse_step3(OUT3, subtasks) == parsed
    assert any(s["agents"] for s in parsed)


def test_step4_parity(vendored, drive):
    subtasks = stages.parse_step1(OUT1)
    subtasks_agents = stages.parse_step3(OUT3, subtasks)
    prompt, parsed = drive(vendored.step4_generate_agents_edges, OUT4,
                           HISTORY, QUESTION, GROUND_TRUTH, subtasks_agents)
    assert stages.build_step4(HISTORY, QUESTION, GROUND_TRUTH, subtasks_agents) == prompt
    assert stages.parse_step4(OUT4) == parsed
    assert parsed


def _dag():
    subtasks = stages.parse_step1(OUT1)
    subtasks_agents = stages.parse_step3(OUT3, subtasks)
    return stages.build_dag_graph(subtasks_agents,
                                  stages.parse_step2(OUT2)["subtasks_edges"],
                                  stages.parse_step4(OUT4))


def test_dag_assembly_matches(vendored):
    subtasks = stages.parse_step1(OUT1)
    subtasks_agents = stages.parse_step3(OUT3, subtasks)
    edges = stages.parse_step2(OUT2)["subtasks_edges"]
    agent_edges = stages.parse_step4(OUT4)
    assert stages.build_dag_graph(subtasks_agents, edges, agent_edges) == \
        vendored.build_dag_graph(subtasks_agents, edges, agent_edges)


def test_step5_parity(vendored, drive):
    dag = _dag()
    prompt, parsed = drive(vendored.step5_predict_candidate_set, OUT5,
                           HISTORY, QUESTION, GROUND_TRUTH, dag)
    assert stages.build_step5(HISTORY, QUESTION, GROUND_TRUTH, dag) == prompt
    assert stages.parse_step5(OUT5) == parsed
    assert len(parsed["candidate_error_steps"]) == 2


def test_step6_parity(vendored, monkeypatch):
    # Stage 6 is the one vendored stage returning a bare dict, not (parsed, raw).
    dag = _dag()
    candidate_set = stages.parse_step5(OUT5)
    seen = {}

    def fake_call_model(prompt):
        seen["prompt"] = prompt
        return OUT6

    monkeypatch.setattr(vendored, "call_model", fake_call_model)
    parsed = vendored.step6_predict_final_answer(
        HISTORY, QUESTION, GROUND_TRUTH, candidate_set, dag)

    assert stages.build_step6(HISTORY, QUESTION, GROUND_TRUTH,
                              candidate_set, dag) == seen["prompt"]
    assert stages.parse_step6(OUT6) == parsed
    # The vendored split("(") cleanup is load-bearing on hand-crafted roles.
    assert parsed["final"]["mistake_agent"] == "WebSurfer"
    assert parsed["final"]["mistake_step"] == 1


def test_without_gt_drops_only_the_answer(rag_text):
    """--gt without removes the answer sentence and nothing else."""
    builders = [
        (stages.build_step1, (HISTORY, QUESTION, GROUND_TRUTH, rag_text)),
        (stages.build_step2, (HISTORY, QUESTION, GROUND_TRUTH, [])),
        (stages.build_step3, (HISTORY, QUESTION, GROUND_TRUTH, [])),
        (stages.build_step4, (HISTORY, QUESTION, GROUND_TRUTH, [])),
        (stages.build_step5, (HISTORY, QUESTION, GROUND_TRUTH, {})),
        (stages.build_step6, (HISTORY, QUESTION, GROUND_TRUTH, {}, {})),
    ]
    sentence = f"The correct answer for the problem is: {GROUND_TRUTH}\n"
    for build, args in builders:
        with_gt = build(*args)
        without = build(*args, include_gt=False)
        assert sentence in with_gt
        assert GROUND_TRUTH not in without
        # Removing exactly the sentence (keeping its blank line) recovers it.
        assert with_gt.replace(sentence, "", 1) == without
