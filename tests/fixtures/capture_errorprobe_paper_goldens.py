"""One-off capture of golden prompt strings for the ErrorProbe paper mode.

Run from the repo root:  python tests/fixtures/capture_errorprobe_paper_goldens.py

The paper mode has no vendored counterpart, so nothing can be diffed against
upstream code: its prompts are built from the paper's description (Section 4)
and the MAST annotator. These goldens pin the resulting bytes.
``tests/test_errorprobe_paper_pipeline.py`` asserts the builders keep
reproducing them character for character, and separately asserts the
sentences that carry each prompt's contract, so a fixture regeneration cannot
silently change what the prompts ask for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from baselines.errorprobe.paper.graph import (  # noqa: E402
    build_dependency_prompt, condense, effective_receptive_field, merge_edges, structural_edges,
)
from baselines.errorprobe.paper.structure import (  # noqa: E402
    agent_list, agent_roles, build_steps, task_step_index,
)
from baselines.errorprobe.paper.tagger import build_tagger_prompt, chunk_steps  # noqa: E402
from baselines.errorprobe.paper.team import (  # noqa: E402
    build_arbiter_prompt, build_investigator_prompt, build_strategist_prompt,
)

QUESTION = "What is twice the population of Springfield?"
HISTORY = [
    {"role": "human", "content": QUESTION},
    {"role": "Orchestrator (thought)", "content": (
        "Initial plan:\n\nWe are working to address the following user request:\n\n"
        f"{QUESTION}\n\n\nTo answer this request we have assembled the following team:\n\n"
        "Assistant: A helpful and general-purpose AI assistant with Python skills.\n"
        "WebSurfer: A helpful assistant with access to a web browser.\n"
        "ComputerTerminal: A computer terminal that runs Python scripts.\n\n\n"
        "Here is an initial fact sheet to consider:\n1. GIVEN OR VERIFIED FACTS\n- none")},
    {"role": "Orchestrator (-> WebSurfer)",
     "content": "Please look up the current population of Springfield and report the number."},
    {"role": "WebSurfer", "content": (
        "I typed 'population of Springfield' into the browser search bar.\n\n"
        "Here is a screenshot of the results page. The first result says the population "
        "of Springfield, Illinois is 114,394 (2020 census).")},
    {"role": "Orchestrator (thought)", "content": (
        "Updated Ledger:\n{\n  \"is_request_satisfied\": {\"reason\": \"We have a population figure "
        "but have not doubled it.\", \"answer\": false},\n  \"next_speaker\": {\"reason\": \"Compute.\", "
        "\"answer\": \"ComputerTerminal\"}\n}")},
    {"role": "Orchestrator (-> ComputerTerminal)",
     "content": "Run this:\n```python\nprint(8336000 * 2)\n```"},
    {"role": "ComputerTerminal", "content": "exitcode: 0 (execution succeeded)\nCode output: 16672000"},
    {"role": "Assistant", "content": "FINAL ANSWER: 16672000"},
    {"role": "Orchestrator (termination condition)", "content": "TERMINATE"},
]
RECORD = {
    "id": "7", "filename": "7.json", "question_id": "golden-7",
    "question": QUESTION, "ground_truth": "228,788", "history": HISTORY,
    "gold_agent": "Orchestrator", "gold_step": 5,
    "system": "A Magentic-One style team: an Orchestrator plans and delegates to specialists.",
}
TAGS = [
    {"step": 5, "mode": "ignored_input", "evidence": "The code doubles 8336000, not the 114,394 WebSurfer reported.", "severity": 0.8, "chunk": 0},
    {"step": 7, "mode": "incorrect_verification", "evidence": "The final answer is stated without checking the figure.", "severity": 0.5, "chunk": 0},
]
LLM_EDGES = [
    {"from": 3, "to": 5, "kind": "cites", "source": "llm", "why": "the code should use the reported figure"},
    {"from": 6, "to": 7, "kind": "cites", "source": "llm", "why": "the answer repeats the output"},
]
HYPOTHESES = [
    {"step": 5, "agent": "Orchestrator", "agent_fixed": False, "mode": "ignored_input",
     "rationale": "The Orchestrator doubled a number WebSurfer never reported.",
     "tool": "code_exec", "check": "Compare the literal in the code with the figure at step 3.",
     "related_steps": [3, 6]},
    {"step": 3, "agent": "WebSurfer", "agent_fixed": False, "mode": "fail_task_spec",
     "rationale": "WebSurfer may have reported the wrong Springfield.",
     "tool": "logic_probe", "check": "Does the task name which Springfield?",
     "related_steps": [0]},
]
EVIDENCE_CONCLUSIVE = [
    {"tool": "code_exec", "supports_hypothesis": True, "conclusive": True, "confidence": 0.9,
     "discrepancy": "- print(114394 * 2)\n+ print(8336000 * 2)", "parse_error": False,
     "execution_log": "print(8336000 * 2) -> 16672000", "expected_output": "228788",
     "observed_output": "16672000", "notes": ""},
    {"tool": "logic_probe", "supports_hypothesis": False, "conclusive": False, "confidence": 0.2,
     "discrepancy": "", "parse_error": False,
     "preconditions": [{"condition": "The task names a Springfield", "source_steps": [0], "holds": False, "evidence": "The task says only 'Springfield'."}],
     "postconditions": []},
]
EVIDENCE_INCONCLUSIVE = [
    {"tool": "code_exec", "supports_hypothesis": False, "conclusive": False, "confidence": 0.0,
     "discrepancy": "", "parse_error": True},
    {"tool": "logic_probe", "supports_hypothesis": True, "conclusive": False, "confidence": 0.3,
     "discrepancy": "", "parse_error": False, "preconditions": [], "postconditions": [],
     "bare_assertion": True},
]
PATTERNS = [{"mode": "ignored_input", "guard": "Before computing, verify the operand appears in a specialist's report."}]


def build_all() -> dict[str, str]:
    steps = build_steps(RECORD)
    agents = agent_list(steps)
    chunk = chunk_steps(steps, 12000, 1200)[0]
    edges = merge_edges(LLM_EDGES, structural_edges(steps))
    task_step = task_step_index(steps, QUESTION)
    erf, depth, _ = effective_receptive_field(edges, len(steps), task_step=task_step)
    condensed = condense(steps, erf, depth, TAGS, task_step=task_step)
    common = dict(record=RECORD, steps=steps)
    arbiter_ok, _ = build_arbiter_prompt(RECORD, steps, agents, HYPOTHESES, EVIDENCE_CONCLUSIVE)
    arbiter_none, _ = build_arbiter_prompt(RECORD, steps, agents, HYPOTHESES, EVIDENCE_INCONCLUSIVE)
    return {
        "tagger": build_tagger_prompt(chunk=chunk, **common),
        "tagger_with_gt": build_tagger_prompt(chunk=chunk, include_gt=True, **common),
        "dependency": build_dependency_prompt(chunk=chunk, **common),
        "strategist": build_strategist_prompt(
            RECORD, steps, condensed["text"], TAGS, agents,
            system_text=RECORD["system"], roles=agent_roles(RECORD, steps)),
        "strategist_with_patterns": build_strategist_prompt(
            RECORD, steps, condensed["text"], TAGS, agents, retrieved_patterns=PATTERNS,
            roles=agent_roles(RECORD, steps)),
        "investigator_code_exec": build_investigator_prompt(RECORD, steps, edges, HYPOTHESES[0]),
        "investigator_logic_probe": build_investigator_prompt(RECORD, steps, edges, HYPOTHESES[1]),
        "arbiter": arbiter_ok,
        "arbiter_all_inconclusive": arbiter_none,
        "condensed_trace": condensed["text"],
    }


if __name__ == "__main__":
    out = REPO_ROOT / "tests" / "fixtures" / "errorprobe_paper_golden_prompts.json"
    out.write_text(json.dumps(build_all(), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
