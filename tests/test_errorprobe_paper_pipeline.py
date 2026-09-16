"""ErrorProbe paper-mode tests (golden prompts, deterministic stages, dummy backend).

The paper mode has no vendored counterpart, so there is no parity test. Three
things stand in for it: golden fixtures that pin every prompt's bytes
(``tests/fixtures/errorprobe_paper_golden_prompts.json``), phrase pins on the
sentences that carry each prompt's contract, and unit tests on the
deterministic stages (the action classifier, the structural edges, the
backward search, the masking, every parser's failure path). The program is
driven end to end with a responder keyed on the prompt's opening line, so
responses stay a deterministic function of the prompt under any concurrency.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
sys.path.insert(0, str(FIXTURES))

import capture_errorprobe_paper_goldens as golden  # noqa: E402

from baselines.errorprobe.paper import graph, structure, tagger, team  # noqa: E402
from baselines.errorprobe.paper.program import (  # noqa: E402
    METHOD_PAPER, PAPER_DEFAULTS, errorprobe_paper_program,
)
from baselines.errorprobe.prompts import MAST_FAMILIES  # noqa: E402
from baselines.shared.backends.dummy import DummyBackend  # noqa: E402
from baselines.shared.runner import run_batched  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def toy_data_dir(tmp_path: Path, n: int = 3) -> Path:
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    for i in range(1, n + 1):
        doc = {
            "question_ID": f"q{i}",
            "question": f"question {i}",
            "ground_truth": f"answer {i}",
            "history": [
                {"role": "Orchestrator", "content": f"plan {i}"},
                {"role": "Worker", "content": f"do {i}"},
            ],
            "mistake_agent": "Worker",
            "mistake_step": 1,
            "mistake_reason": f"reason {i}",
        }
        (d / f"{i}.json").write_text(json.dumps(doc))
    return d


def run_module(module: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", module, *args]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=check)


def _predict(data_dir: Path, out_dir: Path, *extra: str,
             check: bool = True) -> subprocess.CompletedProcess:
    return run_module(
        "baselines.errorprobe.predict",
        "--backend", "dummy", "--model", "dummy-model", "--model-name", "dummy",
        "--input", str(data_dir), "--output", str(out_dir), "--mode", "paper", *extra,
        check=check,
    )


def _run_program(record, backend, **kwargs):
    out = {}
    run_batched([(record, errorprobe_paper_program(record, **kwargs))], backend,
                lambda key, pred: out.update(pred))
    return out


TAGS_JSON = json.dumps({"tags": [
    {"step": 5, "mode": "2.5", "evidence": "doubles 8336000", "severity": 0.8},
    {"step": 7, "mode": "No or Incorrect Verification", "evidence": "no check", "severity": 0.5},
    {"step": 99, "mode": "ignored_input", "evidence": "out of range", "severity": 0.9},
    {"step": 5, "mode": "made_up_mode", "evidence": "unknown", "severity": 0.9},
]})
EDGES_JSON = json.dumps({"edges": [
    {"from": 3, "to": 5, "kind": "cites", "why": "uses the figure"},
    {"from": 6, "to": 7, "kind": "bogus", "why": "answer repeats output"},
    {"from": 7, "to": 3, "kind": "cites", "why": "backwards"},
]})
HYPOTHESES_JSON = json.dumps({"hypotheses": [
    {"step": 5, "agent": "orchestrator", "mode": "ignored_input", "rationale": "wrong operand",
     "tool": "code_exec", "check": "compare literals", "related_steps": [3, 6]},
    {"step": 3, "agent": "Nobody", "mode": "fail_task_spec", "rationale": "wrong city",
     "tool": "logic_probe", "check": "which Springfield", "related_steps": [0]},
    {"step": 5, "agent": "Orchestrator", "mode": "ignored_input", "rationale": "duplicate"},
]})
EVIDENCE_CODE_JSON = json.dumps({
    "tool": "code_exec", "execution_log": "print(8336000*2) -> 16672000",
    "expected_output": "228788", "observed_output": "16672000",
    "discrepancy": "-114394\n+8336000", "supports_hypothesis": True, "conclusive": True,
    "confidence": 0.9, "notes": ""})
EVIDENCE_LOGIC_JSON = json.dumps({
    "tool": "logic_probe",
    "preconditions": [{"condition": "task names a city", "source_steps": [0], "holds": False,
                       "evidence": "only 'Springfield'"}],
    "postconditions": [], "discrepancy": "", "supports_hypothesis": False, "conclusive": False,
    "confidence": 0.2})
VERDICT_JSON = json.dumps({
    "agent": "Orchestrator", "step": 5, "mode": "ignored_input", "confidence": 0.85,
    "reason": "The code doubled a literal WebSurfer never reported.", "chosen_hypothesis": 0,
    "novel_and_robust": True,
    "signature": {"mast_mode": "ignored_input", "tool": "python", "args": ["int"],
                  "context": ["orchestrator", "arithmetic"]},
    "guard": "Check that every operand in generated code appears in a specialist's report."})


def _responder(overrides: dict | None = None):
    """Responses keyed on each prompt's opening line; ``overrides`` swaps stages."""
    table = {
        "tagger": TAGS_JSON, "dependency": EDGES_JSON, "strategist": HYPOTHESES_JSON,
        "code_exec": EVIDENCE_CODE_JSON, "logic_probe": EVIDENCE_LOGIC_JSON,
        "arbiter": VERDICT_JSON,
    }
    table.update(overrides or {})

    def respond(messages):
        text = messages[-1]["content"]
        first = text.splitlines()[0]
        if first.startswith("You annotate"):
            return table["tagger"]
        if first.startswith("You map information flow"):
            return table["dependency"]
        if first.startswith("You lead the diagnosis"):
            return table["strategist"]
        if first.startswith("You are the investigator"):
            return table["code_exec"] if '"code_exec" probe' in first else table["logic_probe"]
        if first.startswith("You are the arbiter"):
            return table["arbiter"]
        raise AssertionError(f"unexpected prompt: {first[:60]}")
    return respond


# ─────────────────────────────────────────────────────────────────────────────
# golden prompts
# ─────────────────────────────────────────────────────────────────────────────

def test_prompts_match_goldens():
    expected = json.loads((FIXTURES / "errorprobe_paper_golden_prompts.json").read_text())
    actual = golden.build_all()
    assert set(actual) == set(expected)
    for name in expected:
        assert actual[name] == expected[name], f"golden drift in {name}"


def test_prompts_carry_their_contracts():
    g = golden.build_all()
    assert "Only mark a failure mode at a step if you can quote that step" in g["tagger"]
    assert "The trace has 9 steps, numbered from 0" in g["tagger"]
    assert "Never list a step merely because it comes before" in g["dependency"]
    assert team.DECISIVE_STEP in g["strategist"] and team.DECISIVE_STEP in g["arbiter"]
    assert team.NO_BARE_ASSERTION in g["investigator_code_exec"]
    assert team.NO_BARE_ASSERTION in g["investigator_logic_probe"]
    assert "PROBE: code_exec" in g["investigator_code_exec"]
    assert "PROBE: logic_probe" in g["investigator_logic_probe"]
    assert "NO HYPOTHESIS HAS CONCLUSIVE SUPPORTING EVIDENCE" in g["arbiter_all_inconclusive"]
    assert "FILTERED (evidence empty or inconclusive" in g["arbiter"]
    assert "novel_and_robust" in g["arbiter"] and '"signature"' in g["arbiter"]
    # The memory section exists only when patterns are retrieved.
    assert team.MEMORY_HEADER not in g["strategist"]
    assert team.MEMORY_HEADER in g["strategist_with_patterns"]
    assert "Before computing, verify the operand" in g["strategist_with_patterns"]
    # Every MAST mode id appears in the tagger's taxonomy and the Strategist's list.
    for mode in MAST_FAMILIES:
        assert f"- {mode} (MAST" in g["tagger"]
        assert f"- {mode}:" in g["strategist"]


def test_gt_line_is_the_only_with_gt_difference():
    g = golden.build_all()
    diff = g["tagger_with_gt"].replace("\nThe Answer for the problem is: 228,788", "")
    assert diff == g["tagger"]


# ─────────────────────────────────────────────────────────────────────────────
# structure: the deterministic parse
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("role,content,expected", [
    ("Orchestrator (termination condition)", "The task is complete.", "termination"),
    ("Assistant", "TERMINATE", "termination"),
    ("WebSurfer", '[WebSurfer] tool_call web_search(\n{\n  "query": "x"\n}\n)', "tool_call"),
    ("Computer_terminal", "exitcode: 0 (execution succeeded)\nCode output: 42", "execution_output"),
    ("Computer_terminal", "exitcode: 1 (execution failed)\nTraceback (most recent call last):\n  x", "execution_output"),
    ("WebSurfer", "I typed 'x' into '0 characters'.\n\nHere is a screenshot of [x](http://x).", "execution_output"),
    ("Coder", "Let's compute it:\n```python\nprint(1)\n```", "code"),
    ("Orchestrator (thought)", 'Updated Ledger:\n{\n  "is_request_satisfied": {"answer": false}\n}', "ledger"),
    ("Orchestrator", '{"instruction_or_question":{"answer":"Search","reason":"x"},"is_in_loop":{}}', "instruction"),
    ("Orchestrator (-> WebSurfer)", "Please look up the boundaries.", "instruction"),
    ("Planner", "Next Agent: WebSurfer\nPlease search.", "instruction"),
    ("Assistant", "## ANSWER\n42", "final_answer"),
    ("Assistant", "FINAL ANSWER: 42", "final_answer"),
    ("Movie_Expert", "Here is a list of movies under 150 minutes.", "message"),
])
def test_classify_action(role, content, expected):
    assert structure.classify_action(role, content) == expected


def test_speaker_selection_needs_a_known_agent():
    record = {"history": [
        {"role": "Orchestrator", "content": "Plan."},
        {"role": "Orchestrator", "content": "Next speaker WebSurfer"},
        {"role": "WebSurfer", "content": "Searching."},
        {"role": "Orchestrator", "content": "Nobody"},
    ]}
    steps = structure.build_steps(record)
    assert steps[1].action == "speaker_selection" and steps[1].addressee == "WebSurfer"
    assert steps[3].action == "message" and steps[3].addressee is None


def test_agent_identity_and_list():
    assert structure.agent_of("Orchestrator (-> WebSurfer)") == "Orchestrator"
    assert structure.agent_of("human") == "Unknown"
    assert structure.agent_of("user") == "Unknown"
    assert structure.agent_of("Assistant") == "Assistant"   # a real Magentic-One agent
    assert structure.agent_of("") == "Unknown"
    steps = structure.build_steps(golden.RECORD)
    assert structure.agent_list(steps) == ["Orchestrator", "WebSurfer", "ComputerTerminal", "Assistant"]
    # The alg-gen ``name: assistant`` key is dropped, so the role names the agent.
    rec = {"history": [{"role": "Movie_Expert", "name": "assistant", "content": "x"}]}
    assert structure.build_steps(rec)[0].agent == "Movie_Expert"


def test_agent_roles_from_both_shapes():
    steps = structure.build_steps(golden.RECORD)
    roles = structure.agent_roles(golden.RECORD, steps)
    assert set(roles) == {"Assistant", "WebSurfer", "ComputerTerminal"}
    assert roles["WebSurfer"].startswith("A helpful assistant with access")
    alg = {"system_prompt": {"Movie_Expert": "x" * 500}, "history": []}
    assert structure.agent_roles(alg, [])["Movie_Expert"] == "x" * 300
    assert structure.agent_roles({"history": []}, []) == {}
    assert structure.system_description(golden.RECORD).startswith("A Magentic-One")
    assert structure.system_description({}) is None


def test_clip_keeps_head_and_tail():
    text = "HEAD" + "x" * 1000 + "TAIL"
    out = structure.clip(text, 100)
    assert out.startswith("HEAD") and out.endswith("TAIL") and "chars omitted" in out
    assert structure.clip("short", 100) == "short"


def test_task_step_index():
    steps = structure.build_steps(golden.RECORD)
    assert structure.task_step_index(steps, golden.QUESTION) == 0
    rec = {"history": [{"role": "Planner", "content": "Plan"},
                       {"role": "Planner", "content": f"The task: {golden.QUESTION}"}]}
    assert structure.task_step_index(structure.build_steps(rec), golden.QUESTION) == 1


@pytest.mark.parametrize("raw,expected", [
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('prose then {"a": {"b": 2}} more prose', {"a": {"b": 2}}),
    ("<think>hmm</think>\n{\"a\": 3}", {"a": 3}),
    # LaTeX and literal newlines inside strings: invalid JSON, read leniently.
    (r'{"a": "$\sqrt{x}$ and \omega", "b": "one' + "\n" + 'two", "c": "tab\\t \\"q\\""}',
     {"a": "$\\sqrt{x}$ and \\omega", "b": "one\ntwo", "c": "tab\t \"q\""}),
    ('{"a": [1, 2,], "b": {"c": 1,},}', {"a": [1, 2], "b": {"c": 1}}),   # trailing commas
    ("no json here", None),
    ("[1, 2]", None),
    ("", None),
    (None, None),
])
def test_extract_json_object_never_raises(raw, expected):
    assert structure.extract_json_object(raw) == expected


# ─────────────────────────────────────────────────────────────────────────────
# tagger
# ─────────────────────────────────────────────────────────────────────────────

def test_definitions_cover_the_vendored_taxonomy():
    defs = tagger.load_definitions()
    assert set(defs) == set(MAST_FAMILIES)
    assert [m for m, _, _ in tagger.MAST_MODE_IDS] == list(MAST_FAMILIES)
    assert all(defs[m] for m in defs)
    # The 3.2/3.3 naming is resolved by name: "Weak" is the incomplete mode.
    assert defs["incomplete_verification"].startswith("Weak verification refers")
    assert defs["incorrect_verification"].startswith("Omission of proper checking")


@pytest.mark.parametrize("value,expected", [
    ("ignored_input", "ignored_input"), ("2.5", "ignored_input"), ("FM-2.6", "reasoning_action_mismatch"),
    ("Ignored Other Agent's Input", "ignored_input"), ("weak verification", "incomplete_verification"),
    ("No or Incorrect Verification", "incorrect_verification"), ("bogus", None), (None, None), ("", None),
])
def test_normalize_mode(value, expected):
    assert tagger.normalize_mode(value) == expected


def test_parse_tags_filters_and_dedupes():
    tags = tagger.parse_tags(TAGS_JSON, 0, 8)
    assert tags == [
        {"step": 5, "mode": "ignored_input", "evidence": "doubles 8336000", "severity": 0.8},
        {"step": 7, "mode": "incorrect_verification", "evidence": "no check", "severity": 0.5},
    ]
    assert tagger.parse_tags(TAGS_JSON, 6, 8) == [
        {"step": 7, "mode": "incorrect_verification", "evidence": "no check", "severity": 0.5}]
    assert tagger.parse_tags("garbage", 0, 8) is None
    assert tagger.parse_tags('{"tags": "nope"}', 0, 8) == []


def test_chunking_is_contiguous_and_complete():
    steps = [structure.Step(i, "A", "A", "message", None, "x" * 1000) for i in range(60)]
    chunks = tagger.chunk_steps(steps, 12000, 1200)
    assert len(chunks) > 1
    assert chunks[0][0] == 0 and chunks[-1][1] == 59
    for (a, b), (c, d) in zip(chunks, chunks[1:]):
        assert a <= b and c == b + 1
    assert tagger.chunk_steps(steps[:1], 10, 1200) == [(0, 0)]   # never empty
    assert tagger.chunk_steps([], 12000, 1200) == []


def test_tagger_examples_are_optional_and_large():
    steps = structure.build_steps(golden.RECORD)
    chunk = tagger.chunk_steps(steps, 12000, 1200)[0]
    without = tagger.build_tagger_prompt(golden.RECORD, steps, chunk)
    with_ex = tagger.build_tagger_prompt(golden.RECORD, steps, chunk, include_examples=True)
    assert "WORKED EXAMPLES FROM THE MAST AUTHORS" not in without
    assert "WORKED EXAMPLES FROM THE MAST AUTHORS" in with_ex
    assert len(with_ex) - len(without) > 60000


# ─────────────────────────────────────────────────────────────────────────────
# graph: edges, search, masking
# ─────────────────────────────────────────────────────────────────────────────

def test_structural_edges_on_the_fixture():
    steps = structure.build_steps(golden.RECORD)
    edges = graph.structural_edges(steps)
    pairs = {(e["from"], e["to"], e["kind"]) for e in edges}
    assert (5, 6, "executes") in pairs           # code -> its exit code
    assert (2, 3, "instructs") in pairs          # (-> WebSurfer) -> WebSurfer's turn
    assert (5, 6, "instructs") not in pairs      # code beats instruction in the classifier
    assert (7, 8, "responds_to") in pairs        # the symptom is never isolated
    assert all(e["source"] == "structural" for e in edges)


def test_parse_edges_drops_malformed_pairs():
    edges = graph.parse_edges(EDGES_JSON, 0, 8, 9)
    assert [(e["from"], e["to"], e["kind"]) for e in edges] == [(3, 5, "cites"), (6, 7, "cites")]
    assert graph.parse_edges(EDGES_JSON, 6, 8, 9) == [
        {"from": 6, "to": 7, "kind": "cites", "source": "llm", "why": "answer repeats output"}]
    assert graph.parse_edges("garbage", 0, 8, 9) is None
    assert graph.parse_edges('{"edges": {}}', 0, 8, 9) == []


def test_bfs_masks_a_succeeded_side_branch():
    # 0 task, 1..3 a side branch, 4 feeds 6, 5 feeds 6, 6 symptom.
    edges = [{"from": 4, "to": 6, "kind": "cites"}, {"from": 5, "to": 6, "kind": "cites"},
             {"from": 0, "to": 4, "kind": "cites"}, {"from": 1, "to": 2, "kind": "cites"},
             {"from": 2, "to": 3, "kind": "cites"}]
    erf, depth, floor = graph.effective_receptive_field(edges, 7, task_step=0, min_erf_steps=3)
    assert erf == [0, 4, 5, 6] and floor is False
    assert depth[6] == 0 and depth[4] == 1 and depth[0] == 2


def test_bfs_floor_on_an_edgeless_graph():
    erf, depth, floor = graph.effective_receptive_field([], 20, task_step=0, min_erf_steps=6)
    assert floor is True
    assert erf == [0, 14, 15, 16, 17, 18, 19]     # six from the tail plus the task step
    erf, _, floor = graph.effective_receptive_field([], 3, task_step=0, min_erf_steps=6)
    assert erf == [0, 1, 2]                        # floor capped at the trace length


def test_condense_masks_gaps_and_trims_by_priority():
    steps = [structure.Step(i, f"A{i % 2}", f"A{i % 2}", "message", None, f"turn {i} " + "y" * 900)
             for i in range(12)]
    erf = [0, 3, 4, 8, 11]
    depth = {11: 0, 8: 1, 4: 2, 3: 3, 0: 4}
    tags = [{"step": 3, "mode": "ignored_input", "evidence": "e", "severity": 0.7}]
    full = graph.condense(steps, erf, depth, tags, task_step=0, condensed_chars=100000)
    assert full["kept"] == erf
    assert full["masked"] == [[1, 2, "erf"], [5, 7, "erf"], [9, 10, "erf"]]
    assert "[steps 5-7 masked: 3 turns (A1 x2, A0 x1), no dependency path to the failure]" in full["text"]
    # Under budget pressure: symptom, task step and the tagged step 3 survive; 4 and 8 go first.
    tight = graph.condense(steps, erf, depth, tags, task_step=0, condensed_chars=2600)
    assert {0, 3, 11} <= set(tight["kept"])
    assert tight["step_chars"] == 600
    assert any(m[2] == "budget" for m in tight["masked"])
    dropped = {i for m in tight["masked"] if m[2] == "budget" for i in range(m[0], m[1] + 1)}
    assert dropped and dropped <= {4, 8}


def test_neighbour_helpers():
    edges = [{"from": 1, "to": 5, "kind": "cites"}, {"from": 3, "to": 5, "kind": "cites"},
             {"from": 5, "to": 6, "kind": "executes"}, {"from": 5, "to": 7, "kind": "cites"}]
    assert graph.incoming_neighbours(edges, 5) == [3, 1]
    assert graph.outgoing_outputs(edges, 5) == [6]


# ─────────────────────────────────────────────────────────────────────────────
# team parsers
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_hypotheses_fixes_and_caps():
    steps = structure.build_steps(golden.RECORD)
    agents = structure.agent_list(steps)
    hyps = team.parse_hypotheses(HYPOTHESES_JSON, steps, agents, 3)
    assert len(hyps) == 2                                    # the duplicate (5, ignored_input) is dropped
    assert hyps[0]["agent"] == "Orchestrator" and hyps[0]["agent_fixed"] is False
    assert hyps[1]["agent"] == "WebSurfer" and hyps[1]["agent_fixed"] is True   # "Nobody" -> speaker
    assert hyps[1]["tool"] == "logic_probe"                  # the Strategist's choice stands
    assert hyps[0]["related_steps"] == [3, 6]
    hammer = json.dumps({"hypotheses": [{"step": 3, "agent": "WebSurfer", "mode": "2.3", "tool": "hammer"}]})
    assert team.parse_hypotheses(hammer, steps, agents, 3)[0]["tool"] == "code_exec"   # default for an output
    assert team.parse_hypotheses(HYPOTHESES_JSON, steps, agents, 1) == hyps[:1]
    assert team.parse_hypotheses("garbage", steps, agents, 3) is None
    assert team.parse_hypotheses('{"hypotheses": []}', steps, agents, 3) == []
    out_of_range = json.dumps({"hypotheses": [{"step": 40, "agent": "WebSurfer", "mode": "x"}]})
    assert team.parse_hypotheses(out_of_range, steps, agents, 3) == []


def test_default_tool_follows_the_action_class():
    steps = structure.build_steps(golden.RECORD)
    assert team.default_tool(steps[3]) == "code_exec"        # a WebSurfer observation is an output
    assert team.default_tool(steps[5]) == "code_exec"
    assert team.default_tool(steps[2]) == "logic_probe"


def test_parse_evidence_failure_paths():
    bad = team.parse_evidence("garbage", "code_exec")
    assert bad["parse_error"] and not bad["conclusive"] and bad["confidence"] == 0.0
    bare = team.parse_evidence(json.dumps({"tool": "logic_probe", "conclusive": True,
                                           "supports_hypothesis": True, "confidence": 0.9}),
                               "logic_probe")
    assert bare["bare_assertion"] and bare["conclusive"] is False   # asserted, never shown
    good = team.parse_evidence(EVIDENCE_CODE_JSON, "code_exec")
    assert good["conclusive"] and good["supports_hypothesis"] and good["confidence"] == 0.9
    assert team.parse_evidence('{"confidence": 7, "discrepancy": "x"}', "code_exec")["confidence"] == 1.0   # clamped


def test_parse_verdict_and_signature_shape():
    steps = structure.build_steps(golden.RECORD)
    agents = structure.agent_list(steps)
    v = team.parse_verdict(VERDICT_JSON, steps, agents)
    assert (v["agent"], v["step"], v["mode"], v["confidence"]) == ("Orchestrator", 5, "ignored_input", 0.85)
    assert v["signature"] == {"tool": "python", "api": "default", "arg_schema": ["int"],
                              "context_slots": ["orchestrator", "arithmetic"],
                              "err_family": "ignored_input"}
    assert v["novel_and_robust"] is True
    fixed = team.parse_verdict(json.dumps({"agent": "Ghost", "step": 3, "mode": "2.2"}), steps, agents)
    assert fixed["agent"] == "WebSurfer" and fixed["agent_fixed"] and fixed["mode"] == "fail_clarification"
    assert fixed["signature"]["err_family"] == "fail_clarification"
    assert team.parse_verdict("garbage", steps, agents) is None
    assert team.parse_verdict('{"step": "nine"}', steps, agents) is None
    assert team.parse_verdict('{"step": 99}', steps, agents) is None


def test_select_fallback_and_memory_candidate():
    hyps = golden.HYPOTHESES
    assert team.select_fallback(hyps, golden.EVIDENCE_CONCLUSIVE) == 0
    assert team.select_fallback(hyps, golden.EVIDENCE_INCONCLUSIVE) == 0
    two_good = [dict(golden.EVIDENCE_CONCLUSIVE[0], confidence=0.4),
                dict(golden.EVIDENCE_CONCLUSIVE[0], confidence=0.6)]
    assert team.select_fallback(hyps, two_good) == 1
    steps = structure.build_steps(golden.RECORD)
    v = team.parse_verdict(VERDICT_JSON, steps, structure.agent_list(steps))
    mc = team.memory_candidate(v, golden.EVIDENCE_CONCLUSIVE[0])
    assert mc["eligible"] is True and mc["tau"] == 0.7
    assert team.memory_candidate(dict(v, confidence=0.5), golden.EVIDENCE_CONCLUSIVE[0])["eligible"] is False
    assert team.memory_candidate(v, golden.EVIDENCE_INCONCLUSIVE[0])["eligible"] is False


# ─────────────────────────────────────────────────────────────────────────────
# the program
# ─────────────────────────────────────────────────────────────────────────────

def test_program_four_rounds_and_output_doc():
    backend = DummyBackend(responder=_responder())
    out = _run_program(golden.RECORD, backend)
    roles = [c["role"] for c in out["calls"]]
    assert roles == ["tagger", "dependency", "strategist", "investigator", "investigator", "arbiter"]
    assert out["calls"][0]["steps"] == [0, 8] and out["calls"][0]["parsed"] is True
    assert [c["tool"] for c in out["calls"] if c["role"] == "investigator"] == ["code_exec", "logic_probe"]
    assert all("prompt" not in c for c in out["calls"])
    assert (out["predicted_agent"], out["predicted_step"]) == ("Orchestrator", 5)
    assert out["raw"] == VERDICT_JSON and out["failure_stage"] is None
    assert out["error_family"] == "ignored_input" and out["confidence"] == 0.85
    assert [t["mode"] for t in out["tags"]] == ["ignored_input", "incorrect_verification"]
    assert {e["source"] for e in out["graph"]["edges"]} == {"llm", "structural"}
    assert out["graph"]["n_steps"] == 9 and 5 in out["erf"]["steps"] and 8 in out["erf"]["steps"]
    assert out["arbiter_saw_only_inconclusive"] is False
    assert out["memory_candidate"]["eligible"] is True
    assert out["paper_params"] == PAPER_DEFAULTS
    assert len(out["structure"]) == 9 and out["structure"][5]["action"] == "code"
    # Round structure as the backend saw it: 2 + 1 + 2 + 1 calls.
    assert len(backend.calls) == 6
    json.dumps(out)


def test_program_no_hypotheses_stops_after_the_strategist():
    backend = DummyBackend(responder=_responder({"strategist": '{"hypotheses": []}'}))
    out = _run_program(golden.RECORD, backend)
    assert [c["role"] for c in out["calls"]] == ["tagger", "dependency", "strategist"]
    assert out["predicted_step"] is None and out["predicted_agent"] is None
    assert out["failure_stage"] == "strategist" and out["raw"] == '{"hypotheses": []}'
    assert out["hypotheses"] == [] and out["verdict"] is None
    assert "erf" in out and "tags" in out


def test_program_all_inconclusive_still_reaches_the_arbiter():
    backend = DummyBackend(responder=_responder({"code_exec": "garbage", "logic_probe": EVIDENCE_LOGIC_JSON}))
    out = _run_program(golden.RECORD, backend)
    assert out["arbiter_saw_only_inconclusive"] is True
    assert out["evidence"][0]["parse_error"] is True
    assert out["predicted_step"] == 5 and out["failure_stage"] is None


def test_program_arbiter_garbage_falls_back_to_best_evidence():
    backend = DummyBackend(responder=_responder({"arbiter": "not json"}))
    out = _run_program(golden.RECORD, backend)
    assert out["failure_stage"] == "arbiter" and out["raw"] == "not json"
    assert out["verdict"]["fallback"] is True
    assert (out["predicted_agent"], out["predicted_step"]) == ("Orchestrator", 5)
    assert out["confidence"] == 0.9                       # the code_exec evidence confidence
    assert out["memory_candidate"]["eligible"] is False


def test_program_default_dummy_never_crashes():
    out = _run_program(golden.RECORD, DummyBackend())
    assert out["predicted_step"] is None and out["failure_stage"] == "strategist"
    assert out["raw"] is not None


@pytest.mark.parametrize("n_turns", [1, 2])
def test_program_tiny_traces(n_turns):
    record = {"id": "1", "question": "q", "ground_truth": "a",
              "history": [{"role": "Orchestrator", "content": "plan"},
                          {"role": "Worker", "content": "do"}][:n_turns]}
    out = _run_program(record, DummyBackend(responder=_responder(
        {"strategist": json.dumps({"hypotheses": [{"step": 0, "agent": "Orchestrator", "mode": "2.3"}]})})))
    # The scripted verdict names step 5, which does not exist here, so the Arbiter
    # parse fails and the lone hypothesis (step 0) stands.
    assert out["predicted_step"] == 0 and out["failure_stage"] == "arbiter"
    assert out["graph"]["n_steps"] == n_turns and out["erf"]["steps"] == list(range(n_turns))


def test_program_long_trace_is_chunked_with_absolute_indices():
    record = {"id": "1", "question": "q", "ground_truth": "a",
              "history": [{"role": f"Agent{i % 3}", "content": f"turn {i} " + "z" * 1000}
                          for i in range(60)]}
    backend = DummyBackend(responder=_responder())
    out = _run_program(record, backend)
    tagger_calls = [c for c in out["calls"] if c["role"] == "tagger"]
    assert len(tagger_calls) > 1
    assert tagger_calls[0]["steps"][0] == 0 and tagger_calls[-1]["steps"][1] == 59
    second = [m for m in backend.calls if m[0]["content"].startswith("You annotate")][1][0]["content"]
    lo = tagger_calls[1]["steps"][0]
    assert f"You are tagging steps {lo} to" in second and f"Step {lo} [Agent{lo % 3}" in second


def test_gt_flag_reaches_every_task_prompt():
    record = dict(golden.RECORD, ground_truth="SECRET-ANSWER")
    for include_gt in (False, True):
        backend = DummyBackend(responder=_responder())
        _run_program(record, backend, include_gt=include_gt)
        for messages in backend.calls:
            assert ("SECRET-ANSWER" in messages[0]["content"]) is include_gt


def test_retrieved_patterns_hook_is_off_by_default():
    backend = DummyBackend(responder=_responder())
    _run_program(golden.RECORD, backend)
    strategist = [m for m in backend.calls if m[0]["content"].startswith("You lead")][0][0]["content"]
    assert team.MEMORY_HEADER not in strategist
    backend = DummyBackend(responder=_responder())
    _run_program(golden.RECORD, backend, retrieved_patterns=golden.PATTERNS)
    strategist = [m for m in backend.calls if m[0]["content"].startswith("You lead")][0][0]["content"]
    assert team.MEMORY_HEADER in strategist


# ─────────────────────────────────────────────────────────────────────────────
# predict CLI
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"id", "filename", "question_id", "method", "model", "backend",
                 "predicted_agent", "predicted_step", "gold_agent", "gold_step", "raw", "calls"}


def test_predict_e2e_and_resume(tmp_path):
    data = toy_data_dir(tmp_path)
    out_dir = tmp_path / "out"
    res = _predict(data, out_dir)
    assert "3 to run" in res.stdout
    mdir = out_dir / METHOD_PAPER
    assert {p.name for p in mdir.glob("[0-9]*.json")} == {"1.json", "2.json", "3.json"}
    doc = json.loads((mdir / "1.json").read_text())
    assert REQUIRED_KEYS <= set(doc)
    assert doc["method"] == METHOD_PAPER and doc["mode"] == "paper" and doc["model"] == "dummy"
    assert doc["gt_in_prompt"] is False and doc["paper_params"] == PAPER_DEFAULTS
    assert doc["predicted_step"] is None and doc["raw"] is not None   # dummy answers are not JSON
    run_cfg = json.loads((mdir / "_run.json").read_text())
    assert run_cfg["method"] == METHOD_PAPER and run_cfg["n_total"] == 3
    assert run_cfg["paper_params"] == PAPER_DEFAULTS and "request_params" in run_cfg

    (mdir / "3.json").unlink()
    before = (mdir / "1.json").stat().st_mtime_ns
    res = _predict(data, out_dir)
    assert "2 done, 1 to run" in res.stdout
    assert (mdir / "1.json").stat().st_mtime_ns == before
    assert "skip (complete)" in _predict(data, out_dir).stdout
    assert "3 to run" in _predict(data, out_dir, "--overwrite").stdout


def test_predict_with_gt_and_knobs(tmp_path):
    data = toy_data_dir(tmp_path)
    out_dir = tmp_path / "out"
    _predict(data, out_dir, "--gt", "with", "--max-hypotheses", "1", "--chunk-chars", "5000",
             "--method-dir", "errorprobe_paper.h1")
    mdir = out_dir / "errorprobe_paper.h1"
    doc = json.loads((mdir / "1.json").read_text())
    assert doc["gt_in_prompt"] is True and doc["method"] == METHOD_PAPER
    assert doc["paper_params"]["max_hypotheses"] == 1 and doc["paper_params"]["chunk_chars"] == 5000
    assert not (out_dir / METHOD_PAPER).exists()


def test_other_modes_do_not_carry_paper_params(tmp_path):
    data = toy_data_dir(tmp_path)
    out_dir = tmp_path / "out"
    run_module("baselines.errorprobe.predict", "--backend", "dummy", "--model", "d",
               "--input", str(data), "--output", str(out_dir), "--mode", "truncated")
    doc = json.loads((out_dir / "errorprobe" / "1.json").read_text())
    run_cfg = json.loads((out_dir / "errorprobe" / "_run.json").read_text())
    assert "paper_params" not in doc and "paper_params" not in run_cfg


# ─────────────────────────────────────────────────────────────────────────────
# sweep and report
# ─────────────────────────────────────────────────────────────────────────────

def _commands(stdout: str) -> list[str]:
    flat = stdout.replace("\\\n    ", " ")
    return [line for line in flat.splitlines() if "baselines.errorprobe.predict" in line]


@pytest.mark.parametrize("name", ["ww", "ww-api", "traceelephant", "traceelephant-api",
                                  "tracertraj", "tracertraj-api",
                                  "correct-error", "correct-error-api"])
def test_sweep_dry_run_paper_mode(name):
    res = run_module("baselines.errorprobe.sweep", "--config",
                     f"baselines/errorprobe/configs/{name}.yaml", "--set", "modes=[paper]", "--dry-run")
    cmds = _commands(res.stdout)
    assert cmds and all("--mode paper" in c for c in cmds)
    assert all("--include-mast-examples False" in c and "--sequential-edges fallback" in c for c in cmds)
    assert "--mode truncated" not in res.stdout
    for c in cmds:
        if "--model-name qwen3.5-9b" in c and not name.endswith("-api"):
            # The local configs run qwen on a tight budget: the spec's decode
            # handicap plus a one-hypothesis paper block at the default chunk size.
            assert "--max-hypotheses 1" in c and "--chunk-chars 12000" in c
            assert "--condensed-chars 10000" in c
            assert "--gen_max_tokens 512" in c and "--temperature 1.0" in c and "--top_p 0.95" in c
            assert "--max_model_len 16384" in c and "--truncate_prompt_tokens 15872" in c
            for stale in ("--gen_max_tokens 1024", "--chunk-chars 24000", "--max_model_len 24576"):
                assert stale not in c
        elif name.endswith("-api"):
            # Both API specs carry the same paper block; vLLM knobs never reach them.
            assert "--max-hypotheses 1" in c and "--chunk-chars 12000" in c
            assert "--condensed-chars 10000" in c and "--gen_max_tokens" not in c
        else:
            assert "--max-hypotheses 3" in c and "--gen_max_tokens 512" not in c


def test_per_model_paper_block_applies_to_paper_mode_only():
    res = run_module("baselines.errorprobe.sweep", "--config", "baselines/errorprobe/configs/ww.yaml",
                     "--set", "modes=[truncated,paper]", "--set", "models=[qwen3.5-9b]",
                     "--set", "subsets=[hand-crafted]", "--dry-run")
    cmds = _commands(res.stdout)
    truncated = [c for c in cmds if "--mode truncated" in c]
    paper = [c for c in cmds if "--mode paper" in c]
    assert len(truncated) == 1 and len(paper) == 1
    # The spec's handicap reaches every mode; the paper block reaches only paper.
    for c in (truncated[0], paper[0]):
        assert "--gen_max_tokens 512" in c and "--max_model_len 16384" in c
    assert "--max-hypotheses" not in truncated[0] and "--condensed-chars" not in truncated[0]
    assert "--max-hypotheses 1" in paper[0] and "--condensed-chars 10000" in paper[0]


def test_qwen_spec_is_handicapped_in_every_mode():
    """The official qwen3.5-9b spec decodes on a tight budget in all three modes.

    The results under outputs*/…/qwen3.5-9b/errorprobe* were produced with these
    flags (their _run.json records them), so the config must keep emitting them.
    """
    for name in ["ww", "traceelephant", "tracertraj", "correct-error"]:
        default = run_module("baselines.errorprobe.sweep", "--config",
                             f"baselines/errorprobe/configs/{name}.yaml", "--dry-run").stdout
        assert "weak" not in default                    # the handicap is the model, not a variant
        res = run_module("baselines.errorprobe.sweep", "--config",
                         f"baselines/errorprobe/configs/{name}.yaml", "--dry-run",
                         "--set", "models=[qwen3.5-9b]", "--set", "modes=[truncated,backward,paper]")
        cmds = _commands(res.stdout)
        assert len(cmds) == 3 * len(load_cfg_subsets(name))
        for c in cmds:
            assert "--temperature 1.0" in c and "--top_p 0.95" in c
            assert "--max_model_len 16384" in c and "--truncate_prompt_tokens 15872" in c
            assert "--gen_max_tokens 512" in c and "--gpu_memory_utilization 0.6" in c
        paper = [c for c in cmds if "--mode paper" in c]
        assert paper and all("--max-hypotheses 1" in c and "--chunk-chars 12000" in c
                             and "--condensed-chars 10000" in c for c in paper)


def load_cfg_subsets(name: str) -> list[str]:
    import yaml
    return yaml.safe_load(open(f"baselines/errorprobe/configs/{name}.yaml"))["subsets"]


def test_paper_args_merge_and_validation():
    from baselines.errorprobe.sweep import paper_args
    cfg = {"paper": {"max_hypotheses": 3, "chunk_chars": 12000}}
    spec = {"backend": "vllm", "paper": {"max_hypotheses": 1, "gen_max_tokens": 512}}
    argv = paper_args("m", spec, cfg)
    assert argv == ["--max-hypotheses", "1", "--chunk-chars", "12000", "--gen_max_tokens", "512"]
    # vLLM knobs are dropped for API backends; unknown keys are rejected.
    assert paper_args("m", {"backend": "openai", "paper": {"gen_max_tokens": 512}}, {}) == []
    with pytest.raises(SystemExit):
        paper_args("m", {"paper": {"bogus": 1}}, {})


def test_shipped_configs_keep_paper_opt_in():
    for name in ["ww", "ww-api", "traceelephant", "traceelephant-api",
                 "tracertraj", "tracertraj-api", "correct-error", "correct-error-api"]:
        res = run_module("baselines.errorprobe.sweep", "--config",
                         f"baselines/errorprobe/configs/{name}.yaml", "--dry-run")
        assert "--mode paper" not in res.stdout.replace("\\\n    ", " ")


def test_sweep_e2e_dummy(tmp_path):
    toy_data_dir(tmp_path)
    outputs = tmp_path / "outputs"
    argv = ["--config", "baselines/errorprobe/configs/ww-api.yaml", "--gt", "with",
            "--set", f"data_dir={tmp_path}", "--set", "subsets=[data]",
            "--set", f"outputs_root={outputs}", "--set", "models=[dummy]",
            "--set", "modes=[paper]", "--set", "paper.max_hypotheses=2",
            "--set", "model_specs={dummy: {backend: dummy}}"]
    run_module("baselines.errorprobe.sweep", *argv)
    mdir = outputs / "data" / "dummy" / METHOD_PAPER
    assert len(list(mdir.glob("[0-9]*.json"))) == 3
    doc = json.loads((mdir / "1.json").read_text())
    assert doc["mode"] == "paper" and doc["paper_params"]["max_hypotheses"] == 2
    assert "skip (complete)" in run_module("baselines.errorprobe.sweep", *argv).stdout


def test_report_scores_all_three_modes(tmp_path):
    data = toy_data_dir(tmp_path)
    subset_data = tmp_path / "corpus" / "sub"
    subset_data.mkdir(parents=True)
    for f in data.glob("*.json"):
        (subset_data / f.name).write_text(f.read_text())
    pred_root = tmp_path / "pred"
    for mode in ("truncated", "backward", "paper"):
        run_module("baselines.errorprobe.predict", "--backend", "dummy", "--model", "d",
                   "--model-name", "dummy", "--input", str(subset_data),
                   "--output", str(pred_root / "sub" / "dummy"), "--mode", mode)
    report_cfg = tmp_path / "report.yaml"
    report_cfg.write_text(f"""
models:  [dummy]
subsets: [sub]
methods: [errorprobe, errorprobe_bt, errorprobe_paper]
gt: with
data_dir:  {tmp_path / "corpus"}
pred_root: {pred_root}
out_root:  {tmp_path / "reports"}
splits: {{train: 0.3, val: 0.2, test: 0.5}}
seeds: [1, 2]
gt_in_prompt: true
""")
    assert "DONE" in run_module("baselines.errorprobe.report", "--config", str(report_cfg),
                                "--check-only").stdout
    run_module("baselines.errorprobe.report", "--config", str(report_cfg))
    header = (tmp_path / "reports" / "dummy" / "sub" / "comparison_by_seed.tsv").read_text().splitlines()[0]
    assert "errorprobe_paper_step@1_test" in header and "errorprobe_bt_step@1_test" in header
