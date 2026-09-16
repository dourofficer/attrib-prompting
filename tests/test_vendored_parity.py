"""Execution-level parity between our method programs and the vendored baseline.

Loads ``vendored/Agents_Failure_Attribution/Automated_FA/Lib/local_model.py``
(with torch/transformers/tqdm stubbed if absent), monkeypatches its
``_run_local_generation`` to capture the exact chat messages, runs the vendored
analyze functions on a synthetic trajectory, and asserts our programs issue
**byte-identical** prompts and reach the same decisions.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

from baselines.prompting.methods import (
    AGENT_RE,
    STEP_RE,
    METHODS,
    build_all_at_once_prompt,
    build_binary_search_prompt,
    build_step_by_step_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_LIB = REPO_ROOT / "vendored/Agents_Failure_Attribution/Automated_FA/Lib/local_model.py"
GOLDENS = json.loads((REPO_ROOT / "tests/fixtures/golden_prompts.json").read_text())

HISTORY = [
    {"role": "Orchestrator", "content": "Plan: search the web, then compute."},
    {"role": "WebSurfer", "content": "I found the population is 8,336,000."},
    {"role": "Assistant", "content": "Final answer: 8,336,000 * 2 = 16,672,000."},
    {"role": "Verifier", "content": "Looks good to me."},
]
RECORD = {
    "id": "1",
    "history": HISTORY,
    "question": "What is twice the population of the city?",
    "ground_truth": "16,672,000",
}


def _stub(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ImportError:
        pass
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


@pytest.fixture(scope="module")
def vendored():
    _stub("torch")
    _stub("tqdm", tqdm=lambda it, **kw: it)
    _stub(
        "transformers",
        pipeline=None,
        AutoTokenizer=None,
        AutoModelForCausalLM=None,
        Pipeline=type("Pipeline", (), {}),
    )
    spec = importlib.util.spec_from_file_location("vendored_local_model", VENDORED_LIB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def data_dir(tmp_path):
    doc = {**RECORD, "history": HISTORY}
    (tmp_path / "1.json").write_text(json.dumps(doc), encoding="utf-8")
    return tmp_path


def _run_program(method: str, responses: list[str], step_mode: str = "early_stop"):
    """Drive one program with scripted responses; return (prompts_issued, pred)."""
    gen = METHODS[method](RECORD, step_mode=step_mode)
    script = list(responses)
    issued: list[list[dict]] = []
    try:
        prompts = next(gen)
        while True:
            issued.extend(prompts)
            outs = [script.pop(0) for _ in prompts]
            prompts = gen.send(outs)
    except StopIteration as si:
        return issued, si.value


def _capture_vendored(vendored, monkeypatch, responses: list[str]):
    captured: list[list[dict]] = []
    script = list(responses)

    def fake_generation(model_obj, messages, model_family="llama"):
        captured.append(messages)
        return script.pop(0)

    monkeypatch.setattr(vendored, "_run_local_generation", fake_generation)
    return captured


def test_all_at_once_prompts_identical(vendored, monkeypatch, data_dir):
    captured = _capture_vendored(vendored, monkeypatch, ["Agent Name: X\nStep Number: 0"])
    vendored.analyze_all_at_once_local(None, str(data_dir), is_handcrafted=True, model_family="qwen")

    ours, pred = _run_program("all_at_once", ["Agent Name: X\nStep Number: 0"])
    assert captured == ours
    assert pred["predicted_agent"] == "X" and pred["predicted_step"] == 0


def test_step_by_step_early_stop_identical(vendored, monkeypatch, data_dir):
    # "Yes" at step 2: the vendored loop stops there and never issues steps 3+.
    responses = ["1. No.", "1. No.", "1. Yes.\n2. Reason: wrong sum."]
    captured = _capture_vendored(vendored, monkeypatch, list(responses))
    vendored.analyze_step_by_step_local(None, str(data_dir), is_handcrafted=True, model_family="qwen")

    ours, pred = _run_program("step_by_step", list(responses), step_mode="early_stop")
    assert captured == ours  # same prompts, same count (early stop after step 2)
    assert len(ours) == 3
    assert pred["predicted_step"] == 2
    assert pred["predicted_agent"] == HISTORY[2]["role"]
    assert pred["raw"] == responses[-1]


def test_step_by_step_batch_matches_early_stop_prompts(vendored, monkeypatch, data_dir):
    # Batch mode issues every step's prompt; the issued early-stop prompts must
    # be a strict prefix, and the prediction identical, wherever the Yes lands.
    n = len(HISTORY)
    for yes_at in [*range(n), None]:
        responses = ["1. No."] * n
        if yes_at is not None:
            responses[yes_at] = "1. Yes."
        batch_prompts, batch_pred = _run_program("step_by_step", list(responses), "batch")
        k = (yes_at + 1) if yes_at is not None else n
        early_prompts, early_pred = _run_program("step_by_step", list(responses)[:k], "early_stop")
        assert early_prompts == batch_prompts[:len(early_prompts)]
        assert len(early_prompts) == k
        for key in ("predicted_agent", "predicted_step", "raw"):
            assert batch_pred[key] == early_pred[key], (yes_at, key)


def test_binary_search_identical(vendored, monkeypatch, data_dir):
    for responses, expected_step in [
        (["upper half", "lower half"], 1),
        (["lower half", "upper half"], 2),
        (["garbage response", "also garbage"], 0),  # ambiguous → upper half
        (["lower half", "lower half"], 3),
    ]:
        captured = _capture_vendored(vendored, monkeypatch, list(responses))
        reported: list[int] = []
        monkeypatch.setattr(
            vendored, "_report_binary_search_error_local",
            lambda chat_history, step, json_file, is_handcrafted: reported.append(step),
        )
        vendored.analyze_binary_search_local(None, str(data_dir), is_handcrafted=True, model_family="qwen")

        ours, pred = _run_program("binary_search", list(responses))
        assert captured == ours, responses
        assert reported == [expected_step], responses
        assert pred["predicted_step"] == expected_step
        assert pred["predicted_agent"] == HISTORY[expected_step]["role"]
        assert pred["raw"] == responses[-1]


def test_builders_match_goldens():
    assert build_all_at_once_prompt(HISTORY[:3], RECORD["question"], RECORD["ground_truth"]) \
        == GOLDENS["all_at_once"]
    assert build_step_by_step_prompt(
        RECORD["question"], RECORD["ground_truth"],
        "Step 0 - Orchestrator: Plan: search the web, then compute.\n", 0, "Orchestrator",
    ) == GOLDENS["step_by_step"]
    assert "{idx}" in GOLDENS["step_by_step"]  # the intentionally unformatted literal
    assert build_binary_search_prompt(
        RECORD["question"], RECORD["ground_truth"],
        "Orchestrator: Plan: search the web, then compute.\nWebSurfer: I found the population is 8,336,000.",
        range_description="from step 0 to step 1",
        upper_half_desc="from step 0 to step 0",
        lower_half_desc="from step 1 to step 1",
    ) == GOLDENS["binary_search"]


def test_parse_regexes_match_vendored_evaluate():
    # Our regexes are the vendored evaluate.py patterns plus optional
    # surrounding parentheses (a documented deviation: API models mimic the
    # prompt's literal "Agent Name: (Your prediction)" format). Pin both the
    # vendored originals and the exact tolerant form so any further drift
    # fails here.
    evaluate_src = (REPO_ROOT / "vendored/Agents_Failure_Attribution/Automated_FA/evaluate.py") \
        .read_text(encoding="utf-8")
    assert r"Agent Name:\s*([\w_]+)" in evaluate_src
    assert r"Step Number:\s*(\d+)" in evaluate_src
    assert AGENT_RE.pattern == r"Agent Name:\s*\(?\s*([\w_]+)\s*\)?"
    assert STEP_RE.pattern == r"Step Number:\s*\(?\s*(\d+)\s*\)?"
    # The wrappers are a strict superset: unparenthesized (vendored-shaped)
    # outputs parse identically; parenthesized ones now parse too.
    assert AGENT_RE.search("Agent Name: WebSurfer").group(1) == "WebSurfer"
    assert AGENT_RE.search("Agent Name: (WebSurfer)").group(1) == "WebSurfer"
    assert STEP_RE.search("Step Number: 4").group(1) == "4"
    assert STEP_RE.search("Step Number: (4)").group(1) == "4"


def test_parse_all_at_once_variants():
    from baselines.prompting.methods import parse_all_at_once

    plain = "Agent Name: WebSurfer\nStep Number: 4\nReason for Mistake: bad click."
    assert parse_all_at_once(plain) == ("WebSurfer", 4)
    bolded = "**Agent Name:** WebSurfer\n**Step Number:** 4"
    assert parse_all_at_once(bolded) == ("WebSurfer", 4)
    think = "<think>step 9? no, 4.</think>Agent Name: WebSurfer\nStep Number: 4"
    assert parse_all_at_once(think) == ("WebSurfer", 4)
    dangling = "...reasoning...</think>Agent Name: WebSurfer\nStep Number: 4"
    assert parse_all_at_once(dangling) == ("WebSurfer", 4)
    parenthesized = "Agent Name: (Planner)\n, Step Number: (9)\n, Reason: bad plan"
    assert parse_all_at_once(parenthesized) == ("Planner", 9)
    assert parse_all_at_once("no structured answer") == (None, None)


def test_parse_all_at_once_resolves_multi_word_agents_against_the_vocabulary():
    """tracertraj names its agents "Product Manager", which the vendored
    ``[\\w_]+`` capture cuts to "Product". With the trajectory's agent names the
    parser returns the agent as spelled in history; without them, or when the
    vendored capture already names a known agent, nothing changes."""
    from baselines.prompting.methods import parse_all_at_once

    agents = ["Team Leader", "Product Manager", "Architect", "Engineer", "Data Analyst"]
    for raw in ("Agent Name: Product Manager\nStep Number: 3",
                "Agent Name: Product Manager (step 3)\nStep Number: 3",
                "Agent Name: The Product Manager\nStep Number: 3",
                "**Agent Name:** Product Manager\n**Step Number:** 3",
                "Agent Name: (Product Manager)\n, Step Number: (3)",
                "Agent Name: product manager\nStep Number: 3"):
        assert parse_all_at_once(raw, agents) == ("Product Manager", 3), raw
    # Vendored behaviour without a vocabulary.
    assert parse_all_at_once("Agent Name: Product Manager\nStep Number: 3") == ("Product", 3)
    # A single-word capture that is a known agent stands; "Engineers" is a
    # different word, and a capture no agent matches keeps the vendored value.
    assert parse_all_at_once("Agent Name: Engineer\nStep Number: 3", agents) == ("Engineer", 3)
    assert parse_all_at_once("Agent Name: Engineers\nStep Number: 3", agents) == ("Engineers", 3)
    assert parse_all_at_once("Agent Name: Nobody here\nStep Number: 3", agents) == ("Nobody", 3)
    # The search stops at the next label or sentence, so an agent mentioned in
    # the reason text never becomes the prediction.
    one_line = "Agent Name: Team Leader, Step Number: 5, Reason: the Engineer was fine"
    assert parse_all_at_once(one_line, agents) == ("Team Leader", 5)
    no_agent = "Agent Name: No agent, Step Number: 5, Reason: the Engineer was fine"
    assert parse_all_at_once(no_agent, agents) == ("No", 5)
    sentence = "Agent Name: No agent made a mistake. The Engineer did fine\nStep Number: 5"
    assert parse_all_at_once(sentence, agents) == ("No", 5)
    # A capture the report would already score (gold-in-pred) is left alone.
    ww = ["Planner", "user", "WebSurfer"]
    assert parse_all_at_once("Agent Name: Planners (user)\nStep Number: 1", ww) == ("Planners", 1)
    # standardize_role collapses Orchestrator variants on both sides.
    hc = ["Orchestrator (thought)", "WebSurfer", "Orchestrator (termination condition)"]
    assert parse_all_at_once("Agent Name: Orchestrator (thought)\nStep Number: 2", hc) == ("Orchestrator", 2)
    # A longer known agent at the capture's own position wins.
    assert parse_all_at_once("Agent Name: Engineer Lead\nStep Number: 2",
                             ["Engineer", "Engineer Lead"]) == ("Engineer Lead", 2)
    # Hyphenated names are recovered the same way.
    assert parse_all_at_once("Agent Name: Blu-Ray_Expert\nStep Number: 2",
                             ["Blu-Ray_Expert", "Video_Expert"]) == ("Blu-Ray_Expert", 2)


def test_without_gt_removes_only_the_answer_line():
    # include_gt=False must equal the with-GT golden minus exactly the
    # "The Answer for the problem is: <gt>\n" line — nothing else changes.
    gt_line = f"The Answer for the problem is: {RECORD['ground_truth']}\n"

    with_gt = build_all_at_once_prompt(HISTORY[:3], RECORD["question"], RECORD["ground_truth"])
    without = build_all_at_once_prompt(HISTORY[:3], RECORD["question"], RECORD["ground_truth"],
                                       include_gt=False)
    assert with_gt == GOLDENS["all_at_once"]
    assert without == with_gt.replace(gt_line, "")
    assert "The Answer for the problem is:" not in without

    acc = "Step 0 - Orchestrator: Plan: search the web, then compute.\n"
    with_gt = build_step_by_step_prompt(RECORD["question"], RECORD["ground_truth"], acc, 0, "Orchestrator")
    without = build_step_by_step_prompt(RECORD["question"], RECORD["ground_truth"], acc, 0, "Orchestrator",
                                        include_gt=False)
    assert with_gt == GOLDENS["step_by_step"]
    assert without == with_gt.replace(gt_line, "")

    seg = "Orchestrator: Plan: search the web, then compute.\nWebSurfer: I found the population is 8,336,000."
    kw = dict(range_description="from step 0 to step 1", upper_half_desc="from step 0 to step 0",
              lower_half_desc="from step 1 to step 1")
    with_gt = build_binary_search_prompt(RECORD["question"], RECORD["ground_truth"], seg, **kw)
    without = build_binary_search_prompt(RECORD["question"], RECORD["ground_truth"], seg,
                                         include_gt=False, **kw)
    assert with_gt == GOLDENS["binary_search"]
    assert without == with_gt.replace(gt_line, "")
