"""RAFFLES pipeline tests (dummy backend, subprocess, keyless).

RAFFLES has no vendored code, and this repo deliberately runs simplified
prompts (see ``baselines/raffles/prompts.py``) rather than the paper's
Appendix F.3 text. The prompt bytes are still pinned two ways: golden fixtures
lock the current wording (``tests/fixtures/raffles_golden_prompts.json``), and
phrase assertions keep a fixture regeneration from silently changing the
prompts' core contract.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

GOLDENS = json.loads(
    (REPO_ROOT / "tests/fixtures/raffles_golden_prompts.json").read_text())

# One valid Judge response and confidence-scripted Evaluator responses. The toy
# records speak Orchestrator at step 0 and Worker at step 1, so this candidate
# passes the rule-based check (criterion 4 → 100).
JUDGE_JSON = json.dumps({
    "agent_name": "Worker", "step_number": 1,
    "mistake_reason": "r1", "first_mistake": "r2", "mistake_not_corrected": "r3",
})


def _eval_json(confidence: int, reason: str = "ok") -> str:
    return f'```json\n{{"reason": "{reason}", "confidence": {confidence}}}\n```'


def _is_judge(msgs) -> bool:
    """Only the Evaluator prompt carries an Error Step section to verify."""
    return "## Error Step ##" not in msgs[-1]["content"]


def _responder(eval_confidence: int):
    """Judge prompts carry no Error Step; everything else is an Evaluator."""
    def respond(msgs):
        if _is_judge(msgs):
            return JUDGE_JSON
        return _eval_json(eval_confidence)
    return respond


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


def toy_record(i: int = 1) -> dict:
    return {
        "id": str(i), "filename": f"{i}.json", "question_id": f"q{i}",
        "question": f"question {i}", "ground_truth": f"answer {i}",
        "history": [
            {"role": "Orchestrator", "content": f"plan {i}"},
            {"role": "Worker", "content": f"do {i}"},
        ],
        "gold_agent": "Worker", "gold_step": 1,
    }


def run_module(module: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", module, *args]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=check)


def _predict(data_dir: Path, out_dir: Path, *extra: str,
             check: bool = True) -> subprocess.CompletedProcess:
    return run_module(
        "baselines.raffles.predict",
        "--backend", "dummy", "--model", "dummy-model", "--model-name", "dummy",
        "--input", str(data_dir), "--output", str(out_dir), *extra, check=check,
    )


def _run_program(record, backend, **kwargs):
    from baselines.shared.runner import run_batched
    from baselines.raffles.methods import raffles_program
    out = {}
    run_batched([(record, raffles_program(record, **kwargs))], backend,
                lambda key, pred: out.update(pred))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# prompt transcription: golden bytes + paper phrases
# ─────────────────────────────────────────────────────────────────────────────

def test_prompts_match_goldens():
    """The builders reproduce the captured transcription byte-for-byte."""
    from tests.fixtures.capture_raffles_goldens import CANDIDATE, FEEDBACK, RECORD
    from baselines.raffles.prompts import build_evaluator_prompt, build_judge_prompt

    assert build_judge_prompt(RECORD) == GOLDENS["judge"]
    assert build_judge_prompt(RECORD, include_gt=True) == GOLDENS["judge_with_gt"]
    assert build_judge_prompt(RECORD, FEEDBACK) == GOLDENS["judge_with_feedback"]
    for p in (1, 2, 3):
        assert build_evaluator_prompt(p, RECORD, CANDIDATE) == GOLDENS[f"evaluator_{p}"]


def test_judge_prompt_carries_the_core_contract():
    """Key sentences, so regenerated goldens can't silently change the contract."""
    judge = GOLDENS["judge"]
    for phrase in (
        "analyzing a multi-agent conversation history",
        "**You must always output an agent name and a step number.**",
        "1. The agent made a mistake at that step.",
        "2. It is the first mistake step that relates to the final wrong outcome.",
        "3. The mistake was not corrected by or correctable by later agents.",
        "## Handling Ambiguity (Fallback Procedure)",
        "Please answer in the format:",
        "Remember, that your output should only be a json and nothing else.",
        '"agent_name"',
        '"step_number"',
        '"mistake_reason"',
        '"first_mistake"',
        '"mistake_not_corrected"',
    ):
        assert phrase in judge, phrase


def test_evaluator_prompts_carry_the_core_contract():
    for p, criterion in ((1, "correctly pointing out a faulty agent and step number"),
                         (2, "finding the first mistake in the pipeline"),
                         (3, "how this mistake was never corrected afterwards")):
        ev = GOLDENS[f"evaluator_{p}"]
        for phrase in (
            "You are a rigorous and meticulous logic verifier",
            f"Your task is **ONLY** to think whether the argument provided by "
            f"your partner for '{criterion}' is logical.",
            "## Task Log ##",
            "## Error Step ##",
            "## Your output format ##",
            "confidence score between 0 to 100",
        ):
            assert phrase in ev, (p, phrase)


def test_evaluator_sees_only_its_own_rationale():
    """Evaluator p's Claim carries r_j^p, not the other two rationales."""
    fields = {1: "mistake_reason", 2: "first_mistake", 3: "mistake_not_corrected"}
    for p, field in fields.items():
        claim = GOLDENS[f"evaluator_{p}"].split("## Error Step ##")[1]
        assert f'"{field}"' in claim
        for other in set(fields.values()) - {field}:
            assert f'"{other}"' not in claim


def test_gt_line_is_the_only_with_gt_difference():
    assert "The Answer for the problem is:" not in GOLDENS["judge"]
    with_gt, without = GOLDENS["judge_with_gt"], GOLDENS["judge"]
    assert "The Answer for the problem is: 16,672,000\n" in with_gt
    assert with_gt.replace("The Answer for the problem is: 16,672,000\n", "") == without


def test_parsers():
    from baselines.raffles.prompts import extract_json, normalize_step, parse_evaluator

    assert extract_json('noise ```json\n{"a": 1}\n``` more')["a"] == 1
    assert extract_json('bare {"a": 2} text')["a"] == 2
    with pytest.raises(ValueError):
        extract_json("no json here")

    assert parse_evaluator('{"reason": "r", "confidence": 90}') == ("r", 90)
    assert parse_evaluator('{"reason": "r", "confidence": "85%"}') == ("r", 85)
    assert parse_evaluator('{"reason": "r", "confidence": 150}') == ("r", 100)
    assert parse_evaluator('{"reason": "r", "confidence": "high"}') == ("r", 0)
    assert parse_evaluator("garbage") == ("[unparseable evaluator response]", 0)

    assert normalize_step("3.0") == 3 and normalize_step(None) is None


# ─────────────────────────────────────────────────────────────────────────────
# the Judge-Evaluator loop
# ─────────────────────────────────────────────────────────────────────────────

def test_high_confidence_terminates_after_one_iteration():
    from baselines.shared.backends import get_backend

    backend = get_backend("dummy", responder=_responder(95))
    out = _run_program(toy_record(), backend)

    assert out["predicted_agent"] == "Worker" and out["predicted_step"] == 1
    assert out["confidence"] == 95 * 3 + 100  # three evaluators + rule check
    assert out["n_iterations"] == 1
    assert [(c["iteration"], c["role"]) for c in out["calls"]] == \
        [(0, "judge"), (0, "evaluator"), (0, "evaluator"), (0, "evaluator")]
    assert out["raw"] == JUDGE_JSON


def test_low_confidence_runs_all_iterations():
    """K=2 → three Judge rounds when C never clears the threshold."""
    from baselines.shared.backends import get_backend

    backend = get_backend("dummy", responder=_responder(10))
    out = _run_program(toy_record(), backend)

    assert out["n_iterations"] == 3 and len(out["calls"]) == 12
    assert out["confidence"] == 10 * 3 + 100
    assert out["predicted_step"] == 1  # ties → latest, all iterations equal


def test_highest_confidence_candidate_wins():
    """The middle iteration scores best (≤ threshold) and its candidate is kept."""
    from baselines.shared.backends import get_backend

    judges = [json.dumps({"agent_name": "Orchestrator", "step_number": 0}),
              json.dumps({"agent_name": "Worker", "step_number": 1}),
              json.dumps({"agent_name": "Orchestrator", "step_number": 0})]
    confidences = iter([10, 10, 10, 80, 80, 80, 50, 50, 50])

    def respond(msgs):
        if _is_judge(msgs):
            return judges.pop(0)
        return _eval_json(next(confidences))

    backend = get_backend("dummy", responder=respond)
    out = _run_program(toy_record(), backend)

    totals = [it["total_confidence"] for it in out["iterations"]]
    assert totals == [130, 340, 250]
    assert out["confidence"] == 340
    assert out["predicted_agent"] == "Worker" and out["predicted_step"] == 1


def test_rule_check_zeroes_inconsistent_candidates():
    from baselines.raffles.prompts import rule_check

    history = [{"role": "Orchestrator", "content": "a"}, {"role": "Worker", "content": "b"}]
    assert rule_check(history, {"agent_name": "worker", "step_number": "1"})[1] == 100
    assert rule_check(history, {"agent_name": "Worker", "step_number": 5})[1] == 0
    assert rule_check(history, {"agent_name": "Orchestrator", "step_number": 1})[1] == 0
    assert rule_check(history, {"agent_name": "Worker", "step_number": "n/a"})[1] == 0


def test_unparseable_judge_skips_evaluators_and_recovers():
    """A garbage Judge round costs that iteration only; the next one succeeds."""
    from baselines.shared.backends import get_backend

    state = {"judge_calls": 0}

    def respond(msgs):
        if _is_judge(msgs):
            state["judge_calls"] += 1
            return "not json" if state["judge_calls"] == 1 else JUDGE_JSON
        return _eval_json(95)

    backend = get_backend("dummy", responder=respond)
    out = _run_program(toy_record(), backend)

    assert out["n_iterations"] == 2
    assert out["iterations"][0]["candidate"] is None
    assert out["predicted_agent"] == "Worker" and out["confidence"] == 385
    # Iteration 0 issued no Evaluator round.
    assert [(c["iteration"], c["role"]) for c in out["calls"]] == \
        [(0, "judge"), (1, "judge"), (1, "evaluator"), (1, "evaluator"), (1, "evaluator")]


def test_all_judge_rounds_unparseable_yields_null_prediction():
    from baselines.shared.backends import get_backend

    backend = get_backend("dummy", responder=lambda msgs: "never json")
    out = _run_program(toy_record(), backend)

    assert out["predicted_agent"] is None and out["predicted_step"] is None
    assert out["confidence"] is None and out["raw"] is None
    assert len(out["calls"]) == 3  # three judge rounds, no evaluator rounds


def test_evaluator_feedback_reaches_the_next_judge():
    from baselines.shared.backends import get_backend

    backend = get_backend("dummy", responder=_responder(10))
    _run_program(toy_record(), backend, max_iters=1)

    judge_prompts = [msgs[-1]["content"] for msgs in backend.calls if _is_judge(msgs)]
    assert len(judge_prompts) == 2
    assert "## Feedback on your previous answers ##" not in judge_prompts[0]
    assert "## Feedback on your previous answers ##" in judge_prompts[1]
    assert "confidence 10/100" in judge_prompts[1]
    # The rule check's verdict is part of the feedback too.
    assert "confidence 100/100" in judge_prompts[1]


def test_gt_flag_reaches_all_prompts():
    """include_gt=True puts the answer line in the Judge and Evaluator prompts."""
    from baselines.shared.backends import get_backend

    record = toy_record()
    record["ground_truth"] = "SECRET-ANSWER"
    for include_gt, expected in ((True, True), (False, False)):
        backend = get_backend("dummy", responder=_responder(95))
        _run_program(record, backend, include_gt=include_gt)
        assert len(backend.calls) == 4
        for msgs in backend.calls:
            assert ("SECRET-ANSWER" in msgs[-1]["content"]) is expected


def test_max_iters_zero_still_runs_one_full_pass():
    """The paper's 'RAFFLES K=0' rows: one Judge pass, still evaluated."""
    from baselines.shared.backends import get_backend

    backend = get_backend("dummy", responder=_responder(10))
    out = _run_program(toy_record(), backend, max_iters=0)
    assert out["n_iterations"] == 1 and len(out["calls"]) == 4
    assert out["predicted_agent"] == "Worker"


# ─────────────────────────────────────────────────────────────────────────────
# predict
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_e2e_and_resume(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"

    res = _predict(data, out)
    assert "3 to run" in res.stdout
    mdir = out / "raffles"
    assert sorted(p.name for p in mdir.glob("[0-9]*.json")) == ["1.json", "2.json", "3.json"]

    doc = json.loads((mdir / "1.json").read_text())
    for key in ("id", "filename", "question_id", "method", "model", "backend",
                "gt_in_prompt", "max_iters", "threshold", "predicted_agent",
                "predicted_step", "gold_agent", "gold_step", "raw", "confidence",
                "n_iterations", "iterations", "calls"):
        assert key in doc, key
    assert doc["method"] == "raffles" and doc["model"] == "dummy"
    # Default GT setting for this baseline is 'without' — the paper setting.
    assert doc["gt_in_prompt"] is False
    assert doc["max_iters"] == 2 and doc["threshold"] == 350
    assert doc["gold_agent"] == "Worker" and doc["gold_step"] == 1

    run_cfg = json.loads((mdir / "_run.json").read_text())
    assert run_cfg["method"] == "raffles" and run_cfg["n_total"] == 3
    assert run_cfg["gt_in_prompt"] is False
    assert run_cfg["max_iters"] == 2 and run_cfg["threshold"] == 350
    assert "request_params" in run_cfg

    # Resume: delete one file, only it re-runs.
    (mdir / "3.json").unlink()
    mtime_1 = (mdir / "1.json").stat().st_mtime_ns
    res2 = _predict(data, out)
    assert "2 done, 1 to run" in res2.stdout
    assert (mdir / "1.json").stat().st_mtime_ns == mtime_1

    res3 = _predict(data, out)
    assert "skip (complete)" in res3.stdout

    res4 = _predict(data, out, "--overwrite")
    assert "3 to run" in res4.stdout


def test_predict_with_gt(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _predict(data, out, "--gt", "with")
    doc = json.loads((out / "raffles" / "1.json").read_text())
    assert doc["gt_in_prompt"] is True
    assert json.loads((out / "raffles" / "_run.json").read_text())["gt_in_prompt"] is True


def test_predict_slice(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _predict(data, out, "--start_idx", "0", "--end_idx", "2")
    assert sorted(p.name for p in (out / "raffles").glob("[0-9]*.json")) == \
        ["1.json", "2.json"]


def test_predict_method_dir_override(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _predict(data, out, "--max-iters", "5", "--method-dir", "raffles.k5")
    assert (out / "raffles.k5" / "1.json").exists()
    assert not (out / "raffles").exists()
    assert json.loads((out / "raffles.k5" / "_run.json").read_text())["max_iters"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# sweep
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = ["ww-api", "correct-error-api", "traceelephant-api"]


@pytest.mark.parametrize("name", CONFIGS)
def test_sweep_dry_run(name):
    res = run_module("baselines.raffles.sweep",
                     "--config", f"baselines/raffles/configs/{name}.yaml", "--dry-run")
    out = res.stdout
    assert "baselines.raffles.predict" in out
    # Default GT setting is 'without': predictions mirror into outputs-nogt/.
    flat = out.replace("\\\n    ", " ")
    assert "outputs-nogt/" in out and "--gt without" in flat
    assert "--max-iters 2" in flat and "--threshold 350" in flat


@pytest.mark.parametrize("name", CONFIGS)
def test_sweep_dry_run_with_gt(name):
    res = run_module("baselines.raffles.sweep", "--gt", "with",
                     "--config", f"baselines/raffles/configs/{name}.yaml", "--dry-run")
    out = res.stdout
    assert "--gt with" in out.replace("\\\n    ", " ")
    assert "outputs-nogt/" not in out


def test_sweep_e2e_dummy(tmp_path):
    data = toy_data_dir(tmp_path)  # subset dir name: "data"
    outputs = tmp_path / "outputs"
    argv = [
        "--config", "baselines/raffles/configs/ww-api.yaml",
        "--gt", "with",  # tmp roots can't be mapped by nogt_root
        "--set", f"data_dir={tmp_path}",
        "--set", "subsets=[data]",
        "--set", f"outputs_root={outputs}",
        "--set", "models=[dummy]",
        "--set", "model_specs={dummy: {backend: dummy}}",
    ]
    run_module("baselines.raffles.sweep", *argv)

    mdir = outputs / "data" / "dummy" / "raffles"
    assert len(list(mdir.glob("[0-9]*.json"))) == 3
    doc = json.loads((mdir / "1.json").read_text())
    assert doc["gt_in_prompt"] is True and doc["max_iters"] == 2

    # Second run: predict skips as complete.
    res2 = run_module("baselines.raffles.sweep", *argv)
    assert "skip (complete)" in res2.stdout


# ─────────────────────────────────────────────────────────────────────────────
# report integration (the shared report reads raffles' method dirs)
# ─────────────────────────────────────────────────────────────────────────────

def test_report_reads_raffles_outputs(tmp_path):
    data = toy_data_dir(tmp_path)
    subset_data = tmp_path / "corpus" / "sub"
    subset_data.mkdir(parents=True)
    for f in data.glob("*.json"):
        (subset_data / f.name).write_text(f.read_text())

    pred_root = tmp_path / "pred"
    _predict(subset_data, pred_root / "sub" / "dummy")

    report_cfg = tmp_path / "report.yaml"
    report_cfg.write_text(f"""
models:  [dummy]
subsets: [sub]
methods: [raffles]
gt: with
data_dir:  {tmp_path / "corpus"}
pred_root: {pred_root}
out_root:  {tmp_path / "reports"}
splits: {{train: 0.3, val: 0.2, test: 0.5}}
seeds: [1, 2]
gt_in_prompt: true
""")
    res = run_module("baselines.raffles.report", "--config", str(report_cfg), "--check-only")
    assert "DONE" in res.stdout
    run_module("baselines.raffles.report", "--config", str(report_cfg))
    assert (tmp_path / "reports" / "summary_mean_over_seeds.tsv").exists()
    per_seed = tmp_path / "reports" / "dummy" / "sub" / "comparison_by_seed.tsv"
    assert per_seed.exists()
    assert "raffles_step@1_test" in per_seed.read_text().splitlines()[0]
