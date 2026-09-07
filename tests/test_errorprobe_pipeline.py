"""ErrorProbe pipeline tests (vendored parity, dummy backend, keyless).

The faithfulness reference is ``vendored/ERRORPROBE/`` (the authors' own
simplified reproduction), and the parity tests here drive that code itself:

- Analyzer/Verifier prompt and parser parity instantiates the vendored
  ``AnalyzerAgent``/``VerifierAgent`` (with a stub ``litellm`` module — no
  network) and asserts byte-identical prompts and identical parse results.
- Backward-tracing parity drives the real vendored ``BackwardTracer`` with a
  scripted client and asserts our generator issues the identical prompt
  sequence and returns the identical prediction (timestamps excluded — both
  sides stamp ``datetime.now()``).

Records are fed to the vendored side without ``name`` keys, because this
repo's identity rule is ``history[t]["role"]`` and the adaptation routes the
vendored extraction onto its role-fallback branch (``methods.strip_names``).
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED = REPO_ROOT / "vendored" / "ERRORPROBE"


def _import_vendored():
    """Import the vendored package with a stub litellm (keyless, no network)."""
    if "litellm" not in sys.modules:
        stub = types.ModuleType("litellm")

        def _no_api(**kwargs):
            raise RuntimeError("no API calls in tests")

        stub.completion = _no_api
        sys.modules["litellm"] = stub
    if str(VENDORED) not in sys.path:
        sys.path.insert(0, str(VENDORED))
    import simplified_mas  # noqa: F401  (vendored package)
    return simplified_mas


def _vendored_agents():
    sm = _import_vendored()
    config = sm.LLMConfig(str(VENDORED / "config.yaml"))
    assert config.use_backward_tracing is False  # the vendored default
    return sm.AnalyzerAgent(config), sm.VerifierAgent(config)


# ─────────────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _alggen_history(n: int = 20) -> list[dict]:
    return [{"role": f"Agent{i % 3}", "content": f"content of turn {i} " + "x" * 600}
            for i in range(n)]


def _handcrafted_history() -> list[dict]:
    return [
        {"role": "human", "content": "please do the task"},
        {"role": "Orchestrator (thought)", "content": "thinking: make a plan"},
        {"role": "WebSurfer", "content": '{"command_name": "Editor.create_file"}'},
        {"role": "Orchestrator (-> WebSurfer)", "content": "go browse"},
        {"role": "assistant", "content": "generic turn"},
    ]


PARITY_TRACES = {
    "alggen-long": {"question": "solve the spreadsheet task", "history": _alggen_history(20)},
    "alggen-short": {"question": "short one", "history": _alggen_history(3)},
    "handcrafted": {"question": "browse and answer", "history": _handcrafted_history()},
    "generic-only": {"question": "who did it", "history": [
        {"role": "human", "content": "a"}, {"role": "assistant", "content": "b"}]},
}

ANALYSIS_RESPONSES = {
    "legacy-range": json.dumps({"error_start_turn": 3, "error_end_turn": 4,
                                "error_family": "fail_task_spec", "error_agent": "Agent0",
                                "error_reason": "bad", "confidence": 0.9}),
    "single-step": json.dumps({"error_step": 7, "error_family": "context_loss",
                               "error_agent": "Agent1", "error_reason": "lost", "confidence": 0.4}),
    "range-keys": json.dumps({"error_step_start": 1, "error_step_end": 2,
                              "error_family": "ignored_input", "tool_or_component": "Agent2",
                              "error_reason": "ignored", "confidence": 0.7}),
    "fenced": ('```json\n' + json.dumps({"error_start_turn": 0, "error_end_turn": 0,
               "error_family": "premature_termination", "error_agent": "Agent0",
               "error_reason": "stopped", "confidence": 0.5}) + '\n```'),
    "no-step-keys": json.dumps({"error_family": "task_derailment",
                                "error_agent": "Agent1", "error_reason": "off", "confidence": 0.6}),
    "garbage": "I think the error is somewhere in the middle.",
}

VERIFICATION_RESPONSE = json.dumps({
    "verified": True, "verification_confidence": 0.8, "impact_if_fixed": 0.7,
    "evidence_count": 2, "suggested_fix": {"type": "guard", "fix_text": "check it"},
    "verification_notes": "looks right",
})


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


def toy_record(i: int = 1, n_turns: int = 2) -> dict:
    roles = ["Orchestrator", "Worker"]
    return {
        "id": str(i), "filename": f"{i}.json", "question_id": f"q{i}",
        "question": f"question {i}", "ground_truth": f"answer {i}",
        "history": [{"role": roles[t % 2], "content": f"turn {t} of {i}"}
                    for t in range(n_turns)],
        "gold_agent": "Worker", "gold_step": 1,
    }


def run_module(module: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", module, *args]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=check)


def _predict(data_dir: Path, out_dir: Path, *extra: str,
             check: bool = True) -> subprocess.CompletedProcess:
    return run_module(
        "baselines.errorprobe.predict",
        "--backend", "dummy", "--model", "dummy-model", "--model-name", "dummy",
        "--input", str(data_dir), "--output", str(out_dir), *extra, check=check,
    )


def _run_program(record, backend, mode: str = "truncated", **kwargs):
    from baselines.shared.runner import run_batched
    from baselines.errorprobe.methods import METHODS

    program = METHODS["errorprobe" if mode == "truncated" else "errorprobe_bt"]
    out = {}
    run_batched([(record, program(record, **kwargs))], backend,
                lambda key, pred: out.update(pred))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer/Verifier parity — drives the vendored classes themselves
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(PARITY_TRACES))
def test_analyzer_prompt_parity(name):
    from baselines.errorprobe.prompts import build_analyzer_prompt

    analyzer, _ = _vendored_agents()
    trace = PARITY_TRACES[name]
    assert build_analyzer_prompt(trace) == analyzer._build_analysis_prompt(trace, None)


@pytest.mark.parametrize("name", sorted(ANALYSIS_RESPONSES))
def test_analysis_parser_parity(name):
    from baselines.errorprobe.prompts import parse_analysis

    analyzer, _ = _vendored_agents()
    response = ANALYSIS_RESPONSES[name]
    ours, theirs = parse_analysis(response), analyzer._parse_llm_response(response)
    # The vendored dict has an extra "context" mirror of the raw JSON; compare
    # the decision-relevant fields plus context for equality.
    assert ours == theirs


@pytest.mark.parametrize("trace_name", ["alggen-long", "handcrafted"])
@pytest.mark.parametrize("resp_name", ["legacy-range", "single-step", "garbage"])
def test_verifier_prompt_parity(trace_name, resp_name):
    from baselines.errorprobe.prompts import build_verifier_prompt, parse_analysis

    analyzer, verifier = _vendored_agents()
    trace = PARITY_TRACES[trace_name]
    analysis = parse_analysis(ANALYSIS_RESPONSES[resp_name])
    assert analysis == analyzer._parse_llm_response(ANALYSIS_RESPONSES[resp_name])
    assert build_verifier_prompt(trace, analysis) == \
        verifier._build_verification_prompt(trace, analysis)


@pytest.mark.parametrize("response", [
    VERIFICATION_RESPONSE,
    "```json\n" + VERIFICATION_RESPONSE + "\n```",
    "not json at all",
    json.dumps({"verified": False, "impact_if_fixed": "high"}),  # float() fails → sentinel
])
def test_verification_parser_parity(response):
    from baselines.errorprobe.prompts import parse_verification

    _, verifier = _vendored_agents()
    theirs = verifier._parse_verification_response(response)
    ours = parse_verification(response)
    theirs_dict = theirs.to_dict()
    if "parse_error" in ours["probe_details"]:
        # Exception wording may differ between dataclass and dict paths; the
        # decision-relevant fields must still match.
        assert "parse_error" in theirs_dict["probe_details"]
        ours = {**ours, "probe_details": {}}
        theirs_dict = {**theirs_dict, "probe_details": {}}
    assert ours == theirs_dict


def test_mast_taxonomy_matches_vendored():
    from baselines.errorprobe.prompts import MAST_FAMILIES

    _import_vendored()
    import core
    families = core.get_all_mast_families()
    assert list(MAST_FAMILIES) == families
    for family in families:
        assert MAST_FAMILIES[family] == core.ERROR_FAMILIES[family]


# ─────────────────────────────────────────────────────────────────────────────
# Backward-tracing parity — drives the vendored BackwardTracer itself
# ─────────────────────────────────────────────────────────────────────────────

def _bt_script(prompt: str) -> str:
    """Deterministic scripted responses, a pure function of the prompt."""
    if "Is this turn RELEVANT" in prompt:
        t = int(prompt.split("CURRENT TURN (Turn ")[1].split(")")[0])
        if t % 3 == 0:
            return json.dumps({"is_relevant": True, "score": 0.9, "reason": "r"})
        return "no json here"  # exercises the has-action fallback
    if "TURN BEING ANALYZED (Turn " in prompt:
        t = int(prompt.split("TURN BEING ANALYZED (Turn ")[1].split(")")[0])
        conf = max(0.0, round(0.9 - 0.05 * t, 2))
        return json.dumps({"is_potential_cause": True, "error_type": "logic_error",
                           "confidence": conf, "reasoning": f"reason {t}",
                           "observations": [f"obs a {t}", f"obs b {t}"]})
    if "Summarize these error tracing findings" in prompt:
        return "  a running summary of findings  "
    if "Which of these observations are RELEVANT" in prompt:
        return "1, 3, oops, 5"
    if "Which hypotheses should be merged" in prompt:
        return "1,2"
    if "final error diagnosis" in prompt:
        return json.dumps({"mistake_agent": "Agent0", "mistake_step": 6,
                           "mistake_type": "logic_error", "mistake_reason": "final",
                           "confidence": 0.8})
    raise AssertionError(f"unexpected prompt: {prompt[:80]!r}")


def _bt_script_all_garbage(prompt: str) -> str:
    if "Is this turn RELEVANT" in prompt or "TURN BEING ANALYZED" in prompt:
        return "garbage"
    raise AssertionError(f"unexpected prompt: {prompt[:80]!r}")


def _strip_timestamps(obj):
    if isinstance(obj, dict):
        return {k: _strip_timestamps(v) for k, v in obj.items() if k != "timestamp"}
    if isinstance(obj, list):
        return [_strip_timestamps(v) for v in obj]
    return obj


def _drive_vendored_bt(history: list[dict], question: str, script):
    sm = _import_vendored()
    from simplified_mas.backward_tracer import BackwardTracer, TracingConfig
    from simplified_mas.trace_index import TraceIndex

    class FakeClient:
        def __init__(self):
            self.prompts = []

        def invoke(self, prompt, max_tokens=1000):
            self.prompts.append(prompt)
            return script(prompt)

    client = FakeClient()
    # The exact config AnalyzerAgent builds when backward tracing is enabled.
    tracer = BackwardTracer(client, TracingConfig(
        max_turns_to_examine=100, context_window_radius=5,
        confidence_threshold=0.75, min_turns_before_stopping=10))
    # The vendored TraceIndex reads the agent from `name`.
    vendored_trace = {"question": question,
                      "history": [{**t, "name": t["role"]} for t in history]}
    prediction = tracer.trace(TraceIndex(vendored_trace), question)
    return client.prompts, prediction


def _drive_our_bt(history: list[dict], question: str, script):
    from baselines.errorprobe.backward import TraceIndex, backward_trace

    ti = TraceIndex({"question": question, "history": history})
    gen = backward_trace(ti, question)
    prompts_seen = []
    item = next(gen)
    while True:
        _meta, prompts = item
        prompt = prompts[0][0]["content"]
        prompts_seen.append(prompt)
        try:
            item = gen.send([script(prompt)])
        except StopIteration as done:
            return prompts_seen, done.value


@pytest.mark.parametrize("scenario, script, n_turns", [
    ("rich-walk", _bt_script, 20),        # summarize, prune, merge, early stop
    ("fallback", _bt_script_all_garbage, 6),  # no hypotheses → fallback prediction
])
def test_backward_tracing_parity(scenario, script, n_turns):
    history = [{"role": f"Agent{i % 3}", "content": f"content {i}"} for i in range(n_turns)]
    if scenario == "fallback":
        # One tool-flavored turn exercises the relevance parse-failure fallback
        # (has-action → relevant) followed by an unparseable root-cause answer.
        history[2] = {"role": "Agent2", "content": '{"command_name": "Editor.edit"}'}
    question = "the toy task"

    vendored_prompts, vendored_pred = _drive_vendored_bt(history, question, script)
    our_prompts, our_pred = _drive_our_bt(history, question, script)

    assert our_prompts == vendored_prompts
    assert _strip_timestamps(our_pred) == _strip_timestamps(vendored_pred)


def test_backward_tracing_rich_walk_stops_early():
    history = [{"role": f"Agent{i % 3}", "content": f"content {i}"} for i in range(20)]
    prompts, pred = _drive_our_bt(history, "the toy task", _bt_script)
    assert pred["method"] == "backward_tracing"
    assert pred["mistake_step"] == 6 and pred["mistake_agent"] == "Agent0"
    # Confidence 0.75 is first reached at turn 3, i.e. after examining 17 of
    # the 20 turns (19 down to 3) — the walk must stop there, not at turn 0.
    assert pred["turns_examined"] == 17
    # The maintenance interval fired once: exactly one prune and one merge.
    assert sum("observations are RELEVANT" in p for p in prompts) == 1
    assert sum("should be merged" in p for p in prompts) == 1


# ─────────────────────────────────────────────────────────────────────────────
# programs (truncated + backward) on the dummy backend
# ─────────────────────────────────────────────────────────────────────────────

def _truncated_responder(analysis_json: str):
    def respond(msgs):
        prompt = msgs[-1]["content"]
        if "You are verifying an error analysis." in prompt:
            return VERIFICATION_RESPONSE
        return analysis_json
    return respond


def test_truncated_program_remaps_to_absolute():
    from baselines.shared.backends import get_backend

    record = toy_record(n_turns=20)  # window offset = 5
    backend = get_backend("dummy", responder=_truncated_responder(
        ANALYSIS_RESPONSES["legacy-range"]))
    out = _run_program(record, backend)
    assert out["predicted_step"] == 3 + 5
    assert out["predicted_span_raw"] == [3, 4]
    assert out["predicted_span"] == [8, 9]
    assert out["predicted_agent"] == "Agent0"
    assert out["error_family"] == "fail_task_spec"
    assert [c["role"] for c in out["calls"]] == ["analyzer", "verifier"]
    assert out["evidence"]["verified"] is True
    assert out["hypothesis_confidence"] == pytest.approx(0.9 * 0.8)


def test_truncated_program_short_trace_no_offset():
    from baselines.shared.backends import get_backend

    record = toy_record(n_turns=2)
    backend = get_backend("dummy", responder=_truncated_responder(
        ANALYSIS_RESPONSES["single-step"]))
    out = _run_program(record, backend)
    assert out["predicted_step"] == 7 and out["predicted_span_raw"] == [7, 7]


def test_truncated_program_parse_failure_sentinel():
    from baselines.shared.backends import get_backend

    record = toy_record(n_turns=20)
    backend = get_backend("dummy", responder=_truncated_responder("garbage"))
    out = _run_program(record, backend)
    # The vendored sentinel: agent "unknown", span (0, 0) — remapped uniformly.
    assert out["predicted_agent"] == "unknown"
    assert out["predicted_step"] == 0 + 5
    assert out["analysis_confidence"] == 0.0


def test_verifier_sees_the_raw_window_local_span():
    """The Verifier prompt gets the raw span, not the remapped one (vendored)."""
    from baselines.shared.backends import get_backend

    record = toy_record(n_turns=20)
    backend = get_backend("dummy", responder=_truncated_responder(
        ANALYSIS_RESPONSES["legacy-range"]))
    _run_program(record, backend)
    verifier_prompt = backend.calls[1][-1]["content"]
    assert "Error turns: 3 to 4" in verifier_prompt


def test_strip_names_routes_the_role_branch():
    """A record with name='assistant' turns prompts identically to a name-less one."""
    from baselines.errorprobe.methods import strip_names
    from baselines.errorprobe.prompts import build_analyzer_prompt

    record = toy_record()
    with_names = {**record, "history": [{**t, "name": "assistant"}
                                        for t in record["history"]]}
    assert build_analyzer_prompt(strip_names(with_names)) == \
        build_analyzer_prompt(record)
    # Un-stripped, the vendored name-first branch would answer "assistant".
    assert "Agent: assistant" in build_analyzer_prompt(with_names)
    assert "Agent: assistant" not in build_analyzer_prompt(record)


def test_gt_flag_reaches_task_prompts_only():
    from baselines.shared.backends import get_backend

    def respond(msgs):
        prompt = msgs[-1]["content"]
        if "verifying an error analysis" in prompt:
            return VERIFICATION_RESPONSE
        if "expert at analyzing failed multi-agent" in prompt:
            return ANALYSIS_RESPONSES["legacy-range"]
        return _bt_script(prompt)

    record = toy_record(n_turns=12)
    record["history"] = [{"role": f"Agent{i % 3}", "content": f"content {i}"}
                         for i in range(12)]
    record["ground_truth"] = "SECRET-ANSWER"

    for mode in ("truncated", "backward"):
        for include_gt in (True, False):
            backend = get_backend("dummy", responder=respond)
            _run_program(record, backend, mode=mode, include_gt=include_gt)
            saw_task_prompt = False
            for msgs in backend.calls:
                prompt = msgs[-1]["content"]
                # Task-carrying prompts interpolate the question (analyzer;
                # backward root-cause and synthesis). The verifier's literal
                # "TASK: Verify ..." heading carries no task text.
                carries_task = record["question"] in prompt
                saw_task_prompt |= carries_task
                assert ("SECRET-ANSWER" in prompt) is (include_gt and carries_task), \
                    (mode, include_gt, prompt[:60])
            assert saw_task_prompt


def test_backward_program_output_doc():
    from baselines.shared.backends import get_backend

    def respond(msgs):
        return _bt_script(msgs[-1]["content"]) \
            if "verifying an error analysis" not in msgs[-1]["content"] \
            else VERIFICATION_RESPONSE

    record = toy_record(n_turns=20)
    record["history"] = [{"role": f"Agent{i % 3}", "content": f"content {i}"}
                         for i in range(20)]
    backend = get_backend("dummy", responder=respond)
    out = _run_program(record, backend, mode="backward")
    assert out["predicted_agent"] == "Agent0" and out["predicted_step"] == 6
    assert out["bt_method"] == "backward_tracing"
    assert out["turns_examined"] == 17 and out["total_turns"] == 20
    assert out["raw"] and "mistake_step" in out["raw"]
    roles = [c["role"] for c in out["calls"]]
    assert roles[-1] == "verifier" and roles[-2] == "synthesize"
    assert {"relevance", "root_cause", "summarize", "prune", "merge"} <= set(roles)
    assert all("turn_idx" in c for c in out["calls"]
               if c["role"] in ("relevance", "root_cause"))


# ─────────────────────────────────────────────────────────────────────────────
# predict
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_e2e_and_resume(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"

    res = _predict(data, out)
    assert "3 to run" in res.stdout
    mdir = out / "errorprobe"
    assert sorted(p.name for p in mdir.glob("[0-9]*.json")) == ["1.json", "2.json", "3.json"]

    doc = json.loads((mdir / "1.json").read_text())
    for key in ("id", "filename", "question_id", "method", "mode", "model",
                "backend", "gt_in_prompt", "predicted_agent", "predicted_step",
                "gold_agent", "gold_step", "raw", "calls", "error_family",
                "predicted_span_raw", "evidence", "hypothesis_confidence"):
        assert key in doc, key
    assert doc["method"] == "errorprobe" and doc["mode"] == "truncated"
    assert doc["model"] == "dummy"
    # Default GT setting for this baseline is 'without' — the vendored setting.
    assert doc["gt_in_prompt"] is False
    assert doc["gold_agent"] == "Worker" and doc["gold_step"] == 1

    run_cfg = json.loads((mdir / "_run.json").read_text())
    assert run_cfg["method"] == "errorprobe" and run_cfg["mode"] == "truncated"
    assert run_cfg["n_total"] == 3 and run_cfg["gt_in_prompt"] is False
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


def test_predict_backward_mode(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _predict(data, out, "--mode", "backward")
    mdir = out / "errorprobe_bt"
    assert sorted(p.name for p in mdir.glob("[0-9]*.json")) == ["1.json", "2.json", "3.json"]
    doc = json.loads((mdir / "1.json").read_text())
    assert doc["method"] == "errorprobe_bt" and doc["mode"] == "backward"
    # Default responder answers are unparseable everywhere → no hypotheses →
    # the vendored fallback prediction from the symptom observation.
    assert doc["bt_method"] == "backward_tracing_fallback"
    assert doc["predicted_step"] == 1  # the final turn of the 2-turn toy trace
    assert json.loads((mdir / "_run.json").read_text())["mode"] == "backward"


def test_predict_with_gt(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _predict(data, out, "--gt", "with")
    doc = json.loads((out / "errorprobe" / "1.json").read_text())
    assert doc["gt_in_prompt"] is True
    assert json.loads((out / "errorprobe" / "_run.json").read_text())["gt_in_prompt"] is True


def test_predict_slice(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _predict(data, out, "--start_idx", "0", "--end_idx", "2")
    assert sorted(p.name for p in (out / "errorprobe").glob("[0-9]*.json")) == \
        ["1.json", "2.json"]


def test_predict_method_dir_override(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _predict(data, out, "--method-dir", "errorprobe.x")
    assert (out / "errorprobe.x" / "1.json").exists()
    assert not (out / "errorprobe").exists()


# ─────────────────────────────────────────────────────────────────────────────
# sweep
# ─────────────────────────────────────────────────────────────────────────────

# correct-error ships truncated-only (cost); the other two ship both modes.
CONFIG_MODES = {"ww-api": ["truncated", "backward"],
                "traceelephant-api": ["truncated", "backward"],
                "correct-error-api": ["truncated"]}


@pytest.mark.parametrize("name", sorted(CONFIG_MODES))
def test_sweep_dry_run(name):
    res = run_module("baselines.errorprobe.sweep",
                     "--config", f"baselines/errorprobe/configs/{name}.yaml", "--dry-run")
    out = res.stdout
    flat = out.replace("\\\n    ", " ")
    assert "baselines.errorprobe.predict" in out
    # Default GT setting is 'without': predictions mirror into outputs-nogt/.
    assert "outputs-nogt/" in out and "--gt without" in flat
    for mode in CONFIG_MODES[name]:
        assert f"--mode {mode}" in flat
    for mode in {"truncated", "backward"} - set(CONFIG_MODES[name]):
        assert f"--mode {mode}" not in flat


def test_sweep_dry_run_api_params():
    """The API specs reach predict as verbatim --api-param pairs, no vLLM knobs.

    Both specs carry the deliberate handicap (configs/README.md): gpt-4o the
    qwen decode settings, gpt-5 the lowest reasoning effort, since a reasoning
    model takes max_completion_tokens and no temperature. Their paper blocks
    reach the child too; only vLLM knobs are dropped for an API backend.
    """
    res = run_module("baselines.errorprobe.sweep",
                     "--config", "baselines/errorprobe/configs/ww-api.yaml",
                     "--set", "subsets=[hand-crafted]", "--set", "modes=[truncated,paper]", "--dry-run")
    flat = res.stdout.replace("\\\n    ", " ")
    cmds = [" ".join(line.split()) for line in flat.splitlines()
            if "baselines.errorprobe.predict" in line]
    assert len(cmds) == 4
    gpt4o = [c for c in cmds if "--model-name gpt-4o" in c]
    gpt5 = [c for c in cmds if "--model-name gpt-5" in c]
    assert len(gpt4o) == 2 and len(gpt5) == 2
    for c in gpt4o:
        assert "--backend openai --model gpt-4o" in c
        assert "--api-param max_tokens=512" in c and "--api-param temperature=1.0" in c
        assert "--api-param top_p=0.95" in c
    for c in gpt5:
        assert "--backend openai --model gpt-5" in c
        assert "--api-param max_completion_tokens=4096" in c
        assert "reasoning_effort=" in c and "minimal" in c and "temperature" not in c
    for c in cmds:   # vLLM-only knobs never reach an API model
        assert "--temperature" not in c and "--gen_max_tokens" not in c and "--dtype" not in c
        if "--mode paper" in c:   # the per-spec paper block does
            assert "--max-hypotheses 1" in c and "--condensed-chars 10000" in c
            assert "--chunk-chars 12000" in c
        else:
            assert "--max-hypotheses" not in c


def test_sweep_dry_run_with_gt():
    res = run_module("baselines.errorprobe.sweep", "--gt", "with",
                     "--config", "baselines/errorprobe/configs/ww-api.yaml", "--dry-run")
    out = res.stdout
    assert "--gt with" in out.replace("\\\n    ", " ")
    assert "outputs-nogt/" not in out


def test_sweep_e2e_dummy(tmp_path):
    data = toy_data_dir(tmp_path)  # subset dir name: "data"
    outputs = tmp_path / "outputs"
    argv = [
        "--config", "baselines/errorprobe/configs/ww-api.yaml",
        "--gt", "with",  # tmp roots can't be mapped by nogt_root
        "--set", f"data_dir={tmp_path}",
        "--set", "subsets=[data]",
        "--set", f"outputs_root={outputs}",
        "--set", "models=[dummy]",
        "--set", "modes=[truncated]",
        "--set", "model_specs={dummy: {backend: dummy}}",
    ]
    run_module("baselines.errorprobe.sweep", *argv)

    mdir = outputs / "data" / "dummy" / "errorprobe"
    assert len(list(mdir.glob("[0-9]*.json"))) == 3
    doc = json.loads((mdir / "1.json").read_text())
    assert doc["gt_in_prompt"] is True and doc["mode"] == "truncated"

    # Second run: predict skips as complete.
    res2 = run_module("baselines.errorprobe.sweep", *argv)
    assert "skip (complete)" in res2.stdout


# ─────────────────────────────────────────────────────────────────────────────
# report integration (the shared report reads errorprobe's method dirs)
# ─────────────────────────────────────────────────────────────────────────────

def test_report_reads_errorprobe_outputs(tmp_path):
    data = toy_data_dir(tmp_path)
    subset_data = tmp_path / "corpus" / "sub"
    subset_data.mkdir(parents=True)
    for f in data.glob("*.json"):
        (subset_data / f.name).write_text(f.read_text())

    pred_root = tmp_path / "pred"
    _predict(subset_data, pred_root / "sub" / "dummy")
    _predict(subset_data, pred_root / "sub" / "dummy", "--mode", "backward")

    report_cfg = tmp_path / "report.yaml"
    report_cfg.write_text(f"""
models:  [dummy]
subsets: [sub]
methods: [errorprobe, errorprobe_bt]
gt: with
data_dir:  {tmp_path / "corpus"}
pred_root: {pred_root}
out_root:  {tmp_path / "reports"}
splits: {{train: 0.3, val: 0.2, test: 0.5}}
seeds: [1, 2]
gt_in_prompt: true
""")
    res = run_module("baselines.errorprobe.report", "--config", str(report_cfg), "--check-only")
    assert "DONE" in res.stdout
    run_module("baselines.errorprobe.report", "--config", str(report_cfg))
    assert (tmp_path / "reports" / "summary_mean_over_seeds.tsv").exists()
    per_seed = tmp_path / "reports" / "dummy" / "sub" / "comparison_by_seed.tsv"
    assert per_seed.exists()
    header = per_seed.read_text().splitlines()[0]
    assert "errorprobe_step@1_test" in header and "errorprobe_bt_step@1_test" in header
