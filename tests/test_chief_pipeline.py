"""End-to-end CHIEF pipeline runs (dummy backend, subprocess, keyless).

Retrieval is stubbed rather than run: ``ragprep`` needs faiss and
sentence-transformers, which are an optional extra, so the tests that need a
stage-1 artifact write one directly. The one test that does exercise
``ragprep`` skips when those packages are missing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

RAG_BLOCK = "[RAG Example 1]\nQuestion: a similar task\nAnnotated Steps:\n1. Look it up."


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


def _write_rag(path: Path, ids=("1", "2", "3")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({i: RAG_BLOCK for i in ids}))
    return path


def _predict(data_dir: Path, out_dir: Path, *extra: str,
             check: bool = True) -> subprocess.CompletedProcess:
    return run_module(
        "baselines.chief.predict",
        "--backend", "dummy", "--model", "dummy-model", "--model-name", "dummy",
        "--input", str(data_dir), "--output", str(out_dir), *extra, check=check,
    )


# ─────────────────────────────────────────────────────────────────────────────
# the six-call program
# ─────────────────────────────────────────────────────────────────────────────

def test_program_logs_all_six_calls():
    """Every trajectory records one response per stage, in order."""
    from baselines.shared.backends import get_backend
    from baselines.shared.runner import run_batched
    from baselines.chief.methods import chief_program

    record = {"id": "1", "history": [{"role": "A", "content": "x"}],
              "question": "q", "ground_truth": "gt"}
    backend = get_backend("dummy")
    out = {}
    run_batched([(record, chief_program(record))], backend,
                lambda key, pred: out.update(pred))

    assert [c["stage"] for c in out["calls"]] == [1, 2, 3, 4, 5, 6]
    assert len(backend.calls) == 6
    assert out["predicted_agent"] == "DummyAgent" and out["predicted_step"] == 0
    assert out["raw"] == out["calls"][-1]["response"]


def test_program_survives_a_broken_stage():
    """A malformed stage output ends the trajectory without losing the transcript."""
    from baselines.shared.backends import get_backend
    from baselines.shared.runner import run_streaming
    from baselines.chief.methods import chief_program

    record = {"id": "1", "history": [{"role": "A", "content": "x"}],
              "question": "q", "ground_truth": "gt"}
    # Stage 6 never yields an "Agent Name:" line → no prediction, but the five
    # earlier responses are still on record.
    backend = get_backend("dummy", responder=lambda msgs: "unparseable")
    out = {}
    run_streaming([(record, chief_program(record))], backend,
                  lambda key, pred: out.update(pred))

    assert [c["stage"] for c in out["calls"]] == [1, 2, 3, 4, 5, 6]
    assert out["predicted_agent"] is None and out["predicted_step"] is None


def test_gt_flag_reaches_the_prompts():
    """--gt without removes the answer from all six prompts, not just the first."""
    from baselines.shared.backends import get_backend
    from baselines.shared.runner import run_batched
    from baselines.chief.methods import chief_program

    record = {"id": "1", "history": [{"role": "A", "content": "x"}],
              "question": "q", "ground_truth": "SECRET-ANSWER"}
    for include_gt, expected in ((True, True), (False, False)):
        backend = get_backend("dummy")
        run_batched([(record, chief_program(record, include_gt=include_gt))],
                    backend, lambda key, pred: None)
        assert len(backend.calls) == 6
        seen = any("SECRET-ANSWER" in m["content"]
                   for msgs in backend.calls for m in msgs)
        assert seen is expected


def test_rag_text_is_injected_verbatim():
    from baselines.shared.backends import get_backend
    from baselines.shared.runner import run_batched
    from baselines.chief.methods import chief_program

    record = {"id": "1", "history": [{"role": "A", "content": "x"}],
              "question": "q", "ground_truth": "gt"}
    backend = get_backend("dummy")
    run_batched([(record, chief_program(record, rag_text=RAG_BLOCK))],
                backend, lambda key, pred: None)

    stage1 = backend.calls[0][-1]["content"]
    assert RAG_BLOCK in stage1
    assert "Here is the retrieved reference example:" in stage1
    # Retrieval feeds stage 1 only.
    assert RAG_BLOCK not in backend.calls[1][-1]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# predict
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_e2e_and_resume(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    rag = _write_rag(tmp_path / "art" / "data" / "rag" / "all-MiniLM-L6-v2.json")

    res = _predict(data, out, "--rag-texts", str(rag))
    assert "3 to run" in res.stdout
    mdir = out / "chief"
    assert sorted(p.name for p in mdir.glob("[0-9]*.json")) == ["1.json", "2.json", "3.json"]

    doc = json.loads((mdir / "1.json").read_text())
    for key in ("id", "filename", "question_id", "method", "model", "backend",
                "gt_in_prompt", "rag_in_prompt", "predicted_agent", "predicted_step",
                "gold_agent", "gold_step", "raw", "calls"):
        assert key in doc, key
    assert doc["method"] == "chief" and doc["model"] == "dummy"
    # Default GT setting for this baseline is 'with' — the vendored bytes.
    assert doc["gt_in_prompt"] is True and doc["rag_in_prompt"] is True
    assert [c["stage"] for c in doc["calls"]] == [1, 2, 3, 4, 5, 6]
    assert doc["gold_agent"] == "Worker" and doc["gold_step"] == 1

    run_cfg = json.loads((mdir / "_run.json").read_text())
    assert run_cfg["method"] == "chief" and run_cfg["n_total"] == 3
    assert run_cfg["gt_in_prompt"] is True and run_cfg["n_with_rag"] == 3
    assert "request_params" in run_cfg

    # Resume: delete one file, only it re-runs.
    (mdir / "3.json").unlink()
    mtime_1 = (mdir / "1.json").stat().st_mtime_ns
    res2 = _predict(data, out, "--rag-texts", str(rag))
    assert "2 done, 1 to run" in res2.stdout
    assert (mdir / "1.json").stat().st_mtime_ns == mtime_1

    res3 = _predict(data, out, "--rag-texts", str(rag))
    assert "skip (complete)" in res3.stdout

    res4 = _predict(data, out, "--rag-texts", str(rag), "--overwrite")
    assert "3 to run" in res4.stdout


def test_predict_without_gt(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _predict(data, out, "--gt", "without")
    doc = json.loads((out / "chief" / "1.json").read_text())
    assert doc["gt_in_prompt"] is False and doc["rag_in_prompt"] is False
    assert json.loads((out / "chief" / "_run.json").read_text())["gt_in_prompt"] is False


def test_predict_missing_rag_artifact_exits(tmp_path):
    data = toy_data_dir(tmp_path)
    res = _predict(data, tmp_path / "out", "--rag-texts",
                   str(tmp_path / "nope.json"), check=False)
    assert res.returncode != 0
    assert "baselines.chief.ragprep" in res.stdout + res.stderr


def test_predict_slice(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _predict(data, out, "--start_idx", "0", "--end_idx", "2")
    assert sorted(p.name for p in (out / "chief").glob("[0-9]*.json")) == \
        ["1.json", "2.json"]


# ─────────────────────────────────────────────────────────────────────────────
# ragprep
# ─────────────────────────────────────────────────────────────────────────────

def test_ragprep_e2e(tmp_path):
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")
    data = toy_data_dir(tmp_path)
    art = tmp_path / "art" / "rag" / "all-MiniLM-L6-v2.json"

    res = run_module("baselines.chief.ragprep",
                     "--input", str(data), "--output", str(art))
    assert art.exists(), res.stdout + res.stderr
    texts = json.loads(art.read_text())
    assert sorted(texts) == ["1", "2", "3"]
    # The vendored [1:top_k] slice leaves exactly one exemplar per trajectory.
    assert all(t.count("[RAG Example ") == 1 for t in texts.values())

    meta = json.loads(art.with_name("all-MiniLM-L6-v2.meta.json").read_text())
    assert meta["n_trajectories"] == 3 and meta["top_k"] == 2
    assert meta["kbs"] == ["gaia", "assistantbench"]

    # Skip-if-exists.
    res2 = run_module("baselines.chief.ragprep",
                      "--input", str(data), "--output", str(art))
    assert "skip (exists)" in res2.stdout


def test_ragprep_reports_a_missing_knowledge_base(tmp_path):
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")
    data = toy_data_dir(tmp_path)
    res = run_module("baselines.chief.ragprep",
                     "--input", str(data),
                     "--output", str(tmp_path / "rag.json"),
                     "--rag-root", str(tmp_path / "nowhere"), check=False)
    assert res.returncode != 0
    assert "missing RAG artifact" in res.stdout + res.stderr


# ─────────────────────────────────────────────────────────────────────────────
# sweep
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = ["ww-api", "correct-error-api", "traceelephant-api"]


@pytest.mark.parametrize("name", CONFIGS)
def test_sweep_dry_run(name):
    res = run_module("baselines.chief.sweep",
                     "--config", f"baselines/chief/configs/{name}.yaml", "--dry-run")
    out = res.stdout
    assert "baselines.chief.ragprep" in out
    assert "baselines.chief.predict" in out
    # Default GT setting is 'with' — the vendored bytes carry the answer.
    assert "--gt with" in out.replace("\\\n    ", " ")
    assert "outputs-nogt/" not in out
    # Stage-1 artifacts live in artifacts/, never in an output tree.
    assert "artifacts/" in out and "/rag/" in out


@pytest.mark.parametrize("name", CONFIGS)
def test_sweep_dry_run_without_gt_mirrors(name):
    res = run_module("baselines.chief.sweep", "--gt", "without",
                     "--config", f"baselines/chief/configs/{name}.yaml", "--dry-run")
    out = res.stdout
    assert "outputs-nogt/" in out and "--gt without" in out.replace("\\\n    ", " ")
    # One artifacts root serves both GT settings.
    assert "artifacts-nogt" not in out


def test_sweep_e2e_dummy(tmp_path):
    data = toy_data_dir(tmp_path)  # subset dir name: "data"
    outputs = tmp_path / "outputs"
    artifacts = tmp_path / "artifacts"
    _write_rag(artifacts / "data" / "rag" / "all-MiniLM-L6-v2.json")
    argv = [
        "--config", "baselines/chief/configs/ww-api.yaml",
        "--set", f"data_dir={tmp_path}",
        "--set", "subsets=[data]",
        "--set", f"outputs_root={outputs}",
        "--set", f"artifacts_root={artifacts}",
        "--set", "models=[dummy]",
        "--set", "model_specs={dummy: {backend: dummy}}",
    ]
    run_module("baselines.chief.sweep", *argv)

    mdir = outputs / "data" / "dummy" / "chief"
    assert len(list(mdir.glob("[0-9]*.json"))) == 3
    # A model dir holds method dirs only — artifacts never leak into outputs/.
    assert {p.name for p in (outputs / "data" / "dummy").iterdir()} == {"chief"}
    doc = json.loads((mdir / "1.json").read_text())
    assert doc["gt_in_prompt"] is True and doc["rag_in_prompt"] is True

    # Second run: ragprep skipped (artifact exists), predict skips as complete.
    res2 = run_module("baselines.chief.sweep", *argv)
    assert "skip (complete)" in res2.stdout


# ─────────────────────────────────────────────────────────────────────────────
# report integration (the shared report reads chief's method dirs)
# ─────────────────────────────────────────────────────────────────────────────

def test_report_reads_chief_outputs(tmp_path):
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
methods: [chief]
gt: with
data_dir:  {tmp_path / "corpus"}
pred_root: {pred_root}
out_root:  {tmp_path / "reports"}
splits: {{train: 0.3, val: 0.2, test: 0.5}}
seeds: [1, 2]
gt_in_prompt: true
""")
    res = run_module("baselines.chief.report", "--config", str(report_cfg), "--check-only")
    assert "DONE" in res.stdout
    run_module("baselines.chief.report", "--config", str(report_cfg))
    assert (tmp_path / "reports" / "summary_mean_over_seeds.tsv").exists()
    per_seed = tmp_path / "reports" / "dummy" / "sub" / "comparison_by_seed.tsv"
    assert per_seed.exists()
    assert "chief_step@1_test" in per_seed.read_text().splitlines()[0]
