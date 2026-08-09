"""End-to-end CORRECT pipeline runs (dummy backend, subprocess, keyless)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _schemagen(data_dir: Path, out_dir: Path, *extra: str) -> str:
    res = run_module(
        "baselines.correct.schemagen",
        "--backend", "dummy", "--model", "dummy-model", "--model-name", "dummy",
        "--input", str(data_dir), "--output", str(out_dir), *extra,
    )
    return res.stdout


def test_schemagen_e2e_and_resume(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"

    out1 = _schemagen(data, out)
    assert "3 to run" in out1
    sdir = out / "schemagen"
    files = sorted(p.name for p in sdir.glob("[0-9]*.json"))
    assert files == ["1.json", "2.json", "3.json"]

    doc = json.loads((sdir / "2.json").read_text())
    for key in ("id", "filename", "question_id", "method", "model", "backend",
                "predicted_agent", "predicted_step", "gold_agent", "gold_step",
                "raw", "schema", "calls"):
        assert key in doc, key
    assert doc["method"] == "schemagen" and doc["model"] == "dummy"
    assert doc["predicted_agent"] is None and doc["predicted_step"] is None
    assert doc["gold_agent"] == "Worker" and doc["gold_step"] == 1
    assert doc["schema"] and doc["calls"][0]["response"] == doc["raw"]

    run_cfg = json.loads((sdir / "_run.json").read_text())
    assert run_cfg["method"] == "schemagen" and run_cfg["n_total"] == 3
    assert "request_params" in run_cfg

    # Resume: delete one file, only it re-runs.
    (sdir / "2.json").unlink()
    mtime_1 = (sdir / "1.json").stat().st_mtime_ns
    out2 = _schemagen(data, out)
    assert "2 done, 1 to run" in out2
    assert (sdir / "2.json").exists()
    assert (sdir / "1.json").stat().st_mtime_ns == mtime_1

    out3 = _schemagen(data, out)
    assert "skip (complete)" in out3

    # --overwrite clears and redoes everything.
    out4 = _schemagen(data, out, "--overwrite")
    assert "3 to run" in out4


def test_schemagen_slice(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _schemagen(data, out, "--start_idx", "0", "--end_idx", "2")
    files = sorted(p.name for p in (out / "schemagen").glob("[0-9]*.json"))
    assert files == ["1.json", "2.json"]


# ─────────────────────────────────────────────────────────────────────────────
# predict
# ─────────────────────────────────────────────────────────────────────────────

def _write_sims(path: Path, mapping: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping))
    return path


def _predict(data_dir: Path, out_dir: Path, method: str, *extra: str,
             check: bool = True) -> subprocess.CompletedProcess:
    return run_module(
        "baselines.correct.predict",
        "--backend", "dummy", "--model", "dummy-model", "--model-name", "dummy",
        "--input", str(data_dir), "--output", str(out_dir),
        "--method", method, *extra, check=check,
    )


def test_predict_correct_e2e_and_resume(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _schemagen(data, out)
    sims = _write_sims(tmp_path / "out" / "_similarities" / "bge-m3.json",
                       {"1": [2, 3], "2": [1, 3], "3": [1, 2]})

    artifact_args = ("--schemata-dir", str(out / "schemagen"),
                     "--similarities", str(sims),
                     "--num-schemata", "2", "--schema-model", "dummy")
    res = _predict(data, out, "correct", *artifact_args)
    assert "3 to run" in res.stdout
    mdir = out / "correct"
    assert sorted(p.name for p in mdir.glob("[0-9]*.json")) == ["1.json", "2.json", "3.json"]

    doc = json.loads((mdir / "1.json").read_text())
    for key in ("id", "filename", "question_id", "method", "model", "backend",
                "gt_in_prompt", "schema_model", "num_schemata", "schema_cases",
                "predicted_agent", "predicted_step", "gold_agent", "gold_step",
                "raw", "calls"):
        assert key in doc, key
    assert doc["method"] == "correct" and doc["gt_in_prompt"] is False
    assert doc["schema_cases"] == [2, 3] and doc["num_schemata"] == 2
    assert doc["schema_model"] == "dummy"

    run_cfg = json.loads((mdir / "_run.json").read_text())
    assert run_cfg["gt_in_prompt"] is False
    assert run_cfg["num_schemata"] == 2 and run_cfg["n_with_schemata"] == 3
    assert run_cfg["similarities"] == str(sims)

    # Resume.
    (mdir / "3.json").unlink()
    res2 = _predict(data, out, "correct", *artifact_args)
    assert "2 done, 1 to run" in res2.stdout
    res3 = _predict(data, out, "correct", *artifact_args)
    assert "skip (complete)" in res3.stdout


def test_predict_baseline_needs_no_artifacts(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _predict(data, out, "correct_baseline")
    doc = json.loads((out / "correct_baseline" / "2.json").read_text())
    assert doc["schema_cases"] == [] and doc["num_schemata"] == 0
    assert doc["schema_model"] is None


def test_predict_gt_axis_recorded(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _predict(data, out, "correct_baseline", "--gt", "with")
    doc = json.loads((out / "correct_baseline" / "1.json").read_text())
    assert doc["gt_in_prompt"] is True
    run_cfg = json.loads((out / "correct_baseline" / "_run.json").read_text())
    assert run_cfg["gt_in_prompt"] is True


def test_predict_correct_missing_artifacts_exits(tmp_path):
    data = toy_data_dir(tmp_path)
    res = _predict(data, tmp_path / "out", "correct", check=False)
    assert res.returncode != 0
    err = res.stdout + res.stderr
    assert "baselines.correct.schemagen" in err and "baselines.correct.similarity" in err


def test_predict_method_dir_override(tmp_path):
    data = toy_data_dir(tmp_path)
    out = tmp_path / "out" / "dummy"
    _schemagen(data, out)
    sims = _write_sims(tmp_path / "out" / "_similarities" / "bge-m3.json",
                       {"1": [2], "2": [1], "3": [1]})
    _predict(data, out, "correct", "--method-dir", "correct.k1",
             "--schemata-dir", str(out / "schemagen"), "--similarities", str(sims))
    assert (out / "correct.k1" / "1.json").exists()
    assert not (out / "correct").exists()


# ─────────────────────────────────────────────────────────────────────────────
# sweep
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = ["ww", "ww-api", "correct-error", "correct-error-api",
           "traceelephant", "traceelephant-api"]


@pytest.mark.parametrize("name", CONFIGS)
def test_sweep_dry_run(name):
    res = run_module("baselines.correct.sweep",
                     "--config", f"baselines/correct/configs/{name}.yaml", "--dry-run")
    out = res.stdout
    assert "baselines.correct.schemagen" in out
    assert "baselines.correct.similarity" in out
    assert "baselines.correct.predict" in out
    # Default GT setting is 'without': detection mirrors into outputs-nogt/,
    # stage-1/2 artifacts stay under outputs/.
    assert "outputs-nogt/" in out and "--gt without" in out.replace("\\\n    ", " ")


def test_sweep_e2e_dummy(tmp_path):
    data = toy_data_dir(tmp_path)  # subset dir name: "data"
    outputs = tmp_path / "outputs"
    _write_sims(outputs / "data" / "_similarities" / "bge-m3.json",
                {"1": [2, 3], "2": [1, 3], "3": [1, 2]})
    argv = [
        "--config", "baselines/correct/configs/ww.yaml",
        "--gt", "with",  # tmp roots can't be mapped by nogt_root
        "--set", f"data_dir={tmp_path}",
        "--set", "subsets=[data]",
        "--set", f"outputs_root={outputs}",
        "--set", "models=[dummy]",
        "--set", "schema_model=dummy",
        "--set", "model_specs={dummy: {backend: dummy}}",
        "--set", "num_schemata=2",
        "--set", "embed_model=bge-m3",
    ]
    res = run_module("baselines.correct.sweep", *argv)
    sdir = outputs / "data" / "dummy" / "schemagen"
    assert len(list(sdir.glob("[0-9]*.json"))) == 3
    for method in ("correct", "correct_baseline"):
        mdir = outputs / "data" / "dummy" / method
        assert len(list(mdir.glob("[0-9]*.json"))) == 3, method
    doc = json.loads((outputs / "data" / "dummy" / "correct" / "1.json").read_text())
    assert doc["gt_in_prompt"] is True and doc["schema_cases"] == [2, 3]

    # Second run: schemagen skipped as complete, everything else resumes/skips.
    res2 = run_module("baselines.correct.sweep", *argv)
    assert "schemagen complete" in res2.stdout
    assert res2.stdout.count("skip (complete)") == 2


# ─────────────────────────────────────────────────────────────────────────────
# report integration (shared report reads the correct method dirs)
# ─────────────────────────────────────────────────────────────────────────────

def test_report_reads_correct_outputs(tmp_path):
    data = toy_data_dir(tmp_path)
    subset_data = tmp_path / "corpus" / "sub"
    subset_data.mkdir(parents=True)
    for f in data.glob("*.json"):
        (subset_data / f.name).write_text(f.read_text())

    pred_root = tmp_path / "pred"
    out = pred_root / "sub" / "dummy"
    _predict(subset_data, out, "correct_baseline")

    report_cfg = tmp_path / "report.yaml"
    report_cfg.write_text(f"""
models:  [dummy]
subsets: [sub]
methods: [correct_baseline]
gt: with
data_dir:  {tmp_path / "corpus"}
pred_root: {pred_root}
out_root:  {tmp_path / "reports"}
splits: {{train: 0.3, val: 0.2, test: 0.5}}
seeds: [1, 2]
gt_in_prompt: false
""")
    res = run_module("baselines.correct.report", "--config", str(report_cfg), "--check-only")
    assert "DONE" in res.stdout
    res2 = run_module("baselines.correct.report", "--config", str(report_cfg))
    assert (tmp_path / "reports" / "summary_mean_over_seeds.tsv").exists()
    per_seed = tmp_path / "reports" / "dummy" / "sub" / "comparison_by_seed.tsv"
    assert per_seed.exists()
    assert "correct_baseline_step@1_test" in per_seed.read_text().splitlines()[0]
