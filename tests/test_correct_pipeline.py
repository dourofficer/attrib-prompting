"""End-to-end CORRECT pipeline runs (dummy backend, subprocess, keyless)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
