"""Incremental writes and resume: OutputWriter + end-to-end predict runs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from baselines.shared.runner import OutputWriter

REPO_ROOT = Path(__file__).resolve().parents[1]


def _toy_data_dir(tmp_path: Path, n: int = 3) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    for i in range(1, n + 1):
        doc = {
            "question_ID": f"q{i}",
            "question": f"question {i}",
            "ground_truth": f"answer {i}",
            "history": [
                {"role": "Orchestrator", "content": "plan"},
                {"role": "Worker", "content": "do"},
            ],
            "mistake_agent": "Worker",
            "mistake_step": 1,
        }
        (d / f"{i}.json").write_text(json.dumps(doc))
    return d


def _predict(data_dir: Path, out_dir: Path, *extra: str) -> str:
    cmd = [
        sys.executable, "-m", "baselines.prompting.predict",
        "--backend", "dummy", "--model", "dummy-model", "--model-name", "dummy",
        "--input", str(data_dir), "--output", str(out_dir),
        "--method", "all_at_once", *extra,
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return res.stdout


def test_writer_done_ids_and_atomicity(tmp_path):
    w = OutputWriter(tmp_path / "m")
    assert w.done_ids() == set()
    w.write("3", {"id": "3"})
    w.write("11", {"id": "11"})
    (tmp_path / "m" / "_run.json").write_text("{}")
    (tmp_path / "m" / "5.json.tmp").write_text("torn")  # crash leftover
    assert w.done_ids() == {"3", "11"}  # _run.json and .tmp ignored
    # a new writer clears stale tmp files but keeps completed outputs
    w2 = OutputWriter(tmp_path / "m")
    assert not list((tmp_path / "m").glob("*.tmp"))
    assert w2.done_ids() == {"3", "11"}


def test_writer_overwrite_clears(tmp_path):
    w = OutputWriter(tmp_path / "m")
    w.write("1", {"id": "1"})
    w.write_run_config({"x": 1})
    w2 = OutputWriter(tmp_path / "m", overwrite=True)
    assert w2.done_ids() == set()
    assert not (tmp_path / "m" / "_run.json").exists()


def test_predict_resume_skips_done(tmp_path):
    data = _toy_data_dir(tmp_path)
    out = tmp_path / "out"
    out1 = _predict(data, out)
    assert "3 to run" in out1
    files = sorted(p.name for p in (out / "all_at_once").glob("[0-9]*.json"))
    assert files == ["1.json", "2.json", "3.json"]

    # Simulate a crash that lost trajectory 2: only it should re-run.
    (out / "all_at_once" / "2.json").unlink()
    mtime_1 = (out / "all_at_once" / "1.json").stat().st_mtime_ns
    out2 = _predict(data, out)
    assert "2 done, 1 to run" in out2
    assert (out / "all_at_once" / "2.json").exists()
    assert (out / "all_at_once" / "1.json").stat().st_mtime_ns == mtime_1  # untouched

    # Fully complete → skip without running anything.
    out3 = _predict(data, out)
    assert "skip (complete)" in out3

    # --overwrite redoes everything.
    out4 = _predict(data, out, "--overwrite")
    assert "0 done, 3 to run" in out4

    run_cfg = json.loads((out / "all_at_once" / "_run.json").read_text())
    assert run_cfg["backend"] == "dummy"
    assert run_cfg["n_total"] == 3 and run_cfg["n_remaining"] == 3
    assert run_cfg["resumed"] is False

    doc = json.loads((out / "all_at_once" / "1.json").read_text())
    assert doc["method"] == "all_at_once" and doc["model"] == "dummy"
    assert doc["gold_agent"] == "Worker" and doc["gold_step"] == 1
    assert doc["calls"] and doc["raw"]
