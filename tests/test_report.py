"""Evaluation semantics (mirroring attribscope's main_table.py) + table shape."""
from __future__ import annotations

import json
from pathlib import Path

from baselines.prompting.report import (
    _acc,
    _agent_hit,
    _step_hit,
    build_table,
    completion_status,
)


def test_agent_hit_normalization():
    assert _agent_hit("WebSurfer", "websurfer")            # case-insensitive
    assert _agent_hit(" WebSurfer ", "WebSurfer")          # strip
    assert _agent_hit("MagenticOneOrchestrator", "orchestrator (magentic)")  # standardize_role both sides
    assert _agent_hit("WebSurfer_v2", "WebSurfer")         # gold substring of pred
    assert not _agent_hit("WebSurfer", "WebSurfer_v2")     # not the reverse
    assert not _agent_hit(None, "WebSurfer")
    assert not _agent_hit("WebSurfer", None)
    assert not _agent_hit("WebSurfer", "")


def test_step_hit_string_gold():
    assert _step_hit(12, "12")   # ww stores gold_step as a string
    assert _step_hit("4", 4)
    assert not _step_hit(3, "12")
    assert not _step_hit(None, "12")
    assert not _step_hit(4, None)
    assert not _step_hit(4, "not-a-number")


def test_acc_missing_counts_as_wrong():
    preds = {
        "1": {"predicted_agent": "A", "predicted_step": 0, "gold_agent": "A", "gold_step": 0},
        "2": {"predicted_agent": "B", "predicted_step": 5, "gold_agent": "A", "gold_step": 0},
    }
    n, agent_frac, step_frac = _acc(["1", "2", "3", "4"], preds)  # 3, 4 missing
    assert n == 4
    assert agent_frac == 0.25 and step_frac == 0.25


def _write_corpus_and_preds(tmp_path: Path, n: int, correct_ids: set[str]):
    data_dir = tmp_path / "data" / "toy"
    method_root = tmp_path / "outputs" / "toy" / "modelA" / "all_at_once"
    data_dir.mkdir(parents=True)
    method_root.mkdir(parents=True)
    for i in range(1, n + 1):
        (data_dir / f"{i}.json").write_text(json.dumps({
            "question_ID": f"q{i}", "history": [{"role": "A", "content": "x"}],
            "mistake_agent": "A", "mistake_step": 0,
        }))
        sid = str(i)
        doc = {
            "id": sid, "method": "all_at_once", "model": "modelA",
            "predicted_agent": "A" if sid in correct_ids else "B",
            "predicted_step": 0 if sid in correct_ids else 9,
            "gold_agent": "A", "gold_step": 0, "raw": "...", "calls": [],
        }
        (method_root / f"{i}.json").write_text(json.dumps(doc))
    (method_root / "_run.json").write_text("{}")  # must be ignored by readers
    return {
        "models": ["modelA"],
        "subsets": ["toy"],
        "methods": ["all_at_once"],
        "data_dir": str(tmp_path / "data"),
        "pred_root": str(tmp_path / "outputs"),
        "splits": {"train": 0.3, "val": 0.2, "test": 0.5},
        "seeds": [1, 2, 3],
    }


def test_build_table_full_columns_constant(tmp_path):
    cfg = _write_corpus_and_preds(tmp_path, n=20, correct_ids={str(i) for i in range(1, 11)})
    df = build_table("modelA", "toy", cfg)
    assert list(df["seed"]) == [1, 2, 3]
    # full-set accuracy is split-independent: constant across seed rows, = 10/20
    assert set(df["all_at_once_step@1_full"]) == {0.5}
    assert set(df["all_at_once_agent@1_full"]) == {0.5}
    assert (df["n_val"] + df["n_test"] <= 20).all()
    # val/test accuracies stay within [0,1] and vary with the seed's split
    for col in ("all_at_once_step@1_val", "all_at_once_step@1_test"):
        assert df[col].between(0, 1).all()


def test_completion_status_counts(tmp_path):
    cfg = _write_corpus_and_preds(tmp_path, n=5, correct_ids=set())
    st = completion_status(cfg)
    row = st.iloc[0]
    assert row["status"] == "DONE" and row["rows"] == 5 and row["expected"] == 5
    # remove one output → PARTIAL; _run.json must not count as a row
    method_root = Path(cfg["pred_root"]) / "toy" / "modelA" / "all_at_once"
    (method_root / "3.json").unlink()
    st = completion_status(cfg)
    assert st.iloc[0]["status"] == "PARTIAL(4/5)"
