"""``data/correct-error-gt`` invariants.

The corpus is ``data/correct-error`` with the task answer restored (built by
``scripts/build_correct_error_gt.py``; see its docstring for the join). Its whole
value rests on two properties, both pinned here:

1. It differs from ``data/correct-error`` in the ground-truth keys and nothing
   else — same trajectories, same gold labels, same (deliberately still-damaged)
   question text — so an accuracy delta is attributable to the answer alone.
2. Its filename stems are identical, so ``split_data`` produces the *same*
   per-seed val/test partitions and the two corpora compare pairwise.

CPU-only, keyless, no network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.prompting.predict import load_records
from baselines.prompting.report import universe_files, val_test_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "data/correct-error"
GT = REPO_ROOT / "data/correct-error-gt"

SUBSETS = ["arc", "gaia", "hotpot", "math500", "mmlu_pro", "musique", "wikimqa"]
EXPECTED_N = {"arc": 304, "gaia": 50, "hotpot": 578, "math500": 157,
              "mmlu_pro": 92, "musique": 312, "wikimqa": 733}
SEEDS = [1, 2, 3]

# The only keys the builder is allowed to touch.
GT_KEYS = {"ground_truth", "groundtruth", "question_source", "gt_source"}

pytestmark = pytest.mark.skipif(not GT.is_dir(), reason="data/correct-error-gt not built")


@pytest.mark.parametrize("subset", SUBSETS)
def test_same_files_as_base_corpus(subset):
    base = sorted(p.name for p in (BASE / subset).glob("*.json"))
    gt = sorted(p.name for p in (GT / subset).glob("*.json"))
    assert gt == base
    assert len(gt) == EXPECTED_N[subset]


@pytest.mark.parametrize("subset", SUBSETS)
def test_only_ground_truth_keys_differ(subset):
    """Trajectories, questions and gold labels are copied verbatim."""
    for name in sorted(p.name for p in (BASE / subset).glob("*.json")):
        old = json.loads((BASE / subset / name).read_text(encoding="utf-8"))
        new = json.loads((GT / subset / name).read_text(encoding="utf-8"))
        changed = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
        assert changed <= GT_KEYS, f"{subset}/{name} also changed {sorted(changed - GT_KEYS)}"
        assert set(new) - set(old) == GT_KEYS - set(old)
        # spelled out, because these are what every downstream prompt reads
        assert new["history"] == old["history"]
        assert new["question"] == old["question"]
        assert new["mistake_agent"] == old["mistake_agent"]
        assert new["mistake_step"] == old["mistake_step"]


@pytest.mark.parametrize("subset", SUBSETS)
def test_every_record_has_an_answer(subset):
    """Both spellings are populated: this repo reads ``ground_truth``, the
    vendored CORRECT schema generator reads ``groundtruth``."""
    for record in load_records(str(GT / subset)):
        assert record["ground_truth"].strip(), record["filename"]
    for path in (GT / subset).glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["groundtruth"] == doc["ground_truth"]
        assert doc["gt_source"]["row_index"] == int(doc["question_id"].split("_")[0][4:])


@pytest.mark.parametrize("subset", SUBSETS)
def test_base_corpus_still_has_no_answer(subset):
    """The with-GT corpus is additive: the original stays the without-GT control."""
    for record in load_records(str(BASE / subset)):
        assert record["ground_truth"] == ""


@pytest.mark.parametrize("subset", SUBSETS)
@pytest.mark.parametrize("seed", SEEDS)
def test_splits_identical_to_base_corpus(subset, seed):
    """What makes with-GT vs without-GT a paired comparison."""
    base_files = universe_files(BASE / subset)
    gt_files = universe_files(GT / subset)
    assert gt_files == base_files
    assert val_test_ids(gt_files, train=0.3, val=0.2, seed=seed) == \
        val_test_ids(base_files, train=0.3, val=0.2, seed=seed)


def test_provenance_manifest_is_outside_the_split_universe():
    """``_provenance.json`` sits at the corpus root, so no subset scan sees it."""
    assert (GT / "_provenance.json").is_file()
    manifest = json.loads((GT / "_provenance.json").read_text(encoding="utf-8"))
    assert {s["subset"] for s in manifest["subsets"]} == set(SUBSETS)
    for subset in SUBSETS:
        assert "_provenance.json" not in universe_files(GT / subset)
