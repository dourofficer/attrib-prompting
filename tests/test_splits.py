"""Split reproduction: unit behavior + frozen fixtures.

The fixtures in ``fixtures/split_ids.json`` were captured by running the
attribscope project's own ``split_data`` over this repo's corpus file lists
(splits {train:0.3, val:0.2, test:0.5}), so they pin the exact per-seed
partitions every reported number is computed on. This repo's copy must
reproduce them bit-exactly, from ``data/`` alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.shared.common import split_data
from baselines.prompting.report import universe_files, val_test_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = json.loads((REPO_ROOT / "tests/fixtures/split_ids.json").read_text())


def test_split_data_deterministic_partition():
    data = [str(i) for i in range(100)]
    a1, b1 = split_data(data, 0.3, seed=7)
    a2, b2 = split_data(data, 0.3, seed=7)
    assert (a1, b1) == (a2, b2)          # reproducible
    assert len(a1) == 30
    assert sorted(a1 + b1) == sorted(data)  # disjoint cover
    a3, _ = split_data(data, 0.3, seed=8)
    assert a3 != a1                       # seed actually matters
    assert data == [str(i) for i in range(100)]  # input not mutated


@pytest.mark.parametrize("key", sorted(FIXTURES))
def test_frozen_split_fixtures(key):
    ds, subset, seed_s = key.split("/")
    seed = int(seed_s.removeprefix("seed"))
    files = universe_files(REPO_ROOT / "data" / ds / subset)
    val_ids, test_ids = val_test_ids(files, train=0.3, val=0.2, seed=seed)
    assert val_ids == FIXTURES[key]["val"]
    assert test_ids == FIXTURES[key]["test"]


def test_universe_is_numerically_sorted():
    files = universe_files(REPO_ROOT / "data/ww/hand-crafted")
    stems = [int(Path(f).stem) for f in files]
    assert stems == sorted(stems)
    assert len(files) == 58
