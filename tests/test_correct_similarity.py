"""Parity of the CORRECT similarity stage with the vendored implementation."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.correct.retrieval import load_trajectory_similarities
from baselines.correct.similarity import (
    compute_trajectory_similarities,
    read_trajectory_json,
)

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_SIM = REPO_ROOT / "vendored/CORRECT/src/generate_trajectory_similarities.py"


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
    _stub("transformers", AutoTokenizer=None, AutoModel=None)
    _stub("sklearn")
    _stub("sklearn.metrics")
    _stub("sklearn.metrics.pairwise", cosine_similarity=None)
    _stub("tqdm", tqdm=lambda it, **kw: it)
    spec = importlib.util.spec_from_file_location("vendored_similarities", VENDORED_SIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def data_dir(tmp_path):
    docs = {
        # 2.json and 10.json are byte-identical → an exact embedding tie.
        "1": {"question": "What is 2+2?",
              "history": [{"role": "MathAgent", "content": "It is 4."}]},
        "2": {"question": "Where is Paris?",
              "history": [{"role": "GeoAgent", "content": "In France."}]},
        "10": {"question": "Where is Paris?",
               "history": [{"role": "GeoAgent", "content": "In France."}]},
        "3": {"question": "Long trip planning",
              "history": [{"name": "planner", "content": "Step one."},
                          {"role": "Worker", "content": ""}]},
    }
    for stem, doc in docs.items():
        (tmp_path / f"{stem}.json").write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "file_mapping.json").write_text("{}")  # must be excluded
    (tmp_path / "notes.txt").write_text("ignored")
    return tmp_path


def test_read_trajectory_json_parity(vendored, data_dir, tmp_path):
    for f in sorted(data_dir.glob("*.json")):
        assert read_trajectory_json(f) == vendored.read_trajectory_json(str(f))
    # question-less file: no "Question:" line at all (vendored branch).
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"history": [{"role": "A", "content": "hi"}]}))
    assert read_trajectory_json(q) == vendored.read_trajectory_json(str(q)) == "A: hi"
    # unreadable → ""
    assert read_trajectory_json(tmp_path / "missing.json") == ""


def _raw_embed(texts: list[str]):
    rows = [[len(t), sum(t.encode()) % 997 + 1, t.count("a") * 3 + 1, ord(t[0])]
            for t in texts]
    return np.asarray(rows, dtype=np.float32)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None
                    or importlib.util.find_spec("sklearn") is None,
                    reason="ranking parity drives the vendored torch/sklearn pipeline")
def test_ranking_parity_with_vendored(vendored, data_dir, monkeypatch):
    """Same deterministic embeddings through both pipelines → identical maps,
    including exact-tie order (stable sort over lexicographic listdir order)."""
    import torch

    stash: list[list[str]] = []

    class FakeTokenizer:
        def __call__(self, batch_texts, padding, truncation, max_length, return_tensors):
            assert max_length == 8192
            stash.append(list(batch_texts))
            n = len(batch_texts)
            return {"input_ids": torch.zeros(n, 2), "attention_mask": torch.ones(n, 2)}

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, **kw):
            return None

    monkeypatch.setattr(vendored, "AutoTokenizer",
                        SimpleNamespace(from_pretrained=lambda name: FakeTokenizer()))
    monkeypatch.setattr(vendored, "AutoModel",
                        SimpleNamespace(from_pretrained=lambda name: FakeModel()))
    monkeypatch.setattr(vendored, "mean_pooling",
                        lambda model_output, attention_mask: torch.tensor(_raw_embed(stash.pop(0))))

    theirs = vendored.compute_trajectory_similarities(str(data_dir), "fake-model")

    def fake_embed(texts, model_name, batch_size=8, max_length=8192):
        return torch.nn.functional.normalize(
            torch.tensor(_raw_embed(texts)), p=2, dim=1).cpu().numpy()

    ours = compute_trajectory_similarities(str(data_dir), "fake-model", embed_fn=fake_embed)

    assert ours == theirs
    assert set(ours) == {1, 2, 3, 10}
    for key, ranked in ours.items():
        assert key not in ranked and len(ranked) == 3  # self excluded
    # Exact tie between 2 and 10 as neighbours of 1 and 3: stable sort keeps
    # the lexicographic listdir order (10.json before 2.json).
    i2, i10 = ours[1].index(2), ours[1].index(10)
    assert i10 == i2 - 1


def test_dump_load_round_trip(tmp_path):
    mapping = {1: [10, 2, 3], 2: [10, 1, 3], 10: [2, 1, 3], 3: [1, 10, 2]}
    out = tmp_path / "bge-m3.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    # int keys serialize as strings (vendored dump format) ...
    assert set(json.loads(out.read_text())) == {"1", "2", "10", "3"}
    # ... and load back int-keyed (vendored loader semantics).
    assert load_trajectory_similarities(out) == mapping
