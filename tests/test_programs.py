"""Program/driver behavior: call logs, batching shape, streaming equivalence."""
from __future__ import annotations

from baselines.prompting.backends.dummy import DummyBackend
from baselines.prompting.methods import METHODS
from baselines.prompting.runner import run_batched, run_streaming


def _records(lengths: list[int]) -> list[dict]:
    return [
        {
            "id": str(i + 1),
            "history": [
                {"role": f"agent{j}", "content": f"msg {j} of record {i}"} for j in range(n)
            ],
            "question": f"q{i}",
            "ground_truth": f"a{i}",
        }
        for i, n in enumerate(lengths)
    ]


class BatchRecorder:
    """Backend that scripts responses per request and records batch sizes."""

    prefers_streaming = False
    request_params: dict = {}

    def __init__(self, responder):
        self._responder = responder
        self.batch_sizes: list[int] = []

    def generate(self, message_lists):
        self.batch_sizes.append(len(message_lists))
        return [self._responder(m) for m in message_lists]


def _collect(records, method, backend, driver, **kw):
    preds = {}
    programs = [(r, METHODS[method](r, **kw)) for r in records]
    driver(programs, backend, lambda r, p: preds.__setitem__(r["id"], p))
    return preds


def test_all_at_once_single_mega_batch():
    records = _records([3, 5, 2])
    be = BatchRecorder(lambda m: "Agent Name: A\nStep Number: 1")
    preds = _collect(records, "all_at_once", be, run_batched)
    assert be.batch_sizes == [3]  # one round, one prompt per record
    assert all(p["predicted_step"] == 1 for p in preds.values())
    assert all(len(p["calls"]) == 1 for p in preds.values())


def test_step_by_step_batch_shape_and_calls():
    records = _records([3, 5, 2])
    be = BatchRecorder(lambda m: "1. No.")
    preds = _collect(records, "step_by_step", be, run_batched, step_mode="batch")
    assert be.batch_sizes == [10]  # legacy flattened cross-product in ONE batch
    assert [len(preds[r["id"]]["calls"]) for r in records] == [3, 5, 2]
    assert preds["1"]["calls"][0].keys() == {"step", "agent", "response"}
    assert all(p["predicted_step"] is None and p["raw"] is None for p in preds.values())


def test_binary_search_round_batches_shrink():
    # histories 4 and 2: round 1 has both active, round 2 only the length-4 one.
    records = _records([4, 2])
    responses = iter(["upper half", "lower half", "upper half"])
    be = BatchRecorder(lambda m: next(responses))
    preds = _collect(records, "binary_search", be, run_batched)
    assert be.batch_sizes == [2, 1]
    # record 2 (len 2): "lower half" → step 1; record 1: upper, upper → step 0
    assert preds["1"]["predicted_step"] == 0
    assert preds["2"]["predicted_step"] == 1
    assert [c.keys() for c in preds["1"]["calls"]] \
        == [{"round", "start", "end", "mid", "response"}] * 2


def test_streaming_matches_batched_predictions():
    records = _records([4, 3, 5])

    def responder(messages):
        # Deterministic function of the prompt so both drivers see identical
        # responses: flag "Yes" when the user prompt mentions step (2).
        user = messages[1]["content"]
        return "1. Yes." if "(2) was by" in user else "1. No."

    b_preds = _collect(records, "step_by_step", DummyBackend(responder=responder),
                       run_batched, step_mode="batch")
    s_preds = _collect(records, "step_by_step", DummyBackend(responder=responder),
                       run_streaming, step_mode="early_stop")
    for rid in b_preds:
        for key in ("predicted_agent", "predicted_step", "raw"):
            assert b_preds[rid][key] == s_preds[rid][key]
    # early_stop stops at step 2 → 3 calls; batch logs every step.
    assert all(len(p["calls"]) == 3 for p in s_preds.values())


def test_streaming_skips_failed_trajectory():
    records = _records([2, 2, 2])

    class Flaky(DummyBackend):
        def generate(self, message_lists):
            if any("record 1" in m[1]["content"] for m in message_lists):
                raise RuntimeError("boom")
            return super().generate(message_lists)

    preds = {}
    programs = [(r, METHODS["all_at_once"](r)) for r in records]
    run_streaming(programs, Flaky(), lambda r, p: preds.__setitem__(r["id"], p))
    assert set(preds) == {"1", "3"}  # record "2" failed and wrote nothing


def test_empty_history_binary_search_no_calls():
    record = {"id": "1", "history": [], "question": "q", "ground_truth": "a"}
    be = BatchRecorder(lambda m: "never called")
    preds = {}
    run_batched([(record, METHODS["binary_search"](record))], be,
                lambda r, p: preds.__setitem__(r["id"], p))
    assert be.batch_sizes == []  # program finished without yielding
    assert preds["1"]["predicted_step"] is None
    assert preds["1"]["calls"] == []
