"""CPU-only, keyless tests for the StepFinder baseline (baselines-rp/stepfinder/).

The suite has one job above all others: prove that what this repo runs is what
``vendored/StepFinder/`` runs. Where a vendored module can be imported and
driven, the test drives it and compares bit-for-bit; where the vendored code
buries the logic inside a script body, the test reproduces that body and pins
the result.

Importing the vendored code is awkward on purpose. ``vendored/StepFinder/`` is
a flat script directory whose modules import each other by bare name (``import
model``), and ``model`` is generic enough to shadow almost anything.
``vendored_stepfinder`` below therefore puts the directory on the path, imports
what it was asked for, and puts everything back.
"""
from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_SF = REPO_ROOT / "vendored" / "StepFinder"


@contextlib.contextmanager
def vendored_stepfinder(*module_names):
    """Import modules from ``vendored/StepFinder/`` by bare name, then undo it.

    Same contract as ``vendored_oat`` in ``test_oat_pipeline.py``: on exit the
    path entry goes away and every module that came *out of that directory* is
    dropped, while third-party packages pulled in on the way (torch,
    transformers) stay, because tearing them down mid-import leaves them
    unusable for the rest of the session.
    """
    before = dict(sys.modules)
    vendored_dir = str(VENDORED_SF)
    sys.path.insert(0, vendored_dir)
    try:
        yield tuple(importlib.import_module(name) for name in module_names)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(vendored_dir)
        for name, module in list(sys.modules.items()):
            if name in before:
                continue
            origin = getattr(module, "__file__", None) or ""
            if origin.startswith(vendored_dir):
                del sys.modules[name]
        sys.modules.update(before)


def _random_batch(torch, batch_lengths, content_dim=128, agent_dim=32, seed=0):
    """A padded batch with the vendored collate's exact shapes and dtypes."""
    gen = torch.Generator().manual_seed(seed)
    T = max(batch_lengths)
    B = len(batch_lengths)
    content = torch.zeros(B, T, content_dim)
    agent = torch.zeros(B, T, agent_dim)
    mask = torch.zeros(B, T)  # float32, as the vendored collate produces
    for i, L in enumerate(batch_lengths):
        content[i, :L] = torch.randn(L, content_dim, generator=gen)
        agent[i, :L] = torch.randn(L, agent_dim, generator=gen)
        mask[i, :L] = 1.0
    return content, agent, mask


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------

def test_model_forward_matches_vendored():
    """Same weights, same input, same numbers — on several batch shapes.

    This is the test the whole port rests on. ``model.py`` is a transcription,
    so the only honest check is to load one state dict into both copies and
    demand bit equality, not closeness.
    """
    torch = pytest.importorskip("torch")
    from stepfinder.model import StepFinder as Ours

    with vendored_stepfinder("model") as (vendored,):
        for shapes in ([5], [1], [12, 5, 3], [130, 2]):
            ours = Ours()
            theirs = vendored.StepFinder()
            theirs.load_state_dict(ours.state_dict())
            ours.eval(); theirs.eval()

            content, agent, mask = _random_batch(torch, shapes, seed=len(shapes))
            with torch.no_grad():
                our_logits, our_tl = ours(content, agent, mask)
                their_logits, their_tl = theirs(content, agent, mask)

            assert torch.equal(our_logits, their_logits), f"logits differ for {shapes}"
            assert torch.equal(our_tl, their_tl), f"temporal loss differs for {shapes}"


def test_loss_matches_vendored():
    torch = pytest.importorskip("torch")
    from stepfinder.model import StepFinder as Ours, compute_loss as our_loss

    with vendored_stepfinder("model") as (vendored,):
        ours = Ours()
        theirs = vendored.StepFinder()
        theirs.load_state_dict(ours.state_dict())
        ours.eval(); theirs.eval()

        content, agent, mask = _random_batch(torch, [7, 4], seed=3)
        targets = torch.zeros(2, mask.shape[1], dtype=torch.long)
        targets[0, 5] = 1
        targets[1, 2] = 1

        with torch.no_grad():
            lo, tl = ours(content, agent, mask)
            assert torch.equal(
                our_loss(lo, targets, mask, tl, lambda_temporal=0.9),
                vendored.compute_loss(lo, targets, mask, tl, lambda_temporal=0.9),
            )


@pytest.mark.parametrize("scales", [[1], [1, 2], [2], [1, 2, 3]])
def test_multi_scale_differencing_matches_vendored(scales):
    """Including s=3, where the vendored operator is deliberately nonstandard.

    ``H_t - s*H_{t-1} + (s-1)*H_{t-s}`` is a true second difference only at
    s=2; at s=3 it is ``H_t - 3*H_{t-2} + 2*H_{t-3}``. The paper's grid only
    ever uses {1, 2}, but transcribing the formula rather than the intent is
    the point, so the odd case is pinned too.
    """
    torch = pytest.importorskip("torch")
    from stepfinder.model import StepLevelErrorScoring as Ours

    with vendored_stepfinder("model") as (vendored,):
        ours = Ours(hidden_dim=128, scales=scales)
        theirs = vendored.StepLevelErrorScoring(hidden_dim=128, scales=scales)
        theirs.load_state_dict(ours.state_dict())

        gen = torch.Generator().manual_seed(11)
        H = torch.randn(3, 9, 128, generator=gen)
        mask = torch.zeros(3, 9)
        for i, L in enumerate([9, 6, 4]):
            mask[i, :L] = 1.0
            H[i, L:] = 0.0

        assert torch.equal(
            ours._compute_multi_scale_diff(H, mask),
            theirs._compute_multi_scale_diff(H, mask),
        )


def test_position_bias_at_batch_one_is_the_papers_equation():
    """Scoring one trajectory at a time turns the vendored line into Eq. 9.

    ``model.py:242-243`` divides by the batch's padded width. At batch size 1
    that width *is* the trajectory's length, so no code change is needed to get
    the paper's ``b_t = -(t-1)/(T-1)``. This test states that in the only way
    that matters: the term the model actually adds.
    """
    torch = pytest.importorskip("torch")
    from stepfinder.model import StepLevelErrorScoring

    T, gamma = 11, 0.4
    scoring = StepLevelErrorScoring(hidden_dim=128, gamma=gamma)
    with torch.no_grad():
        for p in scoring.mlp.parameters():
            p.zero_()
    scoring.beta = 0.0  # isolate the position term

    H = torch.zeros(1, T, 128)
    mask = torch.ones(1, T)
    with torch.no_grad():
        logits, _ = scoring(H, mask)

    t = torch.arange(T).float()
    expected = gamma * -(t / (T - 1 + 1e-6))
    assert torch.allclose(logits[0], expected, atol=1e-7)


def test_batched_position_bias_differs_from_single_by_a_closed_form():
    """Every term except the position bias is batch-invariant.

    The packed LSTM, the masked attention, the masked-mean gate and the
    mean-normalized differencing all ignore padding, so batching moves the
    logits by exactly ``gamma * (t/(L-1) - t/(T_max-1))`` and nothing else.
    Knowing that is what lets ``predict.py`` train at 16 and score at 1 without
    hand-waving about "small" differences.
    """
    torch = pytest.importorskip("torch")
    from stepfinder.model import StepFinder

    gamma, lengths = 0.4, [12, 5, 3]
    net = StepFinder(gamma=gamma)
    net.eval()
    content, agent, mask = _random_batch(torch, lengths, seed=7)
    T_max = max(lengths)

    with torch.no_grad():
        batched, _ = net(content, agent, mask)
        for i, L in enumerate(lengths):
            single, _ = net(content[i : i + 1, :L], agent[i : i + 1, :L], mask[i : i + 1, :L])
            t = torch.arange(L).float()
            predicted_shift = gamma * (t / (L - 1 + 1e-6) - t / (T_max - 1 + 1e-6))
            observed_shift = batched[i, :L] - single[0]
            assert torch.allclose(observed_shift, predicted_shift, atol=1e-6), (
                f"row {i} (L={L}) moved by something other than the position bias"
            )


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

def test_collate_matches_vendored():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from stepfinder.collate import sequence_collate_fn as ours

    batch = [
        {"content_features": np.random.RandomState(i).randn(L, 128).astype("float32"),
         "agent_features": np.random.RandomState(i + 50).randn(L, 32).astype("float32"),
         "mistake_labels": np.eye(L, dtype="int64")[min(i, L - 1)],
         "id": str(i)}  # an extra key the vendored collate must ignore
        for i, L in enumerate([6, 3, 9])
    ]
    with vendored_stepfinder("collate_fn") as (vendored,):
        theirs = vendored.sequence_collate_fn(batch)
    mine = ours(batch)

    assert set(mine) == set(theirs) == {"content_seq", "agent_seq", "mistake_labels", "mask"}
    for key in theirs:
        assert torch.equal(mine[key], theirs[key]), key
    assert mine["mask"].dtype is torch.float32


# --------------------------------------------------------------------------
# Encoding and features
# --------------------------------------------------------------------------

class _FakeQ3Emb:
    """Stands in for ``Qwen3Embedding`` so the vendored featurizer runs on CPU.

    Returns a deterministic vector per string, wide enough that slicing to 128
    or 32 is a real slice. Both sides of the parity test see the same object,
    so any difference is in the code around it, which is the point.
    """

    WIDTH = 160

    def __init__(self):
        import torch
        self.torch = torch

    def encode(self, sentences, dim=-1):
        import hashlib
        import torch
        if isinstance(sentences, str):
            sentences = [sentences]
        rows = []
        for text in sentences:
            seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
            gen = torch.Generator().manual_seed(seed)
            rows.append(torch.randn(self.WIDTH, generator=gen))
        out = torch.stack(rows)
        return out[:, :dim] if dim > 0 else out


def _vendored_featurizer(vendored_module, fake):
    """A vendored ``TemporalFeatureExtractor`` without its checkpoint load."""
    obj = vendored_module.TemporalFeatureExtractor.__new__(
        vendored_module.TemporalFeatureExtractor
    )
    obj.content_dim, obj.agent_dim, obj.model = 128, 32, fake
    return obj


def test_encoder_matches_vendored_encode_body():
    """Our ``Encoder.embed`` reproduces ``_encode`` including its edge cases."""
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from stepfinder.encode import Encoder

    fake = _FakeQ3Emb()
    ours = Encoder(name="fake", model=None, tokenizer=None)
    ours._forward = lambda text, dim: (
        fake.encode(text, dim).squeeze(0).cpu().numpy().astype(np.float32)
    )

    with vendored_stepfinder("feature_construction") as (vendored,):
        theirs = _vendored_featurizer(vendored, fake)
        for text, dim in [("a normal step", 128), ("WebSurfer", 32),
                          ("", 32), ("   ", 128), ("x" * 5000, 128)]:
            assert np.array_equal(ours.embed(text, dim), theirs._encode(text, dim)), text


def test_features_match_vendored_on_a_vendored_training_record():
    """On the corpus the vendored rule was written for, we must agree exactly."""
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    import json

    from stepfinder.encode import Encoder
    from stepfinder.features import AGENT_FIELD_VENDORED, build_features

    path = REPO_ROOT / "vendored/StepFinder/data/Hand-Crafted/train/trajectory_0000.json"
    log = json.loads(path.read_text())

    fake = _FakeQ3Emb()
    enc = Encoder(name="fake", model=None, tokenizer=None)
    enc._forward = lambda text, dim: (
        fake.encode(text, dim).squeeze(0).cpu().numpy().astype(np.float32)
    )

    record = {"id": "0", "history": log["history"], "question": log["question"],
              "ground_truth": log["ground_truth"], "gold_step": log["mistake_step"]}
    ours = build_features(record, enc, agent_field=AGENT_FIELD_VENDORED,
                          agent_normalize="raw", include_gt=False)

    with vendored_stepfinder("feature_construction") as (vendored,):
        theirs = _vendored_featurizer(vendored, fake).process_log(log)

    for key in ("content_features", "agent_features", "mistake_labels"):
        assert np.array_equal(ours[key], theirs[key]), key
    assert ours["trainable"] is True


def test_agent_identity_reads_role_not_name_on_repo_corpora():
    """The trap: ``data/ww/algorithm-generated`` swaps ``role`` and ``name``.

    Its steps carry ``name`` = ``assistant``/``user`` (the chat role) and
    ``role`` = the agent. The vendored ``name or role`` rule would embed
    "assistant" as the agent for every model turn. This test is the gate that
    keeps the port from regressing to it.
    """
    import json

    from stepfinder.features import (
        AGENT_FIELD_REPO, AGENT_FIELD_VENDORED, agent_text,
    )

    record = json.loads((REPO_ROOT / "data/ww/algorithm-generated/1.json").read_text())
    step = record["history"][0]
    assert step["name"] == "assistant" and step["role"] == "Excel_Expert", (
        "fixture drifted: this test exists because the corpus swaps the fields"
    )
    assert agent_text(step, AGENT_FIELD_REPO, "raw") == "Excel_Expert"

    # ...while the vendored training corpus really does keep the agent in `name`
    vend = json.loads(
        (REPO_ROOT / "vendored/StepFinder/data/Hand-Crafted/train/trajectory_0000.json").read_text()
    )
    vstep = next(s for s in vend["history"] if s["name"] != "user")
    assert agent_text(vstep, AGENT_FIELD_VENDORED, "raw") == vstep["name"]
    assert vstep["role"] in {"assistant", "user", "system"}


def test_agent_normalization_puts_ww_onto_the_training_vocabulary():
    """``Orchestrator (thought)`` and friends collapse onto ``Orchestrator``."""
    from stepfinder.features import agent_text

    for raw in ["Orchestrator (thought)", "Orchestrator (-> WebSurfer)", "MagenticOneOrchestrator"]:
        assert agent_text({"role": raw}, "role", "standardize") == "Orchestrator"
        assert agent_text({"role": raw}, "role", "raw") == raw
    # a non-orchestrator agent is left alone by both settings
    assert agent_text({"role": "WebSurfer"}, "role", "standardize") == "WebSurfer"


def test_label_vector_rejects_an_unusable_gold_step():
    """An all-zero one-hot would train the model that step 0 was to blame."""
    np = pytest.importorskip("numpy")
    from stepfinder.features import label_vector

    assert np.array_equal(label_vector(2, 5), np.array([0, 0, 1, 0, 0]))
    assert np.array_equal(label_vector("2", 5), np.array([0, 0, 1, 0, 0]))  # ww stores strings
    assert label_vector(9, 5) is None      # past the end — traceelephant/magentic
    assert label_vector(-1, 5) is None
    assert label_vector(None, 5) is None
    assert label_vector("n/a", 5) is None


def test_gt_line_touches_only_the_first_step():
    """What makes a with-GT run reuse the without-GT cache."""
    np = pytest.importorskip("numpy")
    from stepfinder.encode import dummy_encoder
    from stepfinder.features import build_features, derive_gt_features

    record = {
        "id": "1", "gold_step": 1, "ground_truth": "42",
        "history": [{"role": "user", "content": "what is it?"},
                    {"role": "Solver", "content": "seven"},
                    {"role": "Solver", "content": "final: seven"}],
    }
    enc = dummy_encoder()
    without = build_features(record, enc)
    with_gt = build_features(record, enc, include_gt=True)
    derived = derive_gt_features(without, record, enc)

    assert not np.array_equal(without["content_features"][0], with_gt["content_features"][0])
    assert np.array_equal(without["content_features"][1:], with_gt["content_features"][1:])
    assert np.array_equal(without["agent_features"], with_gt["agent_features"])
    # deriving is the same answer as re-encoding from scratch
    assert np.array_equal(derived["content_features"], with_gt["content_features"])
    assert derived["gt_in_prompt"] is True


def test_features_feed_the_vendored_collate_untouched():
    """Extra provenance keys must not disturb the vendored batching."""
    pytest.importorskip("torch")
    from stepfinder.encode import dummy_encoder
    from stepfinder.features import build_features

    enc = dummy_encoder()
    batch = [
        build_features(
            {"id": str(i), "gold_step": 0, "ground_truth": "",
             "history": [{"role": "A", "content": f"s{j}"} for j in range(L)]},
            enc,
        )
        for i, L in enumerate([4, 2])
    ]
    with vendored_stepfinder("collate_fn") as (vendored,):
        out = vendored.sequence_collate_fn(batch)
    assert out["content_seq"].shape == (2, 4, 128)
    assert out["agent_seq"].shape == (2, 4, 32)
    assert out["mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def _toy_features(n=6, lengths=(5, 7, 4, 6, 8, 5), seed=0):
    """Deterministic feature payloads, shaped like the real thing."""
    import numpy as np
    rs = np.random.RandomState(seed)
    out = []
    for i in range(n):
        L = lengths[i % len(lengths)]
        labels = np.zeros(L, dtype=np.int64)
        labels[i % L] = 1
        out.append({
            "content_features": rs.randn(L, 128).astype("float32"),
            "agent_features": rs.randn(L, 32).astype("float32"),
            "mistake_labels": labels,
            "id": str(i),
            "num_steps": L,
            "gold_step": i % L,
            "question": f"q{i % 3}",   # three tasks across six trajectories
            "trainable": True,
        })
    return out


def test_evaluate_matches_vendored():
    """The metric block is a transcription; drive both over the same batches."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from stepfinder.collate import SequenceDataset, sequence_collate_fn
    from stepfinder.model import StepFinder
    from stepfinder.train import evaluate as ours

    feats = _toy_features()
    net = StepFinder()
    dev = torch.device("cpu")

    def loader():
        return DataLoader(SequenceDataset(feats), batch_size=4, shuffle=False,
                          collate_fn=sequence_collate_fn)

    with vendored_stepfinder("main") as (vendored,):
        theirs = vendored.evaluate(net, loader(), dev)
    mine = ours(net, loader(), dev)

    assert mine["acc"] == theirs["acc"]
    assert mine["top2"] == theirs["top2"] and mine["top3"] == theirs["top3"]
    assert mine["mrr"] == theirs["mrr"]
    assert mine["tolerance_acc"] == theirs["tolerance_acc"]


def test_train_one_epoch_matches_the_vendored_loop():
    """Same seed, same batches, same optimizer — the weights must agree.

    The vendored training loop lives inside ``run_experiment``'s body, so the
    comparison reproduces that body here rather than importing it. Everything
    it touches — AdamW, the clip at 1.0, the loss, the shuffle order under a
    fixed seed — is the thing under test.
    """
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from stepfinder.collate import SequenceDataset, sequence_collate_fn
    from stepfinder.model import StepFinder
    from stepfinder.train import set_seed, train_one_epoch

    feats = _toy_features()
    dev = torch.device("cpu")

    def fresh():
        set_seed(42)
        net = StepFinder()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        loader = DataLoader(SequenceDataset(feats), batch_size=4, shuffle=True,
                            collate_fn=sequence_collate_fn)
        return net, opt, loader

    # ours
    net_a, opt_a, loader_a = fresh()
    loss_a = train_one_epoch(net_a, loader_a, opt_a, 0.9, dev)

    # the vendored body, transcribed inline from main.py:202-227
    with vendored_stepfinder("main", "model") as (_vmain, vmodel):
        net_b, opt_b, loader_b = fresh()
        net_b.train()
        epoch_loss = 0.0
        for batch in loader_b:
            opt_b.zero_grad()
            mask = batch["mask"].to(dev)
            content = batch["content_seq"].to(dev)
            agent = batch["agent_seq"].to(dev)
            targets = batch["mistake_labels"].to(dev)
            logits, temporal_loss = net_b(content, agent, mask)
            loss = vmodel.compute_loss(logits, targets, mask, temporal_loss,
                                       lambda_temporal=0.9)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_b.parameters(), max_norm=1.0)
            opt_b.step()
            epoch_loss += loss.item()
        loss_b = epoch_loss / len(loader_b)

    assert loss_a == pytest.approx(loss_b, rel=0, abs=0)
    for (k, va), (_, vb) in zip(net_a.state_dict().items(), net_b.state_dict().items()):
        assert torch.equal(va, vb), f"weights diverged at {k}"


def test_holdout_never_splits_a_question():
    from stepfinder.train import holdout_by_question

    feats = _toy_features(n=6)  # three questions, two trajectories each
    train, val = holdout_by_question(feats, frac=0.34, seed=1)
    train_q = {x["question"] for x in train}
    val_q = {x["question"] for x in val}
    assert train_q and val_q
    assert not (train_q & val_q), "a question appeared on both sides"
    assert len(train) + len(val) == len(feats)


def test_holdout_is_deterministic_under_a_seed():
    from stepfinder.train import holdout_by_question

    feats = _toy_features(n=6)
    a = holdout_by_question(feats, 0.34, 7)
    b = holdout_by_question(feats, 0.34, 7)
    assert [x["id"] for x in a[1]] == [x["id"] for x in b[1]]


def test_presets_carry_the_papers_table_1():
    """The code ships only the Alg column; the HC column exists in the paper."""
    from stepfinder.train import PRESETS

    alg, hc = PRESETS["alg"], PRESETS["hc"]
    assert (alg.lr, alg.alpha, alg.beta, alg.gamma, alg.lambda_temporal) == (1e-3, 0.1, 0.9, 0.40, 0.90)
    assert (hc.lr, hc.alpha, hc.beta, hc.gamma, hc.lambda_temporal) == (1e-5, 0.3, 0.1, 0.75, 0.02)
    assert alg.scales == hc.scales == (1, 2)
    assert alg.weight_decay == hc.weight_decay == 1e-5
    assert alg.fingerprint() != hc.fingerprint()


def test_preset_defaults_equal_the_vendored_cli_defaults():
    """``main.py``'s argparse defaults are exactly the Alg preset."""
    from unittest import mock

    from stepfinder.train import BATCH_SIZE, EPOCHS, PATIENCE, PRESETS

    with vendored_stepfinder("main") as (vendored,):
        with mock.patch.object(sys, "argv", ["main.py"]):
            args = vendored.parse_args()

    alg = PRESETS["alg"]
    assert (args.lr, args.alpha, args.beta, args.gamma) == (alg.lr, alg.alpha, alg.beta, alg.gamma)
    assert args.lambda_temporal == alg.lambda_temporal
    assert tuple(args.scales) == alg.scales
    assert (args.epochs, args.batch_size, args.patience) == (EPOCHS, BATCH_SIZE, PATIENCE)
    assert (args.content_dim, args.agent_dim, args.hidden_dim) == (alg.content_dim, alg.agent_dim, alg.hidden_dim)


def test_fit_without_a_validation_set_keeps_the_last_epoch():
    """Family B's smallest partitions cannot spare a holdout; say so in the result."""
    pytest.importorskip("torch")
    from stepfinder.train import PRESETS, fit

    res = fit(_toy_features(), [], PRESETS["alg"], seed=42, epochs=2, patience=1, log_every=0)
    assert res.model_selection == "val(fallback:last)"
    assert res.epochs_run == 2 and res.best_epoch == 2
    assert res.n_val == 0


def test_fit_selects_the_best_validation_epoch():
    pytest.importorskip("torch")
    from stepfinder.train import PRESETS, fit, holdout_by_question

    feats = _toy_features()
    train, val = holdout_by_question(feats, 0.34, seed=1)
    res = fit(train, val, PRESETS["alg"], seed=42, epochs=3, patience=3, log_every=0)
    assert res.val_n_tasks >= 1
    assert 1 <= res.best_epoch <= 3
    accs = [h["val_acc"] for h in res.history]
    assert res.best_acc == max(accs)
    # The vendored rule needs a *strict* improvement over a 0.0 start, so a
    # run that never scores keeps the last epoch and says so.
    assert res.model_selection == ("val" if max(accs) > 0 else "val(fallback:last)")


def test_score_trajectory_is_argmax_over_a_single_trajectory():
    pytest.importorskip("torch")
    torch = pytest.importorskip("torch")
    from stepfinder.model import StepFinder
    from stepfinder.train import score_trajectory

    payload = _toy_features(n=1)[0]
    out = score_trajectory(StepFinder(), payload, torch.device("cpu"))
    assert len(out["scores"]) == payload["num_steps"] == len(out["scores_logit"])
    assert out["step_indices"] == list(range(payload["num_steps"]))
    assert out["predicted_step"] == max(range(len(out["scores"])), key=lambda i: out["scores"][i])
    assert abs(sum(out["scores"]) - 1.0) < 1e-5, "probabilities over the valid steps"


# --------------------------------------------------------------------------
# The protocol layer
# --------------------------------------------------------------------------

def test_in_corpus_partition_matches_the_reports_own_split():
    """The 30% family B trains on is exactly the slice the report throws away."""
    from baselines.prompting.report import universe_files, val_test_ids
    from stepfinder.protocol import scored_partition_ids, train_partition_ids

    splits = {"train": 0.3, "val": 0.2, "test": 0.5}
    data_dir = REPO_ROOT / "data/ww/hand-crafted"
    files = universe_files(data_dir)

    for seed in (1, 7, 20):
        train = set(train_partition_ids(data_dir, splits, seed))
        scored = scored_partition_ids(data_dir, splits, seed)
        val, test = val_test_ids(files, 0.3, 0.2, seed)

        assert scored == set(val) | set(test)
        assert not (train & scored), "the model would be scored on what it trained on"
        assert train | scored == {f.rsplit(".", 1)[0] for f in files}


def _regen_set(train_set):
    from stepfinder.features import AGENT_FIELD_VENDORED
    from stepfinder.protocol import (
        DEFAULT_TRAIN_DATA_ROOT, VENDORED_DIRS, TrainingSet, load_vendored_records,
    )
    from stepfinder.train import PRESETS

    rel = VENDORED_DIRS[train_set]
    return TrainingSet(
        name=train_set, protocol="regen", source=f"{DEFAULT_TRAIN_DATA_ROOT}/{rel}",
        records=load_vendored_records(REPO_ROOT / DEFAULT_TRAIN_DATA_ROOT / rel),
        agent_field=AGENT_FIELD_VENDORED, preset="hc", hparams=PRESETS["hc"],
    )


def test_train_task_overlap_reproduces_the_measured_leakage():
    """Under the default mapping: ww is clean, GAIA-derived subsets are not.

    ``traceelephant`` and ``correct-error/gaia`` reuse GAIA questions that the
    Hand-Crafted training set also covers, so a vendored-trained model has seen
    a labelled failure on the same task — a different agent system, a different
    trajectory, a different decisive step, but the same question and answer.
    The flag records it per file rather than dropping the subsets.
    """
    from baselines.prompting.predict import load_records
    from stepfinder.protocol import resolve_train_set, train_task_overlap

    counts = {
        "data/ww/hand-crafted": 0,
        "data/ww/algorithm-generated": 0,
        "data/traceelephant/captain": 22,
        "data/traceelephant/magentic": 51,
        "data/traceelephant/swe": 0,
        "data/correct-error/gaia": 30,
        "data/tracertraj/code": 0,
    }
    sets = {name: _regen_set(name) for name in ("hand-crafted", "algorithm-generated")}
    for rel, expected in counts.items():
        subset = Path(rel).name
        ts = sets[resolve_train_set(subset)]
        records = load_records(str(REPO_ROOT / rel))
        got = sum(1 for r in records if train_task_overlap(r, ts))
        assert got == expected, f"{rel}: expected {expected} overlapping tasks, got {got}"


def test_overlap_is_computed_against_the_training_set_actually_used():
    """Swapping the subset mapping would create leakage where there is none.

    ``ww`` is clean only because each subset is paired with the training corpus
    built from the same agent system. Pair ``ww/algorithm-generated`` with the
    Hand-Crafted set instead and 69 of its 126 tasks are suddenly in training —
    which is why the flag is never a constant.
    """
    from baselines.prompting.predict import load_records
    from stepfinder.protocol import train_task_overlap

    records = load_records(str(REPO_ROOT / "data/ww/algorithm-generated"))
    paired = _regen_set("algorithm-generated")
    swapped = _regen_set("hand-crafted")
    assert sum(1 for r in records if train_task_overlap(r, paired)) == 0
    assert sum(1 for r in records if train_task_overlap(r, swapped)) == 69

    hc_records = load_records(str(REPO_ROOT / "data/ww/hand-crafted"))
    assert sum(1 for r in hc_records if train_task_overlap(r, _regen_set("hand-crafted"))) == 0
    assert sum(1 for r in hc_records if train_task_overlap(r, paired)) == 11


def test_the_vendored_training_corpus_is_disjoint_from_ww():
    """The claim the whole vendored protocol rests on."""
    from baselines.prompting.predict import load_records
    from stepfinder.protocol import VENDORED_DIRS, load_vendored_records

    for train_set, subset in [("algorithm-generated", "algorithm-generated"),
                              ("hand-crafted", "hand-crafted")]:
        train_q = {r["question"].strip()
                   for r in load_vendored_records(REPO_ROOT / "vendored/StepFinder/data" / VENDORED_DIRS[train_set])}
        test_q = {r["question"].strip() for r in load_records(str(REPO_ROOT / "data/ww" / subset))}
        assert not (train_q & test_q), f"{train_set} leaks into ww/{subset}"


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------

import json
import os
import shutil
import subprocess


def _rb_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "baselines-rp"), str(REPO_ROOT), env.get("PYTHONPATH", "")]
    )
    # The dummy tokenizer derives token ids with Python's built-in `hash`,
    # which is salted per process. Tests that compare features across two
    # predict subprocesses (wide-cache slice vs fresh encode) need the salt
    # pinned; real encoders never touch Python string hashing.
    env["PYTHONHASHSEED"] = "0"
    return env


@pytest.fixture(scope="module")
def mini_corpus(tmp_path_factory):
    """Four real Who&When trajectories."""
    d = tmp_path_factory.mktemp("corpus") / "mini"
    d.mkdir()
    for i in (1, 2, 3, 4):
        shutil.copy(REPO_ROOT / f"data/ww/hand-crafted/{i}.json", d / f"{i}.json")
    return d


@pytest.fixture(scope="module")
def mini_train(tmp_path_factory):
    """Twelve real regenerated training trajectories, in the vendored layout."""
    root = tmp_path_factory.mktemp("train")
    d = root / "Hand-Crafted" / "train"
    d.mkdir(parents=True)
    src = sorted((REPO_ROOT / "vendored/StepFinder/data/Hand-Crafted/train").glob("*.json"))[:12]
    for f in src:
        shutil.copy(f, d / f.name)
    return root


def _predict(output, corpus, train_root, *extra, seeds="42"):
    argv = [
        sys.executable, "-m", "stepfinder.predict",
        "--input", str(corpus), "--output", str(output),
        "--model-name", "dummy", "--model-path", "dummy", "--device", "cpu",
        "--train-data-root", str(train_root),
        "--seeds", seeds, "--epochs", "2", "--patience", "1", "--log-every", "0",
        *extra,
    ]
    return subprocess.run(argv, cwd=REPO_ROOT, env=_rb_env(), capture_output=True, text=True)


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory, mini_corpus, mini_train):
    root = tmp_path_factory.mktemp("run")
    out = root / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    res = _predict(out, mini_corpus, mini_train, seeds="42,43")
    assert res.returncode == 0, res.stderr
    return root, out


def test_cli_writes_one_file_per_trajectory_per_seed(completed_run):
    _root, out = completed_run
    for seed in (42, 43):
        files = sorted(p.stem for p in (out / f"stepfinder.s{seed}").glob("*.json")
                       if p.stem.isdigit())
        assert files == ["1", "2", "3", "4"]


def test_cli_output_has_every_required_key(completed_run):
    _root, out = completed_run
    required = {
        "id", "filename", "question_id", "method", "model", "backend",
        "predicted_agent", "predicted_step", "gold_agent", "gold_step", "raw", "calls",
    }
    doc = json.loads((out / "stepfinder.s42" / "1.json").read_text())
    assert required <= set(doc)
    assert doc["method"] == "stepfinder" and doc["calls"] == []
    # `raw` must round-trip to `scores`, so --check-only can tell a present
    # prediction from a missing one.
    assert json.loads(doc["raw"]) == doc["scores"]
    assert len(doc["scores"]) == doc["num_steps"] == len(doc["score_step_indices"])
    assert abs(sum(doc["scores"]) - 1.0) < 1e-5


def test_cli_predicted_agent_is_the_role_of_the_predicted_step(completed_run, mini_corpus):
    _root, out = completed_run
    for path in (out / "stepfinder.s42").glob("*.json"):
        if not path.stem.isdigit():
            continue
        doc = json.loads(path.read_text())
        history = json.loads((mini_corpus / f"{doc['id']}.json").read_text())["history"]
        assert doc["predicted_agent"] == history[doc["predicted_step"]]["role"]


def test_cli_conformal_keys_are_null_not_borrowed(completed_run):
    """StepFinder has no calibration machinery; the columns must stay empty."""
    _root, out = completed_run
    doc = json.loads((out / "stepfinder.s42" / "1.json").read_text())
    assert doc["conformal_steps"] is None
    assert doc["conformal_threshold"] is None and doc["conformal_alpha"] is None
    assert doc["topk_steps"] and len(doc["topk_steps"]) == 3


def test_cli_writes_run_config(completed_run):
    _root, out = completed_run
    run = json.loads((out / "stepfinder.s42" / "_run.json").read_text())
    for key in ("model", "method", "method_dir", "subset", "backend", "gt_in_prompt",
                "protocol", "train_seed", "train_set", "n_train_task_overlap",
                "preset", "hparams", "checkpoint", "model_selection", "started_at"):
        assert key in run, key
    assert run["protocol"] == "regen" and run["preset"] == "hc"


def test_cli_caches_features_and_checkpoints_inside_the_output_root(completed_run):
    root, out = completed_run
    assert (out / "_sf-feats" / "1.pt").exists()
    assert (out / "_sf-feats" / "_manifest.json").exists()
    regen = root / "outputs-rb-nogt" / "stepfinder-regen" / "hand-crafted" / "dummy"
    assert (regen / "_sf-feats").is_dir()
    assert (regen / "_sf-ckpt" / "hc" / "val" / "s42" / "model.pt").exists()
    # underscore-prefixed, so the report and OutputWriter never see them
    from baselines.shared.runner import OutputWriter
    assert OutputWriter(out / "stepfinder.s42").done_ids() == {"1", "2", "3", "4"}


def test_cli_resumes_and_overwrites(tmp_path, mini_corpus, mini_train):
    out = tmp_path / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    assert _predict(out, mini_corpus, mini_train).returncode == 0
    before = (out / "stepfinder.s42" / "1.json").read_text()

    again = _predict(out, mini_corpus, mini_train)
    assert again.returncode == 0
    assert "skip (complete)" in again.stdout
    assert (out / "stepfinder.s42" / "1.json").read_text() == before

    (out / "stepfinder.s42" / "1.json").unlink()
    partial = _predict(out, mini_corpus, mini_train)
    assert "1 to run" in partial.stdout
    # Re-running reproduces the prediction exactly; only the wall-clock the
    # scoring took can differ.
    redone = json.loads((out / "stepfinder.s42" / "1.json").read_text())
    original = json.loads(before)
    for doc in (redone, original):
        doc.pop("inference_time_ms")
    assert redone == original


def test_cli_slices_with_start_and_end_idx(tmp_path, mini_corpus, mini_train):
    out = tmp_path / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    assert _predict(out, mini_corpus, mini_train, "--end_idx", "2").returncode == 0
    assert sorted(p.stem for p in (out / "stepfinder.s42").glob("*.json")
                  if p.stem.isdigit()) == ["1", "2"]


def test_cli_in_corpus_covers_only_that_seeds_val_and_test(tmp_path, mini_corpus, mini_train):
    from stepfinder.protocol import scored_partition_ids

    out = tmp_path / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    res = _predict(out, mini_corpus, mini_train, "--protocol", "in-corpus",
                   "--eval-seeds", "1,2")
    assert res.returncode == 0, res.stderr
    splits = {"train": 0.3, "val": 0.2, "test": 0.5}
    for seed in (1, 2):
        expected = scored_partition_ids(mini_corpus, splits, seed)
        got = {p.stem for p in (out / f"stepfinder.e{seed}").glob("*.json") if p.stem.isdigit()}
        assert got == expected
        doc = json.loads(next(iter((out / f"stepfinder.e{seed}").glob("[0-9]*.json"))).read_text())
        assert doc["protocol"] == "in-corpus" and doc["train_set"] == f"in-corpus:s{seed}"


def test_cli_gt_changes_the_features_but_not_the_model(tmp_path, mini_corpus, mini_train):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    root = tmp_path
    nogt = root / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    gt = root / "outputs-rb-gt" / "mini" / "mini" / "dummy"
    assert _predict(nogt, mini_corpus, mini_train).returncode == 0
    assert _predict(gt, mini_corpus, mini_train, "--gt", "with").returncode == 0

    a = torch.load(nogt / "_sf-feats" / "1.pt", weights_only=False)
    b = torch.load(gt / "_sf-feats" / "1.pt", weights_only=False)
    assert not np.array_equal(a["content_features"][0], b["content_features"][0])
    assert np.array_equal(a["content_features"][1:], b["content_features"][1:])
    assert np.array_equal(a["agent_features"], b["agent_features"])
    # one checkpoint, in the without-GT tree, serving both settings
    assert not (gt / "_sf-ckpt").exists()
    assert (root / "outputs-rb-nogt" / "stepfinder-regen" / "hand-crafted" / "dummy"
            / "_sf-ckpt" / "hc" / "val" / "s42" / "model.pt").exists()
    assert json.loads((gt / "stepfinder.s42" / "1.json").read_text())["gt_in_prompt"] is True


def test_cli_refuses_to_score_under_vendored_model_selection(tmp_path, mini_corpus, mini_train):
    """A number fitted while watching the test set must never reach outputs/."""
    out = tmp_path / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    res = _predict(out, mini_corpus, mini_train, "--model-selection", "vendored")
    assert res.returncode != 0
    assert "main.py:231-251" in (res.stderr + res.stdout)
    assert not (out / "stepfinder.s42").exists()


def test_cli_rejects_an_unknown_stage(tmp_path, mini_corpus, mini_train):
    res = _predict(tmp_path / "outputs-rb-nogt/x/y/dummy", mini_corpus, mini_train,
                   "--stages", "feats-train,nope")
    assert res.returncode != 0 and "unknown stage" in (res.stderr + res.stdout)


# --------------------------------------------------------------------------
# Sweep, report and the shipped configs
# --------------------------------------------------------------------------

def test_sweep_dry_run_resolves_to_the_nogt_root():
    res = subprocess.run(
        [sys.executable, "-m", "stepfinder.sweep",
         "--config", "baselines-rp/stepfinder/configs/ww.yaml",
         "--set", "models=[qwen3-embedding-0.6b]", "--set", "subsets=[hand-crafted]",
         "--dry-run"],
        cwd=REPO_ROOT, env=_rb_env(), capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "outputs-rb-nogt/ww/hand-crafted/qwen3-embedding-0.6b" in out
    assert "outputs-rb-gt" not in out
    assert "--gt without" in out and "--protocol regen" in out
    assert "--preset hc" in out and "--train-set hand-crafted" in out


def test_sweep_maps_algorithm_generated_to_the_papers_alg_settings():
    res = subprocess.run(
        [sys.executable, "-m", "stepfinder.sweep",
         "--config", "baselines-rp/stepfinder/configs/ww.yaml",
         "--set", "models=[qwen3-embedding-0.6b]", "--set", "subsets=[algorithm-generated]",
         "--dry-run"],
        cwd=REPO_ROOT, env=_rb_env(), capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert "--train-set algorithm-generated" in res.stdout
    assert "--preset alg" in res.stdout


def test_shipped_configs_parse_and_agree_with_the_corpus():
    yaml = pytest.importorskip("yaml")
    cfg_dir = REPO_ROOT / "baselines-rp/stepfinder/configs"

    for name in ("ww", "correct-error", "traceelephant", "tracertraj"):
        cfg = yaml.safe_load((cfg_dir / f"{name}.yaml").read_text())
        assert cfg["gt"] == "without"
        assert cfg["outputs_root"] == f"outputs-rb-gt/{name}"
        assert cfg["seeds"] == [42, 43, 44, 45, 46]
        assert cfg["model_selection"] == "val"
        assert cfg["agent_normalize"] == "standardize"
        for subset in cfg["subsets"]:
            assert (REPO_ROOT / cfg["data_dir"] / subset).is_dir(), f"{name}/{subset}"
        for model, spec in cfg["model_specs"].items():
            assert "model_path" in spec, model
            assert spec["dtype"] == "fp16"

        regen = yaml.safe_load((cfg_dir / f"report_{name}.yaml").read_text())
        assert regen["methods"] == [f"stepfinder.s{s}" for s in cfg["seeds"]]
        assert regen["pred_root"] == cfg["outputs_root"]
        assert regen["gt"] == "without"
        assert regen["subsets"] == cfg["subsets"]
        assert regen["models"] == cfg["models"]
        assert regen["out_root"].endswith("/reports/stepfinder")

        incorpus = yaml.safe_load((cfg_dir / f"report_{name}_incorpus.yaml").read_text())
        # The in-corpus family's method directories and the report's split
        # seeds are the same index set — that is what makes the diagonal exist.
        assert incorpus["methods"] == [f"stepfinder.e{s}" for s in cfg["eval_seeds"]]
        assert incorpus["seeds"] == cfg["eval_seeds"]
        assert incorpus["out_root"].endswith("/reports/stepfinder-incorpus")


def test_report_reads_the_rb_root_and_reports_done(completed_run, mini_corpus):
    """The shared report scores StepFinder with no changes of its own.

    Run from the output tree's own directory, because the shared ``nogt_root``
    maps the *first* path component — which is exactly how the shipped configs
    address their roots (``outputs-rb-gt/ww``).
    """
    yaml = pytest.importorskip("yaml")
    pytest.importorskip("pandas")
    root, _out = completed_run

    data_dir = root / "data"
    (data_dir / "mini").mkdir(parents=True, exist_ok=True)
    for f in mini_corpus.glob("*.json"):
        shutil.copy(f, data_dir / "mini" / f.name)

    cfg = {
        "models": ["dummy"], "subsets": ["mini"],
        "methods": ["stepfinder.s42", "stepfinder.s43"],
        "data_dir": "data",
        "pred_root": "outputs-rb-gt/mini",
        "out_root": "outputs-rb-gt/mini/reports/stepfinder",
        "gt": "without",
        "splits": {"train": 0.3, "val": 0.2, "test": 0.5},
        "seeds": [1, 2, 3],
    }
    cfg_path = root / "report.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    check = subprocess.run(
        [sys.executable, "-m", "stepfinder.report", "--config", "report.yaml", "--check-only"],
        cwd=root, env=_rb_env(), capture_output=True, text=True,
    )
    assert check.returncode == 0, check.stderr
    assert "DONE" in check.stdout and "PARTIAL" not in check.stdout

    full = subprocess.run(
        [sys.executable, "-m", "stepfinder.report", "--config", "report.yaml"],
        cwd=root, env=_rb_env(), capture_output=True, text=True,
    )
    assert full.returncode == 0, full.stderr
    nogt_out = root / "outputs-rb-nogt" / "mini" / "reports" / "stepfinder"
    assert (nogt_out / "summary_mean_over_seeds.tsv").exists()
    assert (nogt_out / "completion_status.tsv").exists()


def test_report_diagonal_keeps_only_each_methods_own_seed(tmp_path):
    """The in-corpus family's twenty-by-twenty table has twenty useful cells."""
    pd = pytest.importorskip("pandas")
    from stepfinder.report import write_diagonal

    out_root = tmp_path / "reports"
    d = out_root / "dummy" / "mini"
    d.mkdir(parents=True)
    rows = []
    for seed in (1, 2, 3):
        row = {"seed": seed, "n_val": 10, "n_test": 20, "n_full": 50}
        for m in (1, 2, 3):
            # only the diagonal cell is real; the rest are the report scoring
            # a method on ids it never predicted
            on_diagonal = m == seed
            row[f"stepfinder.e{m}_step@1_test"] = 0.5 if on_diagonal else 0.0
            row[f"stepfinder.e{m}_agent@1_test"] = 0.7 if on_diagonal else 0.0
            row[f"stepfinder.e{m}_step@1_val"] = 0.4 if on_diagonal else 0.0
            row[f"stepfinder.e{m}_agent@1_val"] = 0.6 if on_diagonal else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(d / "comparison_by_seed.tsv", sep="\t", index=False)

    written = write_diagonal(out_root)
    assert len(written) == 1
    table = pd.read_csv(written[0], sep="\t")
    body = table[table["seed"] != "mean"]
    assert list(body["method"]) == ["stepfinder.e1", "stepfinder.e2", "stepfinder.e3"]
    assert list(body["step@1_test"]) == [0.5, 0.5, 0.5]
    mean = table[table["seed"] == "mean"].iloc[0]
    assert float(mean["step@1_test"]) == pytest.approx(0.5)
    assert float(mean["agent@1_test"]) == pytest.approx(0.7)


def test_rb_metrics_reads_a_stepfinder_directory(completed_run, tmp_path):
    """Discovery finds the method dirs; conformal columns stay empty."""
    pytest.importorskip("sklearn")
    _root, out = completed_run
    dest = tmp_path / "rb_metrics.tsv"
    res = subprocess.run(
        [sys.executable, "-m", "rb_shared.rb_metrics",
         "--pred-root", str(out.parent.parent), "--out", str(dest)],
        cwd=REPO_ROOT, env=_rb_env(), capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    header, *body = dest.read_text().strip().split("\n")
    cols = header.split("\t")
    assert {"acc@1", "acc@3", "mrr@3", "tol_acc@1", "auroc"} <= set(cols)
    assert len(body) == 2  # discovered stepfinder.s42 and stepfinder.s43
    for line in body:
        cells = dict(zip(cols, line.split("\t")))
        assert cells["method"].startswith("stepfinder.s")
        assert cells["conformal_detection_f1"] == "", "conformal must not borrow top-1"
        assert cells["topk_detection_hit_rate"] != ""


def test_report_diagonal_end_to_end_on_the_in_corpus_family(tmp_path, mini_corpus, mini_train):
    """The whole in-corpus path: predict a diagonal, then reduce it to one."""
    yaml = pytest.importorskip("yaml")
    pytest.importorskip("pandas")
    import pandas as pd

    root = tmp_path
    out = root / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    res = _predict(out, mini_corpus, mini_train, "--protocol", "in-corpus",
                   "--eval-seeds", "1,2,3")
    assert res.returncode == 0, res.stderr

    data_dir = root / "data" / "mini"
    data_dir.mkdir(parents=True)
    for f in mini_corpus.glob("*.json"):
        shutil.copy(f, data_dir / f.name)

    cfg = {
        "models": ["dummy"], "subsets": ["mini"],
        "methods": [f"stepfinder.e{s}" for s in (1, 2, 3)],
        "data_dir": "data",
        "pred_root": "outputs-rb-gt/mini",
        "out_root": "outputs-rb-gt/mini/reports/stepfinder-incorpus",
        "gt": "without",
        "splits": {"train": 0.3, "val": 0.2, "test": 0.5},
        "seeds": [1, 2, 3],
    }
    (root / "report.yaml").write_text(yaml.safe_dump(cfg))

    # --check-only must report PARTIAL: covering only val+test is the design.
    check = subprocess.run(
        [sys.executable, "-m", "stepfinder.report", "--config", "report.yaml", "--check-only"],
        cwd=root, env=_rb_env(), capture_output=True, text=True,
    )
    assert check.returncode == 0, check.stderr
    assert "PARTIAL" in check.stdout

    full = subprocess.run(
        [sys.executable, "-m", "stepfinder.report", "--config", "report.yaml", "--diagonal"],
        cwd=root, env=_rb_env(), capture_output=True, text=True,
    )
    assert full.returncode == 0, full.stderr

    out_root = root / "outputs-rb-nogt" / "mini" / "reports" / "stepfinder-incorpus"
    diag = out_root / "dummy" / "mini" / "diagonal.tsv"
    assert diag.exists(), sorted(p.name for p in out_root.rglob("*"))
    table = pd.read_csv(diag, sep="\t")
    body = table[table["seed"] != "mean"]
    assert list(body["method"]) == ["stepfinder.e1", "stepfinder.e2", "stepfinder.e3"]
    assert "mean" in set(table["seed"].astype(str))
    assert {"step@1_test", "agent@1_test"} <= set(table.columns)


def test_in_corpus_trains_on_without_gt_features_in_both_settings(tmp_path, mini_corpus, mini_train):
    """One model per seed, in the nogt tree, whichever GT setting is scored.

    Family B trains on the corpus itself, whose features *are* GT-dependent —
    so without a rule it would fit two different models and the with/without
    comparison would stop being paired. The rule: always train on the
    without-GT features, so only what the model reads at scoring time changes.
    """
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    root = tmp_path
    nogt = root / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    gt = root / "outputs-rb-gt" / "mini" / "mini" / "dummy"
    common = ("--protocol", "in-corpus", "--eval-seeds", "1")
    assert _predict(nogt, mini_corpus, mini_train, *common).returncode == 0
    assert _predict(gt, mini_corpus, mini_train, *common, "--gt", "with").returncode == 0

    assert list((root / "outputs-rb-gt").rglob("model.pt")) == []
    assert len(list((root / "outputs-rb-nogt").rglob("model.pt"))) == 1

    a = torch.load(nogt / "_sf-feats" / "1.pt", weights_only=False)
    b = torch.load(gt / "_sf-feats" / "1.pt", weights_only=False)
    assert (a["gt_in_prompt"], b["gt_in_prompt"]) == (False, True)
    assert not np.array_equal(a["content_features"][0], b["content_features"][0])
    assert np.array_equal(a["content_features"][1:], b["content_features"][1:])


def test_a_holdout_too_small_to_measure_falls_back_to_the_full_budget(tmp_path, mini_corpus, mini_train):
    """Selecting on a handful of trajectories is worse than not selecting.

    With `--val-frac 0.1` on a small partition the holdout can come out at four
    or five trajectories. Early stopping then picks epoch 1 about as often as
    the best one — because a 5-sample accuracy is noise and the vendored rule
    needs only a strict improvement over 0.0 to latch — and hands back a
    barely-trained network. Measured on `correct-error/math500`, that scored
    `step@1` 0.012 against 0.042 for simply training the full budget, i.e.
    below chance. The guard is a floor on holdout *size*, not just task count.
    """
    pytest.importorskip("torch")
    import json as _json

    torch = pytest.importorskip("torch")

    # A training set spanning several questions, since the holdout is cut along
    # question boundaries — the shared fixture is 12 variations of one task.
    wide = tmp_path / "wide" / "Hand-Crafted" / "train"
    wide.mkdir(parents=True)
    src = sorted((REPO_ROOT / "vendored/StepFinder/data/Hand-Crafted/train").glob("*.json"))
    for f in src[::18][:8]:            # ~18 trajectories per task upstream
        shutil.copy(f, wide / f.name)
    assert len({_json.loads(f.read_text())["question"] for f in wide.glob("*.json")}) > 1

    def n_val_of(root):
        ckpt = next((root / "outputs-rb-nogt" / "stepfinder-regen" / "hand-crafted"
                     / "dummy" / "_sf-ckpt").rglob("model.pt"))
        return torch.load(ckpt, map_location="cpu", weights_only=False)["n_val"]

    # Floor of 20: eight training trajectories cannot supply it, so no holdout
    # is taken and the full budget is trained.
    strict = tmp_path / "strict"
    res = _predict(strict / "outputs-rb-nogt" / "mini" / "mini" / "dummy",
                   mini_corpus, tmp_path / "wide",
                   "--val-min-size", "20", "--val-min-tasks", "1", "--val-frac", "0.25")
    assert res.returncode == 0, res.stderr
    assert n_val_of(strict) == 0, "the size floor should have suppressed the holdout"

    # Floor of 1 lets the same tiny holdout through — which is precisely what
    # the default of 20 exists to prevent.
    loose = tmp_path / "loose"
    res2 = _predict(loose / "outputs-rb-nogt" / "mini" / "mini" / "dummy",
                    mini_corpus, tmp_path / "wide",
                    "--val-min-size", "1", "--val-min-tasks", "1", "--val-frac", "0.25")
    assert res2.returncode == 0, res2.stderr
    assert 0 < n_val_of(loose) < 20, "a low floor should permit the tiny holdout"


def test_default_holdout_floor_is_stated_in_the_shipped_configs():
    yaml = pytest.importorskip("yaml")
    for name in ("ww", "correct-error", "traceelephant", "tracertraj", "tracertraj-alg"):
        cfg = yaml.safe_load(
            (REPO_ROOT / "baselines-rp/stepfinder/configs" / f"{name}.yaml").read_text())
        assert cfg["val_min_size"] == 20
        assert cfg["val_min_tasks"] == 4


def test_min_train_steps_scales_epochs_for_a_small_partition(tmp_path, mini_corpus, mini_train):
    """The vendored 50 epochs is an update count in disguise.

    50 epochs over the 2,604-trajectory training corpus is ~8,000 optimizer
    steps. Over a 91-trajectory in-corpus partition it is ~300 — a fiftieth of
    what the Hand-Crafted preset's learning rate of 1e-5 was tuned for, and the
    model barely leaves initialization. Measured on `correct-error/arc`, family
    B went from `step@1` 0.031 at 50 epochs to 0.815 at 800. `--min-train-steps`
    raises the epoch count until the budget is comparable; family A already
    clears the floor, so only family B moves.
    """
    out = tmp_path / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    res = _predict(out, mini_corpus, mini_train, "--min-train-steps", "60",
                   "--batch-size", "4")
    assert res.returncode == 0, res.stderr
    # 12 training trajectories / batch 4 = 3 batches per epoch, so reaching 60
    # steps needs 20 epochs — well above the 2 that `_predict` asks for.
    assert "20 epochs" in res.stdout, res.stdout

    # ...and with the floor off, the requested epoch count stands.
    out2 = tmp_path / "off" / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    res2 = _predict(out2, mini_corpus, mini_train, "--batch-size", "4")
    assert res2.returncode == 0, res2.stderr
    assert "2 epochs" in res2.stdout, res2.stdout


def test_shipped_configs_set_the_optimizer_step_floor():
    yaml = pytest.importorskip("yaml")
    for name in ("ww", "correct-error", "traceelephant", "tracertraj", "tracertraj-alg"):
        cfg = yaml.safe_load(
            (REPO_ROOT / "baselines-rp/stepfinder/configs" / f"{name}.yaml").read_text())
        assert cfg["min_train_steps"] == 5000
        # family A must be unaffected: 2604 trajectories at batch 16 for 50
        # epochs is already well past the floor
        assert -(-2604 // cfg["batch_size"]) * cfg["epochs"] > cfg["min_train_steps"]


def test_tracertraj_runs_both_training_corpora_as_separate_families():
    """tracertraj's MetaGPT agents match neither vendored corpus, so the two
    arms ship as two configs that differ only in corpus, preset and method-dir
    prefix, and the second arm's dry run names its own family."""
    yaml = pytest.importorskip("yaml")
    cfg_dir = REPO_ROOT / "baselines-rp/stepfinder/configs"
    hc = yaml.safe_load((cfg_dir / "tracertraj.yaml").read_text())
    alg = yaml.safe_load((cfg_dir / "tracertraj-alg.yaml").read_text())
    assert (hc["default_train_set"], hc["default_preset"]) == ("hand-crafted", "hc")
    assert (alg["default_train_set"], alg["default_preset"]) == ("algorithm-generated", "alg")
    assert alg["method_dir_prefix"] == "stepfinder-alg" and "method_dir_prefix" not in hc
    differing = {k for k in set(hc) | set(alg) if hc.get(k) != alg.get(k)}
    assert differing == {"default_train_set", "default_preset", "method_dir_prefix"}
    assert alg["outputs_root"] == hc["outputs_root"]      # side by side, same tree

    report = yaml.safe_load((cfg_dir / "report_tracertraj_alg.yaml").read_text())
    assert report["methods"] == [f"stepfinder-alg.s{s}" for s in alg["seeds"]]
    assert report["out_root"].endswith("/reports/stepfinder-alg")

    result = subprocess.run(
        [sys.executable, "-m", "stepfinder.sweep",
         "--config", str(cfg_dir / "tracertraj-alg.yaml"), "--dry-run"],
        cwd=REPO_ROOT, env=_rb_env(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--method-dir-prefix stepfinder-alg" in result.stdout
    assert "--train-set algorithm-generated" in result.stdout
    assert "outputs-rb-nogt/tracertraj/code/" in result.stdout


# ---------------------------------------------------------------------------
# PCA reduction (reduce.py) and the test-selection variant
# ---------------------------------------------------------------------------

def test_reducer_components_are_orthonormal_and_pin_their_signs():
    import numpy as np
    from stepfinder.reduce import _components

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 40)).astype(np.float32)
    V, kept = _components(X, 8)
    assert V.shape == (40, 8)
    assert np.allclose(V.T @ V, np.eye(8), atol=1e-5)
    assert 0.0 < kept <= 1.0
    # sign pinning makes a refit bit-identical
    V2, _ = _components(X, 8)
    assert np.array_equal(V, V2)


def test_reducer_keeps_a_planted_low_rank_signal():
    """Data living in a known 3-dim subspace is reconstructed exactly by a
    3-component reducer — the property the slice cannot promise."""
    import numpy as np
    from stepfinder.reduce import _components

    rng = np.random.default_rng(1)
    basis, _ = np.linalg.qr(rng.standard_normal((40, 3)))
    X = (rng.standard_normal((300, 3)) @ basis.T).astype(np.float32)
    V, kept = _components(X, 3)
    assert kept > 0.999999
    assert np.allclose((X @ V) @ V.T, X, atol=1e-4)


def test_reducer_on_narrow_input_is_the_slice():
    """When the encoder is already no wider than the target, projection must
    degrade to the prefix slice (identity, zero-padded)."""
    import numpy as np
    from stepfinder.reduce import _components

    V, kept = _components(np.random.default_rng(2).standard_normal((50, 8)).astype(np.float32), 8)
    assert np.array_equal(V, np.eye(8, dtype=np.float32))
    assert kept == 1.0


def test_reducer_is_uncentered_so_cosines_survive_projection():
    """Vectors inside the kept subspace keep their exact cosines — the agent
    bias `cos(r_i, r_j)` must mean the same thing after reduction. A centered
    PCA moves the origin and fails this."""
    import numpy as np
    from stepfinder.reduce import _components

    rng = np.random.default_rng(3)
    basis, _ = np.linalg.qr(rng.standard_normal((40, 4)))
    # offset far from the origin: centering would subtract most of it
    X = (rng.standard_normal((300, 4)) @ basis.T + 0).astype(np.float32)
    V, _ = _components(X, 4)
    a, b = X[0] @ V, X[1] @ V
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    assert abs(cos(a, b) - cos(X[0], X[1])) < 1e-5


def test_fit_and_apply_reducer_roundtrip_and_stamp_provenance():
    import numpy as np
    from stepfinder.reduce import apply_reducer, fit_reducer, load_reducer, save_reducer

    rng = np.random.default_rng(4)
    payloads = [{
        "content_features": rng.standard_normal((5, 160)).astype(np.float32),
        "agent_features": rng.standard_normal((5, 160)).astype(np.float32),
    } for _ in range(6)]
    red = fit_reducer(payloads)
    assert red["content_V"].shape == (160, 128)
    assert red["agent_V"].shape == (160, 32)
    assert red["centered"] is False
    assert red["n_fit_trajectories"] == 6

    out = apply_reducer(payloads[0], red)
    assert out["content_features"].shape == (5, 128)
    assert out["agent_features"].shape == (5, 32)
    assert out["reduce"] == "pca"
    # original untouched
    assert payloads[0]["content_features"].shape == (5, 160)

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        save_reducer(red, Path(d))
        back = load_reducer(Path(d))
        assert np.array_equal(back["content_V"], red["content_V"])


@pytest.fixture(scope="module")
def pca_run(tmp_path_factory, mini_corpus, mini_train):
    root = tmp_path_factory.mktemp("pca-run")
    out = root / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    res = _predict(out, mini_corpus, mini_train, "--reduce", "pca", seeds="42")
    assert res.returncode == 0, res.stderr
    return root, out


def test_cli_pca_writes_its_own_method_family(pca_run):
    _root, out = pca_run
    files = sorted(p.stem for p in (out / "stepfinder-pca.s42").glob("*.json") if p.stem.isdigit())
    assert files == ["1", "2", "3", "4"]
    assert not (out / "stepfinder.s42").exists(), \
        "a pca run must never write into the slice family's directories"
    doc = json.loads((out / "stepfinder-pca.s42" / "1.json").read_text())
    assert doc["method_dir"] == "stepfinder-pca.s42"
    assert doc["reduce"] == "pca"
    assert doc["pca_dim_in"] == 160          # the dummy encoder's width
    assert 0 < doc["pca_content_var_kept"] <= 1


def test_cli_pca_caches_wide_features_and_one_reducer(pca_run):
    import torch
    root, out = pca_run
    # test features at the encoder's full width
    wide = torch.load(next((out / "_sf-featsw").glob("[0-9]*.pt")), weights_only=False)
    assert wide["content_features"].shape[1] == 160
    assert wide["agent_features"].shape[1] == 160
    # training corpus: wide cache and a fitted reducer, no sliced cache
    regen = root / "outputs-rb-nogt" / "stepfinder-regen" / "hand-crafted" / "dummy"
    assert (regen / "_sf-featsw").is_dir()
    assert (regen / "_sf-pca" / "reducer.pt").exists()
    # checkpoints under a pca-tagged selection dir, apart from slice models
    assert (regen / "_sf-ckpt" / "hc" / "val-pca" / "s42" / "model.pt").exists()


def test_cli_slice_features_derive_bit_identically_from_a_wide_cache(pca_run, mini_corpus, mini_train):
    """After a pca run, a slice run re-uses the wide cache: no encoder pass,
    and the derived features equal a from-scratch sliced encode exactly."""
    import numpy as np
    import torch
    root, out = pca_run
    res = _predict(out, mini_corpus, mini_train, seeds="42")   # slice, val
    assert res.returncode == 0, res.stderr
    derived = torch.load(out / "_sf-feats" / "1.pt", weights_only=False)

    fresh_out = root / "outputs-rb-nogt" / "mini" / "fresh" / "dummy"
    res = _predict(fresh_out, mini_corpus, mini_train, seeds="42")
    assert res.returncode == 0, res.stderr
    fresh = torch.load(fresh_out / "_sf-feats" / "1.pt", weights_only=False)

    assert np.array_equal(derived["content_features"], fresh["content_features"])
    assert np.array_equal(derived["agent_features"], fresh["agent_features"])
    assert derived["content_features"].shape[1] == 128


@pytest.fixture(scope="module")
def ww_corpus(tmp_path_factory, mini_corpus):
    """The mini corpus posing as data/ww/hand-crafted — the subset
    test-selection watches for HC-trained models."""
    d = tmp_path_factory.mktemp("wwdata") / "ww" / "hand-crafted"
    d.mkdir(parents=True)
    for f in mini_corpus.glob("*.json"):
        shutil.copy(f, d / f.name)
    return d


@pytest.fixture(scope="module")
def tsel_run(tmp_path_factory, ww_corpus, mini_train):
    root = tmp_path_factory.mktemp("tsel-run")
    out = root / "outputs-rb-nogt" / "ww" / "hand-crafted" / "dummy"
    res = _predict(out, ww_corpus, mini_train, "--model-selection", "test", seeds="42")
    assert res.returncode == 0, res.stderr
    return root, out


def test_cli_test_selection_writes_tsel_and_stamps_it(tsel_run):
    _root, out = tsel_run
    files = sorted(p.stem for p in (out / "stepfinder-tsel.s42").glob("*.json") if p.stem.isdigit())
    assert files == ["1", "2", "3", "4"]
    doc = json.loads((out / "stepfinder-tsel.s42" / "1.json").read_text())
    assert doc["method_dir"] == "stepfinder-tsel.s42"
    assert doc["model_selection"].startswith("test")
    cfg = json.loads((out / "stepfinder-tsel.s42" / "_run.json").read_text())
    assert cfg["model_selection"].startswith("test")


def test_cli_test_selected_checkpoint_lives_with_the_ww_subset(tsel_run):
    """A test-selected model is picked by watching Who&When — the paper's
    rule — so it lives under the matching WW subset's dir, never in the
    shared regen root where a val model could be mistaken for it."""
    root, out = tsel_run
    assert (out / "_sf-ckpt" / "hc" / "test" / "s42" / "model.pt").exists()
    regen = root / "outputs-rb-nogt" / "stepfinder-regen" / "hand-crafted" / "dummy"
    assert not (regen / "_sf-ckpt" / "hc" / "test").exists()


def test_cli_transfer_subset_reuses_the_ww_selected_model(tsel_run, mini_corpus, mini_train):
    """A non-WW subset scores with the WW-selected checkpoint: same weights,
    selection set disjoint from what it reports. It must refuse to train its
    own, and must succeed once the WW checkpoint exists."""
    root, _ww_out = tsel_run
    out = root / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    res = _predict(out, mini_corpus, mini_train, "--model-selection", "test", seeds="42")
    assert res.returncode == 0, res.stderr
    doc = json.loads((out / "stepfinder-tsel.s42" / "1.json").read_text())
    assert "/ww/hand-crafted/dummy/" in doc["checkpoint"]

    # with no WW checkpoint for this corpus, training must refuse loudly
    bare = root / "elsewhere" / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    res = _predict(bare, mini_corpus, mini_train, "--model-selection", "test", seeds="42")
    assert res.returncode != 0
    assert "ww" in res.stderr


def test_cli_pca_and_test_selection_compose(tmp_path, ww_corpus, mini_train):
    out = tmp_path / "outputs-rb-nogt" / "ww" / "hand-crafted" / "dummy"
    res = _predict(out, ww_corpus, mini_train, "--reduce", "pca",
                   "--model-selection", "test", seeds="42")
    assert res.returncode == 0, res.stderr
    assert (out / "stepfinder-pca-tsel.s42" / "1.json").exists()
    assert (out / "_sf-ckpt" / "hc" / "test-pca" / "s42" / "model.pt").exists()
    doc = json.loads((out / "stepfinder-pca-tsel.s42" / "1.json").read_text())
    assert doc["reduce"] == "pca" and doc["model_selection"].startswith("test")


def test_cli_pca_refuses_the_in_corpus_protocol(tmp_path, mini_corpus, mini_train):
    out = tmp_path / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    res = _predict(out, mini_corpus, mini_train, "--reduce", "pca",
                   "--protocol", "in-corpus", "--eval-seeds", "1")
    assert res.returncode != 0
    assert "regen protocol only" in res.stderr


def test_test_selection_accuracy_is_what_deployment_reproduces(tsel_run):
    """The selection metric must be the deployed metric. The position bias is
    batch-dependent (`model.py:242`), so a checkpoint selected with batched
    evaluation can score 0 when deployed at batch 1 — found the hard way on
    ww/hand-crafted, 12.1% at selection vs 0.0% deployed. Test-selection now
    evaluates at batch 1, so the checkpoint's recorded accuracy must equal the
    accuracy recomputed from its own prediction files exactly."""
    import torch
    _root, out = tsel_run
    blob = torch.load(out / "_sf-ckpt" / "hc" / "test" / "s42" / "model.pt",
                      map_location="cpu", weights_only=False)
    docs = [json.loads(p.read_text())
            for p in (out / "stepfinder-tsel.s42").glob("*.json") if p.stem.isdigit()]
    acc = sum(int(str(d["predicted_step"]) == str(d["gold_step"])) for d in docs) / len(docs)
    if blob["model_selection"] == "test":          # not a fallback
        assert abs(acc - blob["best_val_acc"]) < 1e-9, \
            f"deployed acc {acc} != selection acc {blob['best_val_acc']}"
