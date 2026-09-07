"""CPU-only, keyless tests for the OAT baseline (baselines-rp/oat/).

The suite has one job above all others: prove that what this repo runs is what
``vendored/OAT/`` runs. Where a vendored module can be imported and driven, the
test drives it and compares byte-for-byte or bit-for-bit; where it cannot, the
test pins the values the vendored source states.

Importing the vendored code is awkward on purpose. ``vendored/OAT/`` is a flat
script directory whose modules import each other by bare name (``import
config``), and two of those names — ``config``, ``baselines`` — would shadow
this repo's own modules if the directory ever landed on ``sys.path``
permanently. ``vendored_oat`` below therefore puts it on the path, imports what
it was asked for, and puts everything back.
"""
from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_OAT = REPO_ROOT / "vendored" / "OAT"


@contextlib.contextmanager
def vendored_oat(*module_names):
    """Import modules from ``vendored/OAT/`` under their bare names, then undo it.

    Yields the imported modules in the order requested. On exit the path entry
    goes away and every module that came *out of that directory* is dropped,
    so a shadowed ``config`` or ``baselines`` cannot leak into the rest of the
    session. Modules from anywhere else stay: third-party packages the
    vendored code pulled in on its way (transformers, sklearn) are shared with
    the session, and tearing them down mid-import leaves them unusable.
    """
    before = dict(sys.modules)
    vendored_dir = str(VENDORED_OAT)
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


# --------------------------------------------------------------------------
# The neural CDE core
# --------------------------------------------------------------------------

def test_model_constants_match_vendored_config():
    """Our module-level constants restate vendored/OAT/config.py exactly."""
    from oat import model as ours

    with vendored_oat("config") as (vendored_config,):
        for name in (
            "LATENT_DIM",
            "OAT_HIDDEN",
            "OAT_DEPTH",
            "OAT_INTERPOLATION",
            "OAT_SOLVER",
            "OAT_ADJOINT",
            "OAT_CONTROL_GATE",
            "OAT_CONTROL_GATE_HIDDEN",
            "OAT_CONTROL_GATE_DEPTH",
            "OAT_CONTROL_GATE_EPS",
            "OAT_USE_QUESTION_H0",
            "OAT_H0_NORM_REG",
            "OAT_H1_BRIDGE_HIDDEN",
            "OAT_H1_BRIDGE_DEPTH",
            "OAT_H1_BRIDGE_SOLVER",
        ):
            assert getattr(ours, name) == getattr(vendored_config, name), name


@pytest.mark.parametrize("n_steps", [2, 3, 9])
def test_model_forward_matches_vendored(n_steps):
    """Same weights, same input, bit-identical loss and per-step scores."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchcde")
    pytest.importorskip("torchdiffeq")

    from oat.model import OATModel

    with vendored_oat("models") as (vendored_models,):
        torch.manual_seed(0)
        ours = OATModel()
        theirs = vendored_models.OATModel()
        theirs.load_state_dict(ours.state_dict())

        torch.manual_seed(1)
        z_obs = torch.randn(n_steps, ours.latent_dim, dtype=torch.float32)
        time_points = torch.linspace(0.0, 1.0, n_steps, dtype=torch.float32)

        out_ours = ours(z_obs, time_points)
        out_theirs = theirs(z_obs, time_points)

    assert torch.equal(out_ours["recon_per_step"], out_theirs["recon_per_step"])
    assert torch.equal(out_ours["loss"], out_theirs["loss"])
    # the question row carries no score of its own; it is a placeholder zero
    assert out_ours["recon_per_step"].shape == (n_steps,)
    assert out_ours["recon_per_step"][0] == 0.0


def test_model_scores_a_single_step_trajectory_as_zeros():
    """A trajectory with nothing after the question yields no usable score."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchcde")

    from oat.model import OATModel

    model = OATModel()
    z_obs = torch.randn(1, model.latent_dim)
    out = model(z_obs, torch.zeros(1))
    assert out["recon_per_step"].shape == (1,)
    assert float(out["loss"]) == 0.0


# --------------------------------------------------------------------------
# Serialization: the document the extractor reads
# --------------------------------------------------------------------------

def _ww_records(n=4):
    """A few real Who&When hand-crafted records, the vendored reference format."""
    import json

    files = sorted((REPO_ROOT / "data" / "ww" / "hand-crafted").glob("*.json"), key=lambda p: int(p.stem))[:n]
    return [json.loads(p.read_text()) for p in files]


def test_serialize_matches_vendored_hand_crafted():
    """Byte-identical text, bounds and agent labels on real records."""
    from oat.serialize import serialize_trajectory

    records = _ww_records()
    assert records, "fixture corpus missing"

    with vendored_oat("data_pipeline") as (vendored_dp,):
        for record in records:
            ours = serialize_trajectory(record["history"])
            theirs = vendored_dp.serialize_who_and_when_trajectory(record["history"], "hand_crafted")
            assert ours[0] == theirs[0]          # text
            assert ours[1] == theirs[1]          # char bounds
            assert ours[2] == theirs[2]          # agents
            assert ours[3] == theirs[3]          # model-step flags


def test_build_input_text_matches_vendored():
    """The question prefix and the shifted step bounds are the vendored ones."""
    from oat.serialize import build_input_text, serialize_trajectory

    records = _ww_records()

    with vendored_oat("extract_states") as (vendored_es,):
        for record in records:
            text, bounds, _agents, _flags = serialize_trajectory(record["history"])
            question = record.get("question", "")
            ours = build_input_text(question, text, bounds)
            theirs = vendored_es._build_input_text(
                {"question": question, "text": text, "step_char_boundaries": bounds}
            )
            assert ours[0] == theirs[0]
            assert ours[1] == theirs[1]


def test_question_span_covers_the_question():
    """Row 0's char span is exactly the question, wherever the answer goes."""
    from oat.serialize import build_input_text

    text, bounds = build_input_text("what colour is the sky", "[STEP 0] [ROLE: a] [AGENT: a]\nblue\n", [(0, 33)])
    q_start, q_end = bounds[0]
    assert text[q_start : q_end + 1] == "what colour is the sky"


def test_gt_line_appears_only_with_gt():
    """--gt with adds the repo's standard answer sentence, and nothing else."""
    from oat.serialize import question_text

    record = {"question": "how many apples?", "ground_truth": "seven"}
    assert question_text(record, include_gt=False) == "how many apples?"
    assert question_text(record, include_gt=True) == (
        "how many apples?\nThe Answer for the problem is: seven"
    )
    # the difference is exactly one appended line
    with_gt = question_text(record, include_gt=True)
    assert with_gt.startswith(question_text(record, include_gt=False))


def test_model_step_filter_rule():
    """Human and environment turns are dropped; tool-named agent roles stay."""
    from oat.serialize import is_model_step

    for role in ("human", "user", "Computer_terminal", "ComputerTerminal"):
        assert not is_model_step({"role": role}), role
    for role in ("Orchestrator (thought)", "WebSurfer", "Planner", "bash", "str_replace_editor", "submit"):
        assert is_model_step({"role": role}), role


def test_agent_label_collapses_orchestrator_variants():
    from oat.serialize import agent_of

    assert agent_of({"role": "Orchestrator (thought)"}) == "Orchestrator"
    assert agent_of({"role": "Orchestrator (-> WebSurfer)"}) == "Orchestrator"
    assert agent_of({"role": "WebSurfer"}) == "WebSurfer"
    assert agent_of({"role": "human"}) == "human"


def test_mcp_atlas_serializer_matches_vendored():
    """The training corpus is serialized exactly as the authors serialized it."""
    import json

    from oat.serialize import serialize_mcp_atlas_trajectory

    files = sorted((VENDORED_OAT / "dataset" / "MCP-atlas" / "Qwen3.5-27B").glob("*.json"))[:3]
    assert files, "vendored MCP-Atlas corpus missing"

    with vendored_oat("data_pipeline") as (vendored_dp,):
        for path in files:
            history = json.loads(path.read_text()).get("raw_conversation_history", [])
            assert serialize_mcp_atlas_trajectory(history) == vendored_dp.serialize_mcp_atlas_trajectory(history)


# --------------------------------------------------------------------------
# Extraction: tokens to step vectors
# --------------------------------------------------------------------------

def test_token_span_helpers_match_vendored():
    """Both span strategies agree with the vendored implementations."""
    from oat.extract import get_step_last_token_indices, get_step_token_ranges

    offsets = [(0, 0), (0, 5), (5, 9), (9, 14), (14, 20), (0, 0)]
    bounds = [(0, 8), (9, 19), (3, 3)]

    with vendored_oat("data_pipeline") as (vendored_dp,):
        assert get_step_token_ranges(offsets, bounds) == vendored_dp.get_step_token_ranges(offsets, bounds)
        assert get_step_last_token_indices(offsets, bounds) == vendored_dp.get_step_last_token_indices(offsets, bounds)


@pytest.mark.parametrize("aggregation", ["mean", "last"])
def test_extraction_matches_vendored(aggregation):
    """Same document, same model: identical step vectors, all layers kept."""
    torch = pytest.importorskip("torch")

    from oat.extract import dummy_extractor, extract_hidden_states_for_trajectory
    from oat.serialize import build_input_text, serialize_trajectory

    ex = dummy_extractor()
    record = _ww_records(1)[0]
    step_text, bounds, _a, _f = serialize_trajectory(record["history"])
    text, all_bounds = build_input_text(record.get("question", ""), step_text, bounds)

    ours = extract_hidden_states_for_trajectory(
        text, all_bounds, ex.model, ex.tokenizer, aggregation=aggregation, layers=None
    )

    with vendored_oat("extract_states") as (vendored_es,):
        theirs = vendored_es.extract_hidden_states_for_trajectory(
            text, all_bounds, ex.model, ex.tokenizer, aggregation=aggregation
        )

    assert torch.equal(ours, theirs)
    assert ours.shape[0] == len(all_bounds)


def test_layer_slicing_selects_the_named_layers():
    """Storing one layer stores exactly that layer of the full stack."""
    torch = pytest.importorskip("torch")

    from oat.extract import dummy_extractor, extract_hidden_states_for_trajectory
    from oat.serialize import build_input_text, serialize_trajectory

    ex = dummy_extractor()
    record = _ww_records(1)[0]
    step_text, bounds, _a, _f = serialize_trajectory(record["history"])
    text, all_bounds = build_input_text(record.get("question", ""), step_text, bounds)

    full = extract_hidden_states_for_trajectory(text, all_bounds, ex.model, ex.tokenizer, layers=None)
    last = extract_hidden_states_for_trajectory(text, all_bounds, ex.model, ex.tokenizer, layers=(-1,))
    assert last.shape == (full.shape[0], 1, full.shape[2])
    assert torch.equal(last[:, 0, :], full[:, -1, :])


# --------------------------------------------------------------------------
# Latents, training, scoring
# --------------------------------------------------------------------------

def _synthetic_trajectories(n_success=6, n_failure=3, dim=12, seed=0):
    """Trajectories shaped like cached states, small enough to run on CPU."""
    import numpy as np
    import torch

    rng = np.random.RandomState(seed)
    out = []
    for i in range(n_success + n_failure):
        is_success = i < n_success
        steps = 4 + (i % 3)
        hidden = torch.from_numpy(rng.randn(steps + 1, dim).astype("float32"))
        out.append(
            {
                "id": f"{'s' if is_success else 'f'}{i}",
                "is_success": is_success,
                "hidden_states": hidden,
                "step_indices": [-1] + list(range(steps)),
                "step_is_model": [True] * (steps + 1),
                "error_steps": [] if is_success else [1],
                "num_steps": steps,
                "num_steps_original": steps,
            }
        )
    return out[:n_success], out[n_success:]


def test_latent_pipeline_matches_vendored():
    """PCA, normalization and CORAL all reproduce the vendored numbers."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")

    import copy

    from oat import train as ours

    success, failure = _synthetic_trajectories()
    ours_s, ours_f = copy.deepcopy(success), copy.deepcopy(failure)
    theirs_s, theirs_f = copy.deepcopy(success), copy.deepcopy(failure)
    latent_dim = 8

    pca_ours = ours.fit_pca(ours_s, n_components=latent_dim)
    ours.apply_pca(ours_s, pca_ours)
    ours.apply_pca(ours_f, pca_ours)
    mean_o, std_o = ours.compute_normalization_stats(ours_s)
    ours.normalize_trajectories(ours_s, mean_o, std_o)
    ours.normalize_trajectories(ours_f, mean_o, std_o)
    ours.apply_coral_alignment(ours_f, ours_s)

    with vendored_oat("data_pipeline") as (vendored_dp,):
        pca_theirs = vendored_dp.fit_pca(theirs_s, n_components=latent_dim)
        vendored_dp.apply_pca(theirs_s, pca_theirs)
        vendored_dp.apply_pca(theirs_f, pca_theirs)
        mean_t, std_t = vendored_dp.compute_normalization_stats(theirs_s)
        vendored_dp.normalize_trajectories(theirs_s, mean_t, std_t)
        vendored_dp.normalize_trajectories(theirs_f, mean_t, std_t)
        vendored_dp.apply_coral_alignment(theirs_f, theirs_s)

    assert torch.equal(mean_o, mean_t) and torch.equal(std_o, std_t)
    for a, b in zip(ours_s + ours_f, theirs_s + theirs_f):
        assert torch.allclose(a["latent_states"], b["latent_states"], atol=1e-6)


def test_time_points_and_split_match_vendored():
    torch = pytest.importorskip("torch")

    from oat import train as ours

    success, _ = _synthetic_trajectories()

    with vendored_oat("data_pipeline", "train") as (vendored_dp, vendored_train):
        assert torch.equal(ours.get_time_points(5), vendored_dp.get_time_points(5))
        assert torch.equal(ours.get_time_points(1), vendored_dp.get_time_points(1))
        for seed in (42, 43):
            a_tr, a_va = ours.split_train_val(success, seed=seed)
            b_tr, b_va = vendored_train.split_train_val(success, seed=seed)
            assert [t["id"] for t in a_tr] == [t["id"] for t in b_tr]
            assert [t["id"] for t in a_va] == [t["id"] for t in b_va]


def test_detection_rules_match_vendored():
    """Argmax, top-k, MAD normalization and the conformal threshold."""
    np = pytest.importorskip("numpy")

    from oat import train as ours

    scores = np.array([0.4, 9.1, 0.5, 0.2, 3.3, 0.45])
    step_indices = [0, 1, 2, 3, 4, 5]
    calibration = np.array([0.1, 0.3, 0.9, 2.2, 0.4, 0.15, 5.0, 0.6])

    with vendored_oat("train") as (vendored_train,):
        assert np.allclose(ours._normalize_scores(scores), vendored_train._normalize_scores(scores))
        for alpha in (0.05, 0.1, 0.2):
            assert ours.conformal_quantile_threshold(calibration, alpha) == (
                vendored_train.conformal_quantile_threshold(calibration, alpha)
            )
        for k in (1, 3, 99):
            assert ours._topk_detection(scores, step_indices, k) == (
                vendored_train._topk_detection(scores, step_indices, k)
            )
        for threshold in (0.5, 2.0, float("inf")):
            assert ours._conformal_detection(scores, step_indices, threshold, 1) == (
                vendored_train._conformal_detection(scores, step_indices, threshold, 1)
            )


def test_conformal_falls_back_to_the_top_step():
    """An empty detection set never stays empty — the top step stands in."""
    np = pytest.importorskip("numpy")

    from oat import train as ours

    scores = np.array([0.1, 0.2, 0.15])
    assert ours._conformal_detection(scores, [0, 1, 2], float("inf"), 1) == [1]
    assert ours._conformal_detection(scores, [0, 1, 2], float("inf"), 0) == []


def test_question_row_is_never_predicted():
    """Row 0 is the question; its index is -1 and it is dropped before argmax."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    from oat import train as ours

    # the question row carries the largest raw value; it must still be ignored
    scores = torch.tensor([99.0, 0.1, 7.0, 0.2])
    traj = {"step_indices": [-1, 0, 1, 2]}
    kept, indices = ours._scorable_scores_and_indices(scores, traj)
    assert indices == [0, 1, 2]
    assert int(np.argmax(kept)) == 1  # absolute step 1, not the question


def test_scoring_matches_vendored_end_to_end():
    """Same weights and latents produce the same prediction and the same sets."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchcde")

    import copy

    from oat import train as ours
    from oat.model import OATModel

    success, failure = _synthetic_trajectories()
    pca = ours.fit_pca(success, n_components=8)
    ours.apply_pca(success, pca)
    ours.apply_pca(failure, pca)
    mean, std = ours.compute_normalization_stats(success)
    ours.normalize_trajectories(success, mean, std)
    ours.normalize_trajectories(failure, mean, std)

    torch.manual_seed(3)
    model = OATModel(latent_dim=8)
    calibration = ours.collect_calibration_scores(model, success, device="cpu")
    threshold = ours.conformal_quantile_threshold(calibration, 0.2)

    mine = ours.score_failure_trajectories(
        model, copy.deepcopy(failure), device="cpu", conformal_threshold=threshold
    )

    with vendored_oat("models", "train") as (vendored_models, vendored_train):
        theirs_model = vendored_models.OATModel(latent_dim=8)
        theirs_model.load_state_dict(model.state_dict())
        cal_theirs = vendored_train.collect_calibration_scores(theirs_model, success, device="cpu")
        thr_theirs = vendored_train.conformal_quantile_threshold(cal_theirs, 0.2)
        theirs = vendored_train.score_failure_trajectories(
            theirs_model, copy.deepcopy(failure), device="cpu", conformal_threshold=thr_theirs
        )

    assert threshold == thr_theirs
    assert len(mine) == len(theirs) == len(failure)
    for a, b in zip(mine, theirs):
        assert a["predicted_step"] == b["predicted_step"]
        assert a["topk_detection"] == b["topk_detection"]
        assert a["conformal_detection"] == b["conformal_detection"]
        assert a["score_step_indices"] == b["score_step_indices"]
        assert (a["scores"] == b["scores"]).all()


def test_filter_model_steps_keeps_question_and_agent_turns():
    torch = pytest.importorskip("torch")

    from oat.train import filter_model_steps

    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    step_indices = [-1, 0, 1, 2, 3]
    step_is_model = [True, True, False, True, False]
    kept_hidden, kept_indices = filter_model_steps(hidden, step_indices, step_is_model)
    assert kept_indices == [-1, 0, 2]
    assert torch.equal(kept_hidden, hidden[[0, 1, 3]])


# --------------------------------------------------------------------------
# The command-line pipeline, end to end on the dummy extractor
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
    return env


@pytest.fixture(scope="module")
def mini_train_dir(tmp_path_factory):
    """A handful of successful MCP-Atlas logs — enough to fit PCA and a model."""
    source = VENDORED_OAT / "dataset" / "MCP-atlas" / "Qwen3.5-27B"
    target = tmp_path_factory.mktemp("mini-train")
    kept = 0
    for path in sorted(source.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("errors") == [] and data.get("raw_conversation_history"):
            shutil.copy(path, target / path.name)
            kept += 1
        if kept >= 8:
            break
    assert kept >= 4, "need a few successful training trajectories"
    return target


@pytest.fixture(scope="module")
def mini_corpus(tmp_path_factory):
    """A four-trajectory stand-in corpus, so a report can reach DONE quickly."""
    source = REPO_ROOT / "data" / "ww" / "hand-crafted"
    target = tmp_path_factory.mktemp("mini-data") / "mini"
    target.mkdir()
    for path in sorted(source.glob("*.json"), key=lambda p: int(p.stem))[:4]:
        shutil.copy(path, target / path.name)
    return target


def _predict(output, corpus, train_dir, *extra, seeds="42"):
    cmd = [
        sys.executable, "-m", "oat.predict",
        "--input", str(corpus),
        "--output", str(output),
        "--model-name", "dummy", "--model-path", "dummy", "--device", "cpu",
        "--seeds", seeds, "--latent-dim", "8", "--epochs", "2", "--patience", "1",
        "--log-every", "0", "--train-dir", str(train_dir),
        *extra,
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT, env=_rb_env(), capture_output=True, text=True)


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory, mini_train_dir, mini_corpus):
    pytest.importorskip("torchcde")
    output = tmp_path_factory.mktemp("run") / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    result = _predict(output, mini_corpus, mini_train_dir, seeds="42,43")
    assert result.returncode == 0, result.stdout + result.stderr
    return output, result


def test_cli_writes_one_file_per_trajectory_per_seed(completed_run, mini_corpus):
    output, _ = completed_run
    expected = {p.stem for p in mini_corpus.glob("*.json")}
    for seed in (42, 43):
        method_dir = output / f"oat.s{seed}"
        written = {p.stem for p in method_dir.glob("*.json") if not p.name.startswith("_")}
        assert written == expected


def test_cli_output_has_every_required_key(completed_run):
    output, _ = completed_run
    required = {
        "id", "filename", "question_id", "method", "model", "backend",
        "predicted_agent", "predicted_step", "gold_agent", "gold_step", "raw", "calls",
    }
    for path in (output / "oat.s42").glob("*.json"):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text())
        assert required <= set(doc), required - set(doc)
        assert doc["method"] == "oat"
        assert doc["calls"] == []
        assert isinstance(doc["predicted_step"], int)
        # the agent is the role of the step the model pointed at
        assert isinstance(doc["predicted_agent"], str)
        # step indices are absolute and 0-based, with no shift
        assert 0 <= doc["predicted_step"] < doc["num_steps"]
        assert doc["predicted_step"] in doc["score_step_indices"]
        assert doc["raw"] is not None and json.loads(doc["raw"]) == doc["scores"]
        assert doc["train_seed"] == 42
        assert doc["gt_in_prompt"] is False


def test_cli_predicted_agent_is_the_role_of_the_predicted_step(completed_run, mini_corpus):
    output, _ = completed_run
    for path in (output / "oat.s42").glob("*.json"):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text())
        record = json.loads((mini_corpus / doc["filename"]).read_text())
        assert doc["predicted_agent"] == record["history"][doc["predicted_step"]]["role"]


def test_cli_never_predicts_a_filtered_step(completed_run, mini_corpus):
    """The human's turn is not the agent's mistake; it is never the answer."""
    from oat.serialize import is_model_step

    output, _ = completed_run
    for path in (output / "oat.s42").glob("*.json"):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text())
        record = json.loads((mini_corpus / doc["filename"]).read_text())
        assert is_model_step(record["history"][doc["predicted_step"]])


def test_cli_writes_run_config(completed_run):
    output, _ = completed_run
    run_cfg = json.loads((output / "oat.s42" / "_run.json").read_text())
    for key in ("model", "method", "method_dir", "backend", "gt_in_prompt", "train_seed",
                "conformal_threshold", "layer", "aggregation", "checkpoint"):
        assert key in run_cfg, key
    assert run_cfg["gt_in_prompt"] is False
    assert run_cfg["train_seed"] == 42


def test_cli_caches_states_and_checkpoints_inside_the_output_root(completed_run):
    """States and models live beside the predictions, not under artifacts/."""
    output, _ = completed_run
    assert (output / "_oat-states").is_dir()
    assert list((output / "_oat-states").glob("*.pt"))
    assert (output / "_oat-states" / "_manifest.json").exists()

    train_root = output.parents[2] / "mcp-atlas" / "train" / "dummy"
    assert (train_root / "_oat-states").is_dir()
    assert (train_root / "_oat-ckpt" / "projector.pt").exists()
    assert (train_root / "_oat-ckpt" / "s42" / "model.pt").exists()
    assert (train_root / "_oat-ckpt" / "s43" / "model.pt").exists()


def test_cli_resumes_and_overwrites(completed_run, mini_corpus, mini_train_dir):
    output, _ = completed_run
    again = _predict(output, mini_corpus, mini_train_dir, seeds="42")
    assert again.returncode == 0, again.stdout + again.stderr
    assert "skip (complete)" in again.stdout

    victim = sorted((output / "oat.s42").glob("[0-9]*.json"))[0]
    before = victim.read_text()
    victim.unlink()
    resumed = _predict(output, mini_corpus, mini_train_dir, seeds="42")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "1 to run" in resumed.stdout
    assert json.loads(victim.read_text())["predicted_step"] == json.loads(before)["predicted_step"]


def test_cli_slices_with_start_and_end_idx(tmp_path, mini_corpus, mini_train_dir):
    pytest.importorskip("torchcde")
    output = tmp_path / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    result = _predict(output, mini_corpus, mini_train_dir, "--end_idx", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    written = [p for p in (output / "oat.s42").glob("*.json") if not p.name.startswith("_")]
    assert len(written) == 2


def test_gt_setting_changes_the_states_but_not_the_model(tmp_path, mini_corpus, mini_train_dir):
    """--gt with re-extracts the test side and reuses the trained model."""
    pytest.importorskip("torchcde")
    nogt = tmp_path / "outputs-rb-nogt" / "mini" / "mini" / "dummy"
    withgt = tmp_path / "outputs-rb-gt" / "mini" / "mini" / "dummy"
    assert _predict(nogt, mini_corpus, mini_train_dir, "--gt", "without").returncode == 0
    result = _predict(withgt, mini_corpus, mini_train_dir, "--gt", "with")
    assert result.returncode == 0, result.stdout + result.stderr

    # one training root, shared: the with-GT run trained nothing new
    train_root = tmp_path / "outputs-rb-nogt" / "mcp-atlas" / "train" / "dummy"
    assert (train_root / "_oat-ckpt" / "s42" / "model.pt").exists()
    assert not (tmp_path / "outputs-rb-gt" / "mcp-atlas").exists()

    doc = json.loads(next(p for p in (withgt / "oat.s42").glob("[0-9]*.json")).read_text())
    assert doc["gt_in_prompt"] is True


def test_sweep_dry_run_resolves_to_the_nogt_root():
    result = subprocess.run(
        [sys.executable, "-m", "oat.sweep",
         "--config", "baselines-rp/oat/configs/ww.yaml", "--dry-run"],
        cwd=REPO_ROOT, env=_rb_env(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "outputs-rb-nogt/ww/hand-crafted/qwen3.5-9b" in result.stdout
    assert "outputs-rb-gt" not in result.stdout

    with_gt = subprocess.run(
        [sys.executable, "-m", "oat.sweep",
         "--config", "baselines-rp/oat/configs/ww.yaml", "--gt", "with", "--dry-run"],
        cwd=REPO_ROOT, env=_rb_env(), capture_output=True, text=True,
    )
    assert with_gt.returncode == 0, with_gt.stdout + with_gt.stderr
    assert "outputs-rb-gt/ww/hand-crafted/qwen3.5-9b" in with_gt.stdout


def test_shipped_configs_parse_and_agree_with_the_corpus():
    import yaml

    cfg_dir = REPO_ROOT / "baselines-rp" / "oat" / "configs"
    for name in ("ww", "correct-error", "traceelephant"):
        cfg = yaml.safe_load((cfg_dir / f"{name}.yaml").read_text())
        assert cfg["gt"] == "without"
        assert cfg["outputs_root"] == f"outputs-rb-gt/{name}"
        assert cfg["seeds"] == [42, 43, 44, 45, 46]
        for subset in cfg["subsets"]:
            assert (REPO_ROOT / cfg["data_dir"] / subset).is_dir(), f"{name}/{subset}"
        for model, spec in cfg["model_specs"].items():
            assert "model_path" in spec, model

        report = yaml.safe_load((cfg_dir / f"report_{name}.yaml").read_text())
        assert report["methods"] == [f"oat.s{s}" for s in cfg["seeds"]]
        assert report["pred_root"] == cfg["outputs_root"]
        assert report["gt"] == "without"
        assert sorted(report["subsets"]) == sorted(cfg["subsets"])


def test_report_reads_the_rb_root_and_reports_done(tmp_path, completed_run, mini_corpus):
    """The shared report scores these outputs unchanged, step@1 and agent@1."""
    import yaml

    output, _ = completed_run
    pred_root = output.parents[1]          # <root>/mini
    data_root = mini_corpus.parent

    cfg = {
        "models": ["dummy"], "subsets": ["mini"], "methods": ["oat.s42", "oat.s43"],
        "data_dir": str(data_root), "pred_root": str(pred_root),
        "out_root": str(tmp_path / "reports"),
        "splits": {"train": 0.3, "val": 0.2, "test": 0.5}, "seeds": [1, 2, 3],
        "gt_in_prompt": False,
    }
    cfg_path = tmp_path / "report_mini.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    check = subprocess.run(
        [sys.executable, "-m", "oat.report", "--config", str(cfg_path), "--check-only"],
        cwd=REPO_ROOT, env=_rb_env(), capture_output=True, text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    assert "DONE" in check.stdout and "MISSING" not in check.stdout

    full = subprocess.run(
        [sys.executable, "-m", "oat.report", "--config", str(cfg_path)],
        cwd=REPO_ROOT, env=_rb_env(), capture_output=True, text=True,
    )
    assert full.returncode == 0, full.stdout + full.stderr
    summary = tmp_path / "reports" / "summary_mean_over_seeds.tsv"
    assert summary.exists()
    text = summary.read_text()
    assert "oat.s42" in text and "oat.s43" in text
    # both accuracies are reported for a representation-based baseline
    assert "step" in text.lower() and "agent" in text.lower()


def test_rb_metrics_matches_vendored_evaluate(completed_run):
    """The paper-style set metrics reproduce vendored/OAT/evaluate.py."""
    pytest.importorskip("sklearn")

    from rb_shared.rb_metrics import compute_metrics, load_method_dir

    output, _ = completed_run
    docs = load_method_dir(output / "oat.s42")
    ours = compute_metrics(docs)

    # the vendored metrics take the same numbers under their own key names
    vendored_input = [
        {
            "predicted_step": d["predicted_step"],
            "scores": d["scores"],
            "score_step_indices": d["score_step_indices"],
            "topk_detection": d["topk_steps"],
            "conformal_detection": d["conformal_steps"],
            "gt_error_steps": [int(d["gold_step"])],
        }
        for d in docs
    ]
    with vendored_oat("evaluate") as (vendored_eval,):
        theirs = vendored_eval.compute_metrics(vendored_input)

    for key, value in theirs.items():
        assert ours[key] == pytest.approx(value), key


def test_rb_metrics_cli_writes_a_table(tmp_path, completed_run):
    output, _ = completed_run
    out_tsv = tmp_path / "rb_metrics.tsv"
    result = subprocess.run(
        [sys.executable, "-m", "rb_shared.rb_metrics",
         "--pred-root", str(output.parents[1]), "--methods", "oat.s42", "--out", str(out_tsv)],
        cwd=REPO_ROOT, env=_rb_env(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    text = out_tsv.read_text()
    assert "hit_rate" in text and "auroc" in text and "oat.s42" in text


def test_last_layer_shortcut_is_identical_to_slicing_the_stack():
    """Skipping the full layer stack must not change a single number."""
    torch = pytest.importorskip("torch")

    from oat.extract import dummy_extractor, extract_hidden_states_for_trajectory
    from oat.serialize import build_input_text, serialize_trajectory

    ex = dummy_extractor()
    record = _ww_records(1)[0]
    step_text, bounds, _a, _f = serialize_trajectory(record["history"])
    text, all_bounds = build_input_text(record.get("question", ""), step_text, bounds)

    shortcut = extract_hidden_states_for_trajectory(text, all_bounds, ex.model, ex.tokenizer, layers=(-1,))
    full = extract_hidden_states_for_trajectory(text, all_bounds, ex.model, ex.tokenizer, layers=None)
    assert torch.equal(shortcut[:, 0, :], full[:, -1, :])
