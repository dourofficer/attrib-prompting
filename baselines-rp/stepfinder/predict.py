"""Run StepFinder over one subset with one encoder: encode, train, score.

Four stages, each resumable on file existence, mirroring ``oat/predict.py``:

===============  ==========================================================
``feats-train``  embed the trajectories the model will learn from
``train``        fit one model per seed and cache it
``feats-test``   embed the trajectories to be scored
``score``        write one prediction file per trajectory per seed
===============  ==========================================================

``--protocol`` decides only where the training features come from — the
vendored regenerated corpus, or the corpus under test. Everything after that
is the same code, which is why the two families share a cache, a schema and a
resume rule. See ``protocol.py`` for what each family means and
``IMPLEMENTATION.md`` for why both are here.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from baselines.prompting.predict import load_records
from baselines.shared.common import nogt_root
from baselines.shared.runner import OutputWriter
from rb_shared.cache import StateCache
from rb_shared.detect import topk_detection
from stepfinder import protocol as proto
from stepfinder.encode import dummy_encoder, load_encoder
from stepfinder.features import (
    AGENT_FIELD_REPO, build_features, derive_gt_features,
)
from stepfinder.reduce import (
    apply_reducer, fit_reducer, load_reducer, save_reducer,
)
from stepfinder.train import (
    BATCH_SIZE, EPOCHS, PATIENCE, PRESETS, SEEDS, build_model, fit,
    holdout_by_question, score_trajectory,
)

METHOD = "stepfinder"
STAGES = ("feats-train", "train", "feats-test", "score")
FEATS_DIRNAME = "_sf-feats"
FEATS_WIDE_DIRNAME = "_sf-featsw"   # full encoder width, the input PCA is fitted on
PCA_DIRNAME = "_sf-pca"
CKPT_DIRNAME = "_sf-ckpt"
REGEN_DATASET = "stepfinder-regen"
DEFAULT_TOP_K = 3


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

def nogt_output(output: str) -> Path:
    """The without-GT twin of an output directory, at any depth.

    ``baselines.shared.common.nogt_root`` maps the *first* path component,
    which is all the sweeps need because they build roots relative to the repo.
    The checkpoint layout here has to work on an absolute path too, so the
    ``outputs…`` component is found rather than assumed. Already-``-nogt``
    paths come back unchanged.
    """
    parts = Path(output).parts
    for i, part in enumerate(parts):
        if part == "outputs" or part.startswith("outputs-"):
            head = part if part.endswith("-nogt") else nogt_root(part)
            return Path(*parts[:i], head, *parts[i + 1:])
    raise SystemExit(
        f"cannot derive the without-GT twin of {output!r}: the path has no "
        "'outputs…' component to mirror."
    )


def regen_root(output: str, model_name: str, train_set: str) -> Path:
    """Where a vendored-trained model's features and weights live.

    Always in the without-GT tree, and outside any dataset: the regenerated
    corpus has no answer to inject, so both GT settings train on identical
    tensors, and one checkpoint serves every subset that maps to this training
    set. Same shape as OAT's ``mcp-atlas/train/<model>/``
    (``oat/predict.py:80-95``), with the training-set name in the subset slot
    because StepFinder ships two of them.
    """
    parts = Path(output).parts
    for i, part in enumerate(parts):
        if part == "outputs" or part.startswith("outputs-"):
            head = part if part.endswith("-nogt") else nogt_root(part)
            return Path(*parts[:i], head, REGEN_DATASET, train_set, model_name)
    raise SystemExit(
        f"cannot place the training root beside {output!r}: the path has no "
        "'outputs…' component to mirror. Pass --train-root explicitly."
    )


def ckpt_dir(root: Path, preset: str, selection: str, seed: int, letter: str) -> Path:
    """``_sf-ckpt/<preset>/<selection>/<letter><seed>/``.

    ``preset`` keeps two subsets that share a training set but not its
    hyperparameters from overwriting each other. ``selection`` keeps a
    ``--model-selection vendored`` parity checkpoint, which was fitted while
    watching the test set, from ever being mistaken for a real one.
    """
    return root / CKPT_DIRNAME / preset / selection / f"{letter}{seed}"


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def _slice_of_wide(payload: dict, content_dim: int, agent_dim: int) -> dict:
    """The vendored prefix slice, taken from an already-encoded wide payload.

    The slice is literally the first coordinates of the full embedding, so a
    wide cache makes every sliced feature free — no forward pass, identical
    bits to encoding at the sliced width directly.
    """
    return dict(
        payload,
        content_features=payload["content_features"][:, :content_dim].copy(),
        agent_features=payload["agent_features"][:, :agent_dim].copy(),
    )


def _encode_into(cache: StateCache, records: list, encoder, *, agent_field: str,
                 agent_normalize: str, include_gt: bool, nogt_cache: StateCache | None,
                 label: str, log_every: int = 200,
                 content_dim=None, agent_dim=None,
                 wide_fallback: StateCache | None = None) -> tuple[int, int]:
    """Embed every record not already cached. Returns (written, skipped).

    ``content_dim``/``agent_dim`` of ``None`` keep the encoder's full width.
    ``wide_fallback`` short-circuits a sliced encode when the same record is
    already in a wide cache: the slice is derived on CPU instead of re-run
    through the encoder.
    """
    from stepfinder.encode import AGENT_DIM, CONTENT_DIM
    written = skipped = 0
    for i, record in enumerate(records):
        rid = str(record["id"])
        if cache.has(rid):
            skipped += 1
            continue
        if wide_fallback is not None and wide_fallback.has(rid):
            wide = wide_fallback.load(rid)
            payload = _slice_of_wide(wide, content_dim or CONTENT_DIM, agent_dim or AGENT_DIM)
            cache.save(rid, payload)
            written += 1
            continue
        base = nogt_cache.load(rid) if (include_gt and nogt_cache is not None) else None
        if base is not None:
            # Only step 0's content changes between GT settings; reuse the rest.
            payload = derive_gt_features(base, record, encoder, content_dim=content_dim)
        else:
            payload = build_features(
                record, encoder, agent_field=agent_field,
                agent_normalize=agent_normalize, include_gt=include_gt,
                content_dim=content_dim, agent_dim=agent_dim,
            )
        cache.save(rid, payload)
        written += 1
        if log_every and written and written % log_every == 0:
            print(f"  {label}: {written} encoded, {skipped} cached", flush=True)
    print(f"  {label}: {written} encoded, {skipped} already cached", flush=True)
    return written, skipped


def _load_cached(cache: StateCache, ids) -> list:
    out = []
    for rid in ids:
        payload = cache.load(str(rid))
        if payload is not None and int(payload.get("num_steps", 0)) > 0:
            out.append(payload)
    return out


# --------------------------------------------------------------------------
# Prediction document
# --------------------------------------------------------------------------

def _build_doc(record, scored, job, args, *, model_name, backend, include_gt,
               overlap, fit_meta, elapsed_ms, top_k) -> dict:
    history = record.get("history") or []
    predicted_step = None if scored is None else int(scored["predicted_step"])
    predicted_agent = None
    if predicted_step is not None and 0 <= predicted_step < len(history):
        predicted_agent = history[predicted_step].get(AGENT_FIELD_REPO)

    scores = None if scored is None else [round(float(s), 8) for s in scored["scores"]]
    ts = job.training_set
    return {
        # ---- the shared schema every baseline in this repo writes ----
        "id": str(record["id"]),
        "filename": record.get("filename"),
        "question_id": record.get("question_id"),
        "method": METHOD,
        "model": model_name,
        "backend": backend,
        "gt_in_prompt": bool(include_gt),
        "predicted_agent": predicted_agent,
        "predicted_step": predicted_step,
        "gold_agent": record.get("gold_agent"),
        "gold_step": record.get("gold_step"),
        # `raw` is what a prompting baseline would have parsed. StepFinder
        # parses nothing; its per-step distribution is the equivalent evidence.
        "raw": None if scores is None else json.dumps(scores),
        "calls": [],
        # ---- shared with OAT, so rb_metrics reads both the same way ----
        "method_dir": job.method_dir,
        "extractor": model_name,
        "scores": scores,
        "scores_logit": None if scored is None else [round(float(s), 6) for s in scored["scores_logit"]],
        "score_step_indices": None if scored is None else list(scored["step_indices"]),
        "topk_steps": None if scored is None else topk_detection(
            np.asarray(scored["scores"], dtype=float), scored["step_indices"], top_k),
        "topk_k": top_k,
        # StepFinder has no calibration set, no nonconformity score and no
        # threshold. Synthesising a set here would read as a coverage
        # guarantee and carry none, so the conformal columns stay empty.
        "conformal_steps": None,
        "conformal_threshold": None,
        "conformal_alpha": None,
        "num_scored_steps": 0 if scored is None else len(scored["scores"]),
        "num_steps": len(history),
        # StepFinder scores every turn — it has no model-step filter — so the
        # only unreachable gold is one that points past the end of the history.
        "gold_on_filtered_step": _gold_out_of_range(record, len(history)),
        "inference_time_ms": round(elapsed_ms, 3),
        # ---- StepFinder's own ----
        "protocol": ts.protocol,
        "train_seed": job.seed,
        "train_set": ts.name,
        "n_train_trajectories": len(ts.records),
        "train_task_overlap": overlap,
        "preset": ts.preset,
        "hparams": ts.hparams.as_dict(),
        "agent_normalize": args.agent_normalize,
        "eval_batch_size": 1,
        "position_bias_denominator": "trajectory",
        **fit_meta,
    }


def _gold_out_of_range(record: dict, num_steps: int) -> bool:
    try:
        return not 0 <= int(record.get("gold_step")) < num_steps
    except (TypeError, ValueError):
        return True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="stepfinder.predict", description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True,
                   help="Model-level dir; predictions land in <output>/stepfinder.<seed>/.")
    p.add_argument("--model-name", required=True, help="Short encoder label; the <model> path component.")
    p.add_argument("--model-path", required=True, help="Local checkpoint dir, or the literal 'dummy'.")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--dtype", default="fp16", choices=("bf16", "fp16", "fp32"))
    p.add_argument("--device", default=None)
    p.add_argument("--train-device", default=None,
                   help="Device the small network trains on (default: --device).")
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--no-memo", action="store_true",
                   help="Encode every string separately, even repeats.")
    p.add_argument("--agent-normalize", default="standardize", choices=("raw", "standardize"))
    p.add_argument("--gt", default="without", choices=("with", "without"))
    p.add_argument("--protocol", default="regen",
                   help="Comma-separated: regen, in-corpus.")
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                   help="Training seeds for the vendored protocol.")
    p.add_argument("--eval-seeds", default=None,
                   help="Split seeds for the in-corpus protocol; must match the report config.")
    p.add_argument("--splits", default="0.3,0.2,0.5")
    p.add_argument("--method-dir-prefix", default=METHOD)
    p.add_argument("--train-set", default="auto", help="auto | algorithm-generated | hand-crafted")
    p.add_argument("--preset", default="auto", help="auto | alg | hc")
    p.add_argument("--train-data-root", default=proto.DEFAULT_TRAIN_DATA_ROOT)
    p.add_argument("--train-root", default=None)
    p.add_argument("--model-selection", default="val", choices=("val", "vendored", "test"),
                   help="val: early-stop on a held-out slice of the training data. "
                        "test: the paper's own rule — checkpoint on the scored corpus's "
                        "accuracy — kept as a labeled variant because the paper's "
                        "protocol is test-selected; its predictions land in a "
                        "'-tsel' method dir and can never mix with val-selected ones. "
                        "vendored: the same rule as a one-off parity check that "
                        "refuses to write predictions.")
    p.add_argument("--reduce", default="slice", choices=("slice", "pca"),
                   help="slice: the vendored prefix slice (first 128/32 coordinates). "
                        "pca: project the full-width embedding onto the top principal "
                        "directions of the training features (uncentered SVD; see "
                        "reduce.py). Predictions land in a '-pca' method dir.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-min-tasks", type=int, default=4,
                   help="Minimum distinct questions in the holdout; below this, "
                        "early stopping is skipped and the full budget is trained.")
    p.add_argument("--val-min-size", type=int, default=20,
                   help="Minimum holdout trajectories. Accuracy on fewer is noise, "
                        "and selecting on noise is worse than not selecting at all.")
    for name, default in (("lr", None), ("weight-decay", None), ("alpha", None),
                          ("beta", None), ("gamma", None), ("lambda-temporal", None),
                          ("dropout", None)):
        p.add_argument(f"--{name}", type=float, default=default)
    p.add_argument("--scales", default=None, help="Comma-separated, e.g. 1,2")
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--min-train-steps", type=int, default=0,
                   help="Raise --epochs until the run gets at least this many optimizer "
                        "steps. The vendored 50 epochs was set for a 1,564-2,604 "
                        "trajectory corpus; on a small partition it is a fraction of the "
                        "updates the learning rate was tuned for. 0 disables.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def _ensure_reducer(ts, args, fit_now: bool = False) -> dict:
    """The (encoder, training-set) reducer: load it, or fit and save it.

    Fitted once, on the training corpus's full-width features, and shared by
    every subset and seed that trains on this corpus — the test side never
    touches the basis. ``fit_now`` is set by ``feats-train`` so the fit happens
    once, right after encoding, rather than racing in parallel train stages.
    """
    rdir = ts.feats_root / PCA_DIRNAME
    reducer = load_reducer(rdir)
    if reducer is not None:
        return reducer
    wide_cache = StateCache(ts.feats_root / FEATS_WIDE_DIRNAME)
    payloads = _load_cached(wide_cache, ts.ids)
    if len(payloads) < len(ts.ids):
        raise SystemExit(
            f"cannot fit the PCA reducer for {ts.name}: only {len(payloads)} of "
            f"{len(ts.ids)} training trajectories have wide features. Run "
            "--reduce pca --stages feats-train first."
        )
    print(f"  fitting PCA reducer for {ts.name} on {len(payloads)} trajectories …", flush=True)
    reducer = fit_reducer(payloads)
    save_reducer(reducer, rdir)
    print(f"  saved {rdir / 'reducer.pt'}  (content var kept "
          f"{reducer['content_var_kept']:.3f}, agent {reducer['agent_var_kept']:.3f})", flush=True)
    return reducer


def _ckpt_home(ts, args, output: str):
    """Where this job's checkpoint lives: (root, selection-component, letter).

    ``val``-selected regen checkpoints sit with the training corpus and are
    shared by every subset and GT setting — nothing about them depends on the
    scored data. A ``test``-selected checkpoint was picked by watching Who&When
    test accuracy — the paper's own selection rule and its only benchmark — so
    it lives with the WW subset that matches its training corpus
    (Algorithm-Generated -> ``ww/algorithm-generated``, Hand-Crafted ->
    ``ww/hand-crafted``), in the current GT tree, and every other subset that
    maps to that corpus reuses it. For WW the selection is the paper's
    test-leaked rule; for CE and TE the selection set is disjoint from the
    reported data. The selection component also carries the reduction, so
    slice- and PCA-input models can never load each other's weights.
    """
    sel = args.model_selection + ("-pca" if args.reduce == "pca" else "")
    if ts.protocol != "regen":
        return nogt_output(output), sel, "e"
    if args.model_selection == "test":
        return _ww_sibling(output, ts.name, args.model_name), sel, "s"
    return ts.feats_root, sel, "s"


def _ww_sibling(output: str, ww_subset: str, model_name: str) -> Path:
    """``<outputs tree>/ww/<ww_subset>/<model>`` beside any output dir."""
    parts = Path(output).parts
    for i, part in enumerate(parts):
        if part == "outputs" or part.startswith("outputs-"):
            return Path(*parts[:i], part, "ww", ww_subset, model_name)
    raise SystemExit(
        f"cannot place the test-selection checkpoint beside {output!r}: the "
        "path has no 'outputs…' component to mirror."
    )


def _hparams_for(args, preset: str):
    hp = PRESETS[preset]
    overrides = {}
    for flag, field_name in (("lr", "lr"), ("weight_decay", "weight_decay"), ("alpha", "alpha"),
                             ("beta", "beta"), ("gamma", "gamma"),
                             ("lambda_temporal", "lambda_temporal"), ("dropout", "dropout"),
                             ("hidden_dim", "hidden_dim"), ("num_heads", "num_heads"),
                             ("head_dim", "head_dim")):
        value = getattr(args, flag, None)
        if value is not None:
            overrides[field_name] = value
    if args.scales:
        overrides["scales"] = tuple(int(x) for x in args.scales.split(",") if x.strip())
    if not overrides:
        return hp, preset
    from dataclasses import replace
    tuned = replace(hp, **overrides)
    return tuned, f"custom-{tuned.fingerprint()}"


def main(argv=None) -> None:
    args = parse_args(argv)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    if args.model_selection == "vendored" and "score" in stages:
        raise SystemExit(
            "--model-selection vendored fits while watching the corpus being scored, "
            "which is the vendored rule (main.py:231-251) and not a result this repo "
            "can publish. Run it with --stages feats-train,train,feats-test to print "
            "the parity numbers; it will not write predictions."
        )

    protocols = [p.strip() for p in args.protocol.split(",") if p.strip()]
    if args.reduce == "pca" and "in-corpus" in protocols:
        raise SystemExit("--reduce pca is implemented for the regen protocol only: "
                         "the reducer is fitted on the vendored training corpus.")
    # A non-default reduction or selection gets its own method-dir family, so
    # `stepfinder.s42` always means slice + val and nothing else can pose as it.
    if args.method_dir_prefix == METHOD:
        if args.reduce == "pca":
            args.method_dir_prefix += "-pca"
        if args.model_selection == "test":
            args.method_dir_prefix += "-tsel"
    wide = args.reduce == "pca"
    feats_dirname = FEATS_WIDE_DIRNAME if wide else FEATS_DIRNAME
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_device = args.train_device or device
    backend = "dummy" if args.model_path == "dummy" else "hf"
    include_gt = args.gt == "with"
    splits = dict(zip(("train", "val", "test"),
                      (float(x) for x in args.splits.split(","))))
    subset = Path(args.input).name

    records = load_records(args.input)
    sliced = records[args.start_idx:args.end_idx]
    print(f"[stepfinder] {subset} | {args.model_name} | gt={args.gt} | "
          f"{len(sliced)}/{len(records)} trajectories | protocols={protocols}", flush=True)

    train_set_name = proto.resolve_train_set(subset, args.train_set)
    preset_name = proto.resolve_preset(train_set_name, args.preset)
    hparams, preset_key = _hparams_for(args, preset_name)

    regen_dir = Path(args.train_root) if args.train_root else regen_root(
        args.output, args.model_name, train_set_name)
    test_feats = StateCache(Path(args.output) / feats_dirname,
                            overwrite=args.overwrite and "feats-test" in stages)
    nogt_feats = (StateCache(nogt_output(args.output) / feats_dirname)
                  if include_gt else test_feats)
    # When slicing, a wide cache built by an earlier --reduce pca run makes
    # every missing sliced feature a CPU copy instead of a forward pass.
    wide_test = None if wide else StateCache(Path(args.output) / FEATS_WIDE_DIRNAME)
    from stepfinder.encode import AGENT_DIM, CONTENT_DIM
    content_dim = None if wide else CONTENT_DIM
    agent_dim = None if wide else AGENT_DIM

    encoder = None

    def get_encoder():
        nonlocal encoder
        if encoder is None:
            if args.model_path == "dummy":
                encoder = dummy_encoder()
            else:
                encoder = load_encoder(args.model_name, {
                    "path": args.model_path,
                    "tokenizer": args.tokenizer,
                    "dtype": args.dtype,
                    "device": device,
                    "attn_implementation": args.attn_implementation,
                })
            encoder.memo = not args.no_memo
        return encoder

    # ---- build the jobs, one list per protocol ----------------------------
    all_jobs: list = []
    for protocol in protocols:
        if protocol == "regen":
            ts = proto.regen_training_set(args.train_data_root, train_set_name,
                                          preset_key, hparams, regen_dir)
            seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
            all_jobs += proto.jobs_for("regen", method_prefix=args.method_dir_prefix,
                                       seeds=seeds, training_set_factory=lambda _s: ts)
        elif protocol == "in-corpus":
            if not args.eval_seeds:
                raise SystemExit("--protocol in-corpus needs --eval-seeds (the report's split seeds)")
            seeds = [int(s) for s in args.eval_seeds.split(",") if s.strip()]
            all_jobs += proto.jobs_for(
                "in-corpus", method_prefix=args.method_dir_prefix, seeds=seeds,
                training_set_factory=lambda s: proto.in_corpus_training_set(
                    args.input, records, splits, s, preset_key, hparams),
                data_dir=args.input, splits=splits,
            )
        else:
            raise SystemExit(f"protocol must be 'regen' or 'in-corpus', got {protocol!r}")

    # ---- stage: feats-train ----------------------------------------------
    if "feats-train" in stages:
        for ts in _unique_training_sets(all_jobs):
            if ts.protocol == "regen":
                cache = StateCache(ts.feats_root / feats_dirname,
                                   overwrite=args.overwrite)
                wide_train = None if wide else StateCache(ts.feats_root / FEATS_WIDE_DIRNAME)
                _encode_into(cache, ts.records, get_encoder(),
                             agent_field=ts.agent_field,
                             agent_normalize=args.agent_normalize,
                             include_gt=False, nogt_cache=None,
                             label=f"feats-train[{ts.name}]",
                             content_dim=content_dim, agent_dim=agent_dim,
                             wide_fallback=wide_train)
                cache.write_manifest({
                    "corpus": ts.source, "encoder": args.model_name,
                    "model_path": args.model_path, "gt_in_prompt": False,
                    "agent_field": ts.agent_field, "agent_normalize": args.agent_normalize,
                    "n_trajectories": len(ts.records), "reduce_width": "wide" if wide else "sliced",
                })
                if wide:
                    _ensure_reducer(ts, args, fit_now=True)
            else:
                # In-corpus trains on the without-GT features of the corpus
                # itself, so the model is shared across GT settings exactly as
                # the vendored-trained one is. Those are the same tensors
                # `feats-test` writes; encode any the cache is still missing.
                missing = [r for r in ts.records if not nogt_feats.has(str(r["id"]))]
                if missing:
                    _encode_into(nogt_feats, missing, get_encoder(),
                                 agent_field=ts.agent_field,
                                 agent_normalize=args.agent_normalize,
                                 include_gt=False, nogt_cache=None,
                                 label=f"feats-train[{ts.name}]",
                                 content_dim=content_dim, agent_dim=agent_dim)

    # ---- stage: feats-test (as a run-once function: selection on the scored
    # corpus needs these features during train, which precedes feats-test in
    # the stage order) ------------------------------------------------------
    feats_test_ran = False

    def run_feats_test():
        nonlocal feats_test_ran
        if feats_test_ran:
            return
        feats_test_ran = True
        _encode_into(test_feats, sliced, get_encoder(),
                     agent_field=AGENT_FIELD_REPO,
                     agent_normalize=args.agent_normalize,
                     include_gt=include_gt,
                     nogt_cache=nogt_feats if include_gt else None,
                     label=f"feats-test[{subset}]",
                     content_dim=content_dim, agent_dim=agent_dim,
                     wide_fallback=wide_test)
        test_feats.write_manifest({
            "corpus": args.input, "encoder": args.model_name,
            "model_path": args.model_path, "gt_in_prompt": include_gt,
            "agent_field": AGENT_FIELD_REPO, "agent_normalize": args.agent_normalize,
            "n_trajectories": len(sliced),
        })

    # ---- stage: feats-test (defined as a run-once function: test-selection
    # needs the scored corpus's features during train, which comes first in
    # the stage order) ------------------------------------------------------
    feats_test_ran = False

    def run_feats_test():
        nonlocal feats_test_ran
        if feats_test_ran:
            return
        feats_test_ran = True
        _encode_into(test_feats, sliced, get_encoder(),
                     agent_field=AGENT_FIELD_REPO,
                     agent_normalize=args.agent_normalize,
                     include_gt=include_gt,
                     nogt_cache=nogt_feats if include_gt else None,
                     label=f"feats-test[{subset}]",
                     content_dim=content_dim, agent_dim=agent_dim,
                     wide_fallback=wide_test)
        test_feats.write_manifest({
            "corpus": args.input, "encoder": args.model_name,
            "model_path": args.model_path, "gt_in_prompt": include_gt,
            "agent_field": AGENT_FIELD_REPO, "agent_normalize": args.agent_normalize,
            "n_trajectories": len(sliced),
        })

    # ---- stage: train ------------------------------------------------------
    fit_meta_by_job: dict = {}
    if "train" in stages:
        if args.model_selection in ("vendored", "test") and "feats-test" in stages:
            run_feats_test()
        for job in all_jobs:
            ts = job.training_set
            root, sel, letter = _ckpt_home(ts, args, args.output)
            cdir = ckpt_dir(root, ts.preset, sel, job.seed, letter)
            ckpt_file = cdir / "model.pt"
            if ckpt_file.exists() and not args.overwrite:
                print(f"  skip (trained): {ckpt_file}", flush=True)
                continue
            if args.model_selection == "test" and ts.protocol == "regen":
                inp = Path(args.input)
                if not (inp.name == ts.name and inp.parent.name == "ww"):
                    raise SystemExit(
                        f"no test-selected checkpoint at {ckpt_file}. Test-selection "
                        f"watches Who&When: train it by running data/ww/{ts.name} "
                        "with --model-selection test first; this subset then "
                        "reuses that checkpoint at score time."
                    )

            feats_cache = (StateCache(ts.feats_root / feats_dirname)
                           if ts.protocol == "regen" else nogt_feats)
            payloads = [p for p in _load_cached(feats_cache, ts.ids) if p.get("trainable", True)]
            if not payloads:
                raise SystemExit(
                    f"no cached training features for {ts.name}; run --stages feats-train first"
                )
            reducer = _ensure_reducer(ts, args) if wide else None
            if reducer is not None:
                payloads = [apply_reducer(p, reducer) for p in payloads]

            if args.model_selection in ("vendored", "test"):
                # The paper's rule: select on the corpus being scored.
                val = _load_cached(test_feats, [r["id"] for r in sliced])
                if reducer is not None:
                    val = [apply_reducer(p, reducer) for p in val]
                if args.model_selection == "test" and not val:
                    raise SystemExit(
                        "--model-selection test needs the scored corpus's features; "
                        "run --stages feats-test before train."
                    )
                train_part = payloads
            else:
                train_part, val = holdout_by_question(payloads, args.val_frac, job.seed)
                n_tasks = len({str(x.get("question", "")) for x in val})
                if len(val) < args.val_min_size or n_tasks < args.val_min_tasks:
                    # Too small to select on. Train the full budget instead —
                    # a holdout of five picks epoch 1 about as often as the
                    # best one, and hands back an untrained network.
                    train_part, val = payloads, []

            epochs = args.epochs
            if args.min_train_steps > 0 and train_part:
                per_epoch = max(1, -(-len(train_part) // args.batch_size))
                epochs = max(epochs, -(-args.min_train_steps // per_epoch))
            print(f"  training {job.method_dir} on {len(train_part)} trajectories "
                  f"({len(val)} held out, {epochs} epochs) …", flush=True)
            # Test-selection deploys at batch 1, so it must select at batch 1;
            # `vendored` keeps the batched evaluation it exists to reproduce,
            # and `val` keeps the batched evaluation its published checkpoints
            # were selected with.
            result = fit(train_part, val, ts.hparams, seed=job.seed, epochs=epochs,
                         batch_size=args.batch_size, patience=args.patience,
                         device=train_device, model_selection=args.model_selection,
                         eval_batch_size=1 if args.model_selection == "test" else None,
                         log_every=args.log_every)
            cdir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": result.state_dict,
                "hparams": ts.hparams.as_dict(),
                "preset": ts.preset,
                "seed": job.seed,
                "protocol": ts.protocol,
                "train_set": ts.name,
                "model_selection": result.model_selection,
                "best_val_acc": result.best_acc,
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "n_train": result.n_train,
                "n_val": result.n_val,
                "val_n_tasks": result.val_n_tasks,
                "history": result.history,
            }, ckpt_file)
            print(f"  saved {ckpt_file}  (best val acc {result.best_acc:.4f} "
                  f"@ epoch {result.best_epoch}, {result.model_selection})", flush=True)

    # ---- stage: feats-test -------------------------------------------------
    if "feats-test" in stages:
        run_feats_test()

    if "score" not in stages:
        return

    # ---- stage: score ------------------------------------------------------
    score_reducers: dict = {}
    for job in all_jobs:
        ts = job.training_set
        root, sel, letter = _ckpt_home(ts, args, args.output)
        ckpt_file = ckpt_dir(root, ts.preset, sel, job.seed, letter) / "model.pt"
        if not ckpt_file.exists():
            raise SystemExit(f"no checkpoint at {ckpt_file}; run --stages train first")
        reducer = None
        if wide:
            key = (ts.protocol, ts.name)
            if key not in score_reducers:
                score_reducers[key] = _ensure_reducer(ts, args)
            reducer = score_reducers[key]

        todo = [r for r in sliced
                if job.scored_ids is None or str(r["id"]) in job.scored_ids]
        writer = OutputWriter(Path(args.output) / job.method_dir, overwrite=args.overwrite)
        done = writer.done_ids()
        remaining = [r for r in todo if str(r["id"]) not in done]
        print(f"  {job.method_dir}: {len(done)} done, {len(remaining)} to run", flush=True)
        if not remaining:
            print(f"  skip (complete): {writer.dir}", flush=True)
            continue

        blob = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        model = build_model(ts.hparams)
        model.load_state_dict(blob["state_dict"])
        dev = torch.device(train_device if (train_device != "cuda" or torch.cuda.is_available()) else "cpu")
        model.to(dev)

        fit_meta = {
            "model_selection": blob.get("model_selection"),
            "reduce": args.reduce,
            "pca_content_var_kept": None if reducer is None else reducer["content_var_kept"],
            "pca_agent_var_kept": None if reducer is None else reducer["agent_var_kept"],
            "pca_dim_in": None if reducer is None else reducer["content_dim_in"],
            "best_val_acc": blob.get("best_val_acc"),
            "best_epoch": blob.get("best_epoch"),
            "val_n_tasks": blob.get("val_n_tasks"),
            "early_stop_rule": blob.get("model_selection"),
            "checkpoint": str(ckpt_file),
        }
        n_overlap = sum(1 for r in todo if proto.train_task_overlap(r, ts))
        writer.write_run_config({
            "model": args.model_name, "model_arg": args.model_path,
            "method": METHOD, "method_dir": job.method_dir, "subset": subset,
            "backend": backend, "gt_in_prompt": include_gt,
            "protocol": ts.protocol, "train_seed": job.seed, "train_set": ts.name,
            "train_source": ts.source, "n_train_trajectories": len(ts.records),
            "n_train_task_overlap": n_overlap,
            "preset": ts.preset, "hparams": ts.hparams.as_dict(),
            "agent_normalize": args.agent_normalize,
            "reduce": args.reduce,
            "pca": None if reducer is None else {k: v for k, v in reducer.items()
                                                 if not k.endswith("_V")},
            "checkpoint": str(ckpt_file),
            "model_selection": blob.get("model_selection"),
            "best_val_acc": blob.get("best_val_acc"),
            "best_epoch": blob.get("best_epoch"),
            "eval_batch_size": 1, "top_k": args.top_k,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_total": len(todo), "n_already_done": len(done),
            "n_remaining": len(remaining), "resumed": bool(done),
        })

        for record in remaining:
            payload = test_feats.load(str(record["id"]))
            if payload is not None and reducer is not None:
                payload = apply_reducer(payload, reducer)
            t0 = time.perf_counter()
            scored = None if payload is None else score_trajectory(model, payload, dev)
            elapsed = (time.perf_counter() - t0) * 1000.0
            writer.write(str(record["id"]), _build_doc(
                record, scored, job, args, model_name=args.model_name, backend=backend,
                include_gt=include_gt, overlap=proto.train_task_overlap(record, ts),
                fit_meta=fit_meta, elapsed_ms=elapsed, top_k=args.top_k,
            ))


def _unique_training_sets(jobs) -> list:
    seen, out = set(), []
    for job in jobs:
        key = (job.training_set.protocol, job.training_set.name)
        if key not in seen:
            seen.add(key)
            out.append(job.training_set)
    return out


if __name__ == "__main__":
    main()
