"""OAT runner — one (extractor, subset) per invocation.

Four stages, each idempotent, each resumable, run in order unless ``--stages``
names a subset of them:

1. ``states-train`` — read hidden states for the successful MCP-Atlas
   trajectories that ship with ``vendored/OAT/``. This is the only source of
   successes available: every corpus in ``data/`` is failures only, and OAT
   trains on successes alone.
2. ``train`` — fit PCA on those successes, then fit one neural CDE per training
   seed and calibrate its conformal threshold. Both are cached; a second subset
   reuses them.
3. ``states-test`` — read hidden states for the subset under test.
4. ``score`` — project, align, score, and write one JSON per trajectory per
   seed.

Everything lives under one root. Predictions go to
``{output}/oat.s<seed>/<id>.json``; the cached tensors sit beside them in
``{output}/_oat-states/``, and the trained models in
``{train_root}/_oat-ckpt/``. The leading underscore keeps both out of the
report's way, which reads only the method directories it is given.

GT setting: the vendored text carries no answer, so ``--gt without`` is the
faithful setting **and the default**, writing under ``outputs-rb-nogt/``.
``--gt with`` adds the repo's standard answer line to the question and writes
under ``outputs-rb-gt/``. The trained model is shared by both — it never saw an
answer either way — so only the test-side states are extracted twice.

Usage
-----
python -m oat.predict \\
    --model-name qwen3.5-9b --model-path ../hub/Qwen/Qwen3.5-9B \\
    --input  data/ww/hand-crafted \\
    --output outputs-rb-nogt/ww/hand-crafted/qwen3.5-9b
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from baselines.prompting.predict import load_records
from baselines.shared.common import nogt_root
from baselines.shared.runner import OutputWriter
from rb_shared.cache import StateCache

from oat import train as core
from oat.extract import (
    DEFAULT_AGGREGATION,
    DEFAULT_LAYER,
    dummy_extractor,
    extract_hidden_states_for_trajectory,
    load_extractor,
)
from oat.serialize import (
    build_input_text,
    load_mcp_atlas_records,
    question_text,
    serialize_trajectory,
)

METHOD = "oat"
STAGES = ("states-train", "train", "states-test", "score")
DEFAULT_TRAIN_DIR = "vendored/OAT/dataset/MCP-atlas/Qwen3.5-27B"
STATES_DIRNAME = "_oat-states"
CKPT_DIRNAME = "_oat-ckpt"


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def default_train_root(output: str, model_name: str) -> str:
    """Where this extractor's training states and checkpoints live.

    Always in the without-GT tree: the training corpus has no answer field, so
    a with-GT run trains on exactly the same tensors and should reuse them
    rather than extract a second identical copy.
    """
    parts = Path(output).parts
    for i, part in enumerate(parts):
        if part == "outputs" or part.startswith("outputs-"):
            head = part if part.endswith("-nogt") else nogt_root(part)
            return str(Path(*parts[:i], head, "mcp-atlas", "train", model_name))
    raise SystemExit(
        f"cannot place the training root beside {output!r}: the path has no "
        "'outputs…' component to mirror. Pass --train-root explicitly."
    )


# --------------------------------------------------------------------------
# Stage 1 & 3 — extraction
# --------------------------------------------------------------------------

def _get_extractor(args):
    if args.model_path == "dummy":
        return dummy_extractor()
    spec = {
        "path": args.model_path,
        "tokenizer": args.tokenizer,
        "dtype": args.dtype,
        "device": args.device,
    }
    if args.attn_implementation:
        spec["attn_implementation"] = args.attn_implementation
    return load_extractor(args.model_name, spec)


def _extract_into(
    cache: StateCache,
    documents: list[tuple[str, str, list, dict]],
    extractor,
    layers: Sequence[int],
    aggregation: str,
) -> tuple[int, int]:
    """Run the frozen model over every document missing from ``cache``."""
    written = skipped = 0
    todo = [d for d in documents if not cache.has(d[0])]
    if not todo:
        return written, skipped
    print(f"  extracting {len(todo)}/{len(documents)} into {cache.dir}", flush=True)
    t0 = time.perf_counter()
    for i, (traj_id, text, bounds, meta) in enumerate(todo, 1):
        hidden = extract_hidden_states_for_trajectory(
            text,
            bounds,
            extractor.model,
            extractor.tokenizer,
            max_length=extractor.max_length,
            aggregation=aggregation,
            layers=layers,
        )
        if hidden is None:
            # truncation swallowed every step — the vendored skip signal
            skipped += 1
            continue
        actual_t = int(hidden.shape[0])
        cache.save(
            traj_id,
            {
                "trajectory_id": traj_id,
                "hidden_states": hidden.to(torch.float16),
                "layers": list(layers),
                "aggregation": aggregation,
                "model_name": extractor.name,
                # the question occupies row 0 and is indexed -1, so absolute
                # step numbers survive the trip through the cache
                "step_agents": (["question"] + meta["step_agents"])[:actual_t],
                "step_is_model": ([True] + meta["step_is_model"])[:actual_t],
                "step_indices": ([-1] + list(range(meta["num_steps"])))[:actual_t],
                "error_steps": meta.get("error_steps", []),
                "num_steps_original": meta["num_steps"],
                "num_steps_extracted": actual_t,
            },
        )
        written += 1
        if i % 25 == 0 or i == len(todo):
            print(f"    {i}/{len(todo)}  ({time.perf_counter() - t0:.0f}s)")
    return written, skipped


def _test_documents(records: list[dict], include_gt: bool) -> list[tuple[str, str, list, dict]]:
    documents = []
    for record in records:
        step_text, bounds, agents, is_model = serialize_trajectory(record["history"])
        text, all_bounds = build_input_text(question_text(record, include_gt), step_text, bounds)
        documents.append(
            (
                record["id"],
                text,
                all_bounds,
                {
                    "step_agents": agents,
                    "step_is_model": is_model,
                    "num_steps": len(record["history"]),
                },
            )
        )
    return documents


def _train_documents(train_dir: str) -> list[tuple[str, str, list, dict]]:
    documents = []
    for record in load_mcp_atlas_records(train_dir, include_success=True):
        if not record["is_success"]:
            continue
        text, all_bounds = build_input_text(
            record["question"], record["text"], record["step_char_boundaries"]
        )
        documents.append(
            (
                record["id"],
                text,
                all_bounds,
                {
                    "step_agents": record["step_agents"],
                    "step_is_model": record["step_is_model"],
                    "num_steps": record["num_steps"],
                    "error_steps": record["error_steps"],
                },
            )
        )
    return documents


# --------------------------------------------------------------------------
# Loading cached states back into trajectories
# --------------------------------------------------------------------------

def load_cached_trajectories(
    cache: StateCache,
    ids: Sequence[str],
    layer: int,
    model_steps_only: bool = True,
) -> list[dict]:
    """Cached tensors to trajectory dicts the model can consume."""
    trajectories = []
    for traj_id in ids:
        payload = cache.load(traj_id)
        if payload is None:
            continue
        hidden = payload["hidden_states"].float()
        stored_layers = list(payload.get("layers") or [])
        if stored_layers:
            if layer not in stored_layers:
                raise SystemExit(
                    f"{cache.dir}: layer {layer} is not cached (stored: {stored_layers}). "
                    "Re-extract with --overwrite, or ask for a layer that is there."
                )
            hidden = hidden[:, stored_layers.index(layer), :]
        else:  # a full stack, the vendored layout
            hidden = hidden[:, layer, :]

        step_indices = [int(x) for x in payload["step_indices"]]
        step_is_model = [bool(x) for x in payload["step_is_model"]]
        if model_steps_only:
            hidden, step_indices = core.filter_model_steps(hidden, step_indices, step_is_model)
        if hidden.shape[0] < 2:
            continue
        trajectories.append(
            {
                "id": traj_id,
                "hidden_states": hidden,
                "step_indices": step_indices,
                "step_agents": list(payload.get("step_agents") or []),
                "error_steps": list(payload.get("error_steps") or []),
                "num_steps": len(step_indices),
                "num_steps_original": int(payload.get("num_steps_original", len(step_indices))),
            }
        )
    return trajectories


# --------------------------------------------------------------------------
# Stage 2 — training
# --------------------------------------------------------------------------

def _projector_paths(ckpt_dir: Path) -> tuple[Path, Path]:
    return ckpt_dir / "projector.pt", ckpt_dir / "source_latents.pt"


def fit_projector(train_trajectories: list[dict], ckpt_dir: Path, latent_dim: int) -> dict:
    """PCA plus normalization, fitted on successes and shared by every seed."""
    pca = core.fit_pca(train_trajectories, n_components=latent_dim)
    core.apply_pca(train_trajectories, pca)
    mean, std = core.compute_normalization_stats(train_trajectories)
    core.normalize_trajectories(train_trajectories, mean, std)

    projector_path, source_path = _projector_paths(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "components": torch.from_numpy(pca.components_.astype(np.float32)),
            "pca_mean": torch.from_numpy(pca.mean_.astype(np.float32)),
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            "latent_mean": mean,
            "latent_std": std,
            "latent_dim": latent_dim,
            "n_train_trajectories": len(train_trajectories),
        },
        projector_path,
    )
    # CORAL at score time needs the source latents it aligns onto
    torch.save({"latents": core._collect_latents(train_trajectories)}, source_path)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum() * 100:.1f}%", flush=True)
    return {"pca": pca, "mean": mean, "std": std}


def load_projector(ckpt_dir: Path) -> dict:
    projector_path, source_path = _projector_paths(ckpt_dir)
    if not projector_path.exists():
        raise SystemExit(f"no projector at {projector_path}; run the 'train' stage first")
    bundle = torch.load(projector_path, map_location="cpu", weights_only=False)
    bundle["source_latents"] = torch.load(source_path, map_location="cpu", weights_only=False)["latents"]
    return bundle


def apply_projector(trajectories: list[dict], bundle: dict) -> None:
    """Project and normalize with the *training* PCA and statistics."""
    components = bundle["components"]
    pca_mean = bundle["pca_mean"]
    for traj in trajectories:
        z = (traj["hidden_states"] - pca_mean) @ components.t()
        traj["latent_states"] = ((z - bundle["latent_mean"]) / bundle["latent_std"]).float()


def train_seed(train_trajectories: list[dict], seed: int, args, ckpt_dir: Path) -> Path:
    """Train one seed and calibrate it; returns the checkpoint path."""
    target = ckpt_dir / f"s{seed}" / "model.pt"
    if target.exists() and not args.overwrite:
        return target

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_set, val_set = core.split_train_val(train_trajectories, seed=seed)
    print(f"  seed {seed}: {len(train_set)} train / {len(val_set)} val successes", flush=True)

    model = core.build_model(latent_dim=args.latent_dim)
    model, history = core.train_model(
        model,
        train_set,
        val_set,
        device=args.train_device,
        epochs=args.epochs,
        patience=args.patience,
        log_every=args.log_every,
    )
    calibration = core.collect_calibration_scores(model, val_set, device=args.train_device)
    threshold = core.conformal_quantile_threshold(calibration, alpha=args.alpha)
    print(
        f"  seed {seed}: best_val_loss={history['best_val_loss']:.6f} "
        f"best_epoch={history['best_epoch']} conformal_threshold={threshold:.4f}"
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "conformal_threshold": threshold,
            "conformal_alpha": args.alpha,
            "n_calibration_scores": int(calibration.size),
            "history": history,
            "seed": seed,
            "latent_dim": args.latent_dim,
            "epochs": args.epochs,
            "patience": args.patience,
        },
        target,
    )
    return target


def load_seed_model(ckpt_dir: Path, seed: int, device: str) -> tuple[torch.nn.Module, dict]:
    path = ckpt_dir / f"s{seed}" / "model.pt"
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path}; run the 'train' stage first")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = core.build_model(latent_dim=int(payload.get("latent_dim", core.LATENT_DIM)))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the OAT representation-based baseline.")
    p.add_argument("--input", required=True, help="Subset directory of trajectory JSONs.")
    p.add_argument("--output", required=True,
                   help=f"Model-level output directory; predictions land in "
                        f"{{output}}/{METHOD}.s<seed>/ and cached states in "
                        f"{{output}}/{STATES_DIRNAME}/.")
    p.add_argument("--model-name", required=True,
                   help="Short label for the extractor; the <model> path component.")
    p.add_argument("--model-path", required=True,
                   help="Local checkpoint directory, or 'dummy' for a keyless CPU run.")
    p.add_argument("--tokenizer", default=None, help="Tokenizer override (default: --model-path).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default=None,
                   help="Device for the frozen extractor (default: cuda when available).")
    p.add_argument("--train-device", default="cpu",
                   help="Device for the CDE itself (default cpu). The model is a "
                        "handful of small MLPs over 8-to-20-step trajectories, so a "
                        "GPU spends its time launching kernels rather than "
                        "computing: CPU measures about 5x faster here.")
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--gt", default="without", choices=["with", "without"],
                   help="'without' matches the vendored text, which carries no "
                        "answer (default). 'with' adds the answer line to the "
                        "question before extraction.")
    p.add_argument("--seeds", default=",".join(str(s) for s in core.SEEDS),
                   help="Training seeds, comma-separated; each becomes its own "
                        f"{METHOD}.s<seed> method directory (default: the "
                        "vendored 42-46).")
    p.add_argument("--method-dir-prefix", default=METHOD,
                   help=f"Method directory prefix (default {METHOD}, giving {METHOD}.s42).")
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                   help="Which layer's hidden states to use (default -1, the last).")
    p.add_argument("--aggregation", default=DEFAULT_AGGREGATION, choices=["mean", "last"],
                   help="How to pool a step's tokens (default mean, the paper's).")
    p.add_argument("--latent-dim", type=int, default=core.LATENT_DIM)
    p.add_argument("--top-k", type=int, default=core.DETECTION_TOP_K)
    p.add_argument("--alpha", type=float, default=core.CONFORMAL_ALPHA,
                   help="Conformal miscoverage rate (default 0.2, the paper's).")
    p.add_argument("--epochs", type=int, default=core.EPOCHS)
    p.add_argument("--patience", type=int, default=core.PATIENCE)
    p.add_argument("--log-every", type=int, default=10, help="Epoch print interval; 0 silences it.")
    p.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR,
                   help="Corpus of successful trajectories to train on.")
    p.add_argument("--train-root", default=None,
                   help="Where training states and checkpoints live "
                        "(default: <root>-nogt/mcp-atlas/train/<model-name>).")
    p.add_argument("--stages", default=",".join(STAGES),
                   help=f"Comma-separated subset of {','.join(STAGES)}.")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="Redo everything the named stages produce.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # the extractor is huge and wants a GPU; the CDE is tiny and does not
    train_device = args.train_device
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for stage in stages:
        if stage not in STAGES:
            raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    include_gt = args.gt == "with"
    layers = (args.layer,)
    backend = "dummy" if args.model_path == "dummy" else "hf"

    train_root = Path(args.train_root or default_train_root(args.output, args.model_name))
    ckpt_dir = train_root / CKPT_DIRNAME
    train_states = StateCache(train_root / STATES_DIRNAME,
                              overwrite=args.overwrite and "states-train" in stages)
    test_states = StateCache(Path(args.output) / STATES_DIRNAME,
                             overwrite=args.overwrite and "states-test" in stages)

    records = load_records(args.input)
    end_idx = args.end_idx if args.end_idx is not None else len(records)
    sliced = records[args.start_idx:end_idx]
    print(f"  {len(sliced)} trajectories [{args.start_idx}:{end_idx}] from {args.input}", flush=True)

    extractor = None

    def need_extractor():
        nonlocal extractor
        if extractor is None:
            print(f"  loading extractor {args.model_name} ({args.model_path})", flush=True)
            extractor = _get_extractor(args)
        return extractor

    # --- stage 1: training-corpus states ---------------------------------
    if "states-train" in stages:
        documents = _train_documents(args.train_dir)
        if train_states.missing([d[0] for d in documents]):
            written, skipped = _extract_into(
                train_states, documents, need_extractor(), layers, args.aggregation
            )
            print(f"  training states: +{written} written, {skipped} skipped", flush=True)
        train_states.write_manifest({
            "corpus": args.train_dir, "extractor": args.model_name,
            "model_path": args.model_path, "layers": list(layers),
            "aggregation": args.aggregation, "gt_in_prompt": False,
            "n_trajectories": len(train_states),
        })

    # --- stage 2: fit the projector and the per-seed models ---------------
    if "train" in stages:
        documents = _train_documents(args.train_dir)
        train_trajs = load_cached_trajectories(
            train_states, [d[0] for d in documents], args.layer
        )
        if not train_trajs:
            raise SystemExit(
                f"no cached training states in {train_states.dir}; run the "
                "'states-train' stage first"
            )
        print(f"  {len(train_trajs)} successful training trajectories", flush=True)
        projector_path, _ = _projector_paths(ckpt_dir)
        if projector_path.exists() and not args.overwrite:
            bundle = load_projector(ckpt_dir)
            apply_projector(train_trajs, bundle)
        else:
            fit_projector(train_trajs, ckpt_dir, args.latent_dim)
        for seed in seeds:
            train_seed(train_trajs, seed, args, ckpt_dir)
        StateCache(ckpt_dir).write_manifest({
            "extractor": args.model_name, "model_path": args.model_path,
            "train_dir": args.train_dir, "layer": args.layer,
            "aggregation": args.aggregation, "latent_dim": args.latent_dim,
            "seeds": seeds, "alpha": args.alpha, "epochs": args.epochs,
        })

    # --- stage 3: test-corpus states --------------------------------------
    if "states-test" in stages:
        documents = _test_documents(sliced, include_gt)
        if test_states.missing([d[0] for d in documents]):
            written, skipped = _extract_into(
                test_states, documents, need_extractor(), layers, args.aggregation
            )
            print(f"  test states: +{written} written, {skipped} skipped", flush=True)
        test_states.write_manifest({
            "corpus": args.input, "extractor": args.model_name,
            "model_path": args.model_path, "layers": list(layers),
            "aggregation": args.aggregation, "gt_in_prompt": include_gt,
            "n_trajectories": len(test_states),
        })

    if "score" not in stages:
        return

    # --- stage 4: score ---------------------------------------------------
    bundle = load_projector(ckpt_dir)

    # CORAL is fitted over the whole subset, never over the slice: the
    # alignment must not depend on how the work happened to be divided up.
    all_trajs = load_cached_trajectories(test_states, [r["id"] for r in records], args.layer)
    if not all_trajs:
        raise SystemExit(f"no cached test states in {test_states.dir}; run 'states-test' first")
    if len(all_trajs) < len(records):
        print(f"  note: CORAL sees {len(all_trajs)}/{len(records)} trajectories "
              "— extract the whole subset for the canonical alignment")
    apply_projector(all_trajs, bundle)
    core.apply_coral_alignment(all_trajs, [{"latent_states": bundle["source_latents"]}])
    by_id = {t["id"]: t for t in all_trajs}
    coral_n = len(all_trajs)

    for seed in seeds:
        method_dir = f"{args.method_dir_prefix}.s{seed}"
        writer = OutputWriter(Path(args.output) / method_dir, overwrite=args.overwrite)
        done = writer.done_ids()
        remaining = [r for r in sliced if r["id"] not in done]
        print(f"  {method_dir}: {len(done)} done, {len(remaining)} to run", flush=True)
        if not remaining:
            print(f"  skip (complete): {writer.dir}", flush=True)
            continue

        model, ckpt = load_seed_model(ckpt_dir, seed, train_device)
        writer.write_run_config({
            "model": args.model_name,
            "model_arg": args.model_path,
            "method": METHOD,
            "method_dir": method_dir,
            "subset": Path(args.input).name,
            "backend": backend,
            "gt_in_prompt": include_gt,
            "train_seed": seed,
            "train_dir": args.train_dir,
            "checkpoint": str(ckpt_dir / f"s{seed}" / "model.pt"),
            "conformal_threshold": ckpt["conformal_threshold"],
            "conformal_alpha": args.alpha,
            "top_k": args.top_k,
            "layer": args.layer,
            "aggregation": args.aggregation,
            "latent_dim": args.latent_dim,
            "coral_n_trajectories": coral_n,
            "best_val_loss": ckpt["history"]["best_val_loss"],
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_total": len(sliced),
            "n_already_done": len(done),
            "n_remaining": len(remaining),
            "resumed": bool(done),
        })

        t0 = time.perf_counter()
        for record in remaining:
            traj = by_id.get(record["id"])
            scored = (
                None if traj is None
                else core.score_trajectory(
                    model, traj, device=train_device, top_k=args.top_k,
                    conformal_threshold=ckpt["conformal_threshold"],
                    conformal_min_detections=core.CONFORMAL_MIN_DETECTIONS,
                )
            )
            writer.write(record["id"], _build_doc(record, scored, seed, args, include_gt,
                                                  backend, ckpt, method_dir, coral_n))
        elapsed = time.perf_counter() - t0
        print(f"  wrote {writer.dir}  ({len(writer.done_ids())}/{len(sliced)} files, {elapsed:.1f}s)")


def _build_doc(record, scored, seed, args, include_gt, backend, ckpt, method_dir, coral_n) -> dict:
    """One prediction file: the GUIDE keys, plus what OAT alone can say."""
    history = record.get("history") or []
    predicted_step = None if scored is None else int(scored["predicted_step"])
    predicted_agent = None
    if predicted_step is not None and 0 <= predicted_step < len(history):
        predicted_agent = history[predicted_step].get("role")

    gold_step = record.get("gold_step")
    # True when the annotated step got no score at all — a non-agent turn, or
    # an index past the end of the log. Either way step@1 was unreachable.
    gold_on_filtered = None
    if scored is not None:
        try:
            gold_on_filtered = int(gold_step) not in set(scored["score_step_indices"])
        except (TypeError, ValueError):
            gold_on_filtered = None

    return {
        "id": record["id"],
        "filename": record["filename"],
        "question_id": record["question_id"],
        "method": METHOD,
        "model": args.model_name,
        "backend": backend,
        "gt_in_prompt": include_gt,
        "predicted_agent": predicted_agent,
        "predicted_step": predicted_step,
        "gold_agent": record["gold_agent"],
        "gold_step": gold_step,
        # `raw` is what a prompting baseline would have parsed. OAT parses
        # nothing; its per-step anomaly scores are the equivalent evidence.
        "raw": None if scored is None else json.dumps(
            [round(float(s), 6) for s in scored["scores"]]
        ),
        "calls": [],
        "method_dir": method_dir,
        "train_seed": seed,
        "extractor": args.model_name,
        "layer": args.layer,
        "aggregation": args.aggregation,
        "scores": None if scored is None else [round(float(s), 6) for s in scored["scores"]],
        "score_step_indices": None if scored is None else scored["score_step_indices"],
        "topk_steps": None if scored is None else scored["topk_detection"],
        "topk_k": args.top_k,
        "conformal_steps": None if scored is None else scored["conformal_detection"],
        "conformal_threshold": ckpt["conformal_threshold"],
        "conformal_alpha": args.alpha,
        "num_scored_steps": None if scored is None else scored["num_scored_steps"],
        "num_steps": len(history),
        "gold_on_filtered_step": gold_on_filtered,
        "coral_n_trajectories": coral_n,
        "inference_time_ms": None if scored is None else round(scored["inference_time_ms"], 3),
    }


if __name__ == "__main__":
    main()
