"""The paper's own metrics, recomputed from committed prediction files.

The repo scores every baseline the same way — step@1 and agent@1 over seeded
splits — and that is what makes a representation-based method comparable to a
prompting one. But OAT's paper reports something different: it predicts a
*set* of failure-contributing steps and scores that set with precision, recall,
F1 and hit rate, plus AUROC and AUPRC over the raw per-step scores. Both views
answer real questions, so this module provides the second without disturbing
the first. It reads the ``scores``, ``topk_steps`` and ``conformal_steps`` that
``oat/predict.py`` already stores in every output file; nothing is re-run.

StepFinder's paper reports a third view — Acc@K, MRR@3 and tolerance accuracy
— which reads the same stored ranking, so those columns appear for every
method here. A method with no calibration machinery leaves the four
``conformal_detection_*`` cells empty rather than borrowing its top-1 set.

One caveat travels with every number here. The paper's benchmark annotates
*every* contributing step, while this repo's corpora annotate one decisive
step. Against a single-element gold set, recall of a 3-element prediction is
either 0 or 1, and precision is capped at 1/3. Read hit rate and AUROC; treat
precision and F1 as bookkeeping.

Usage
-----
python -m rb_shared.rb_metrics \\
    --pred-root outputs-rb-nogt/ww --subset hand-crafted \\
    --model qwen3.5-9b --methods oat.s42,oat.s43

With no ``--methods`` every method directory present is scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _gold_steps(doc: dict) -> list[int]:
    """The annotated step, as a set of one — the shape the paper's metrics take."""
    gold = doc.get("gold_step")
    try:
        return [int(gold)]
    except (TypeError, ValueError):
        return []


def _set_metrics(docs: list[dict], detection_key: str) -> dict | None:
    """Vendored ``_set_metrics`` (``vendored/OAT/evaluate.py:21-40``).

    Returns ``None`` when no document carries the key. The vendored line falls
    back to the top-1 step whenever a detection set is missing, which is right
    for OAT — a conformal set that selects nothing still means "the top step".
    It is wrong for a method that has no conformal machinery at all, as
    StepFinder does not: the fallback would file top-1 numbers under a
    "conformal" heading and read as a calibrated result. So an absent key is
    reported as absent.
    """
    if not any(doc.get(detection_key) is not None for doc in docs):
        return None
    precisions, recalls, f1s, hits = [], [], [], []
    for doc in docs:
        decoded = doc.get(detection_key) or ([doc["predicted_step"]] if doc.get("predicted_step") is not None else [])
        pred_set = {int(x) for x in decoded}
        gt_set = set(_gold_steps(doc))
        inter = pred_set & gt_set
        precision = len(inter) / len(pred_set) if pred_set else 0.0
        recall = len(inter) / len(gt_set) if gt_set else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
        hits.append(1.0 if inter else 0.0)
    return {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
    }


def _pooled_labels_and_scores(docs: list[dict], score_key: str = "scores") -> tuple[list[int], list[float]]:
    labels, scores_all = [], []
    for doc in docs:
        gt_set = set(_gold_steps(doc))
        scores = doc.get(score_key) or doc.get("scores") or []
        step_ids = doc.get("score_step_indices") or list(range(len(scores)))
        if len(step_ids) != len(scores):
            step_ids = list(range(len(scores)))
        for step_id, score in zip(step_ids, scores):
            labels.append(1 if int(step_id) in gt_set else 0)
            scores_all.append(float(score))
    return labels, scores_all


def step_auroc(docs: list[dict], score_key: str = "scores") -> float:
    """Vendored ``step_auroc``: how well the scores rank the gold step."""
    from sklearn.metrics import roc_auc_score

    labels, scores_all = _pooled_labels_and_scores(docs, score_key)
    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores_all))


def step_auprc(docs: list[dict], score_key: str = "scores") -> float:
    from sklearn.metrics import average_precision_score

    labels, scores_all = _pooled_labels_and_scores(docs, score_key)
    if len(set(labels)) < 2:
        return 0.5
    return float(average_precision_score(labels, scores_all))


TOLERANCES = (1, 2, 3, 4, 5)


def _ranking_metrics(docs: list[dict]) -> dict:
    """Acc@K, MRR@3 and tolerance accuracy — StepFinder's paper metrics.

    All three read the ranking the method already committed to: ``topk_steps``
    for the candidate list, ``predicted_step`` for the top-1. They apply to any
    method that stores those keys, so OAT gets them too, and they are the only
    way to compare against StepFinder's Table 3 and Figure 3-4.

    ``acc@k`` is the gold step's presence in the first ``k`` candidates. Because
    ``topk_steps`` is sorted by *position*, not by score, a truthful ``acc@1``
    and ``mrr`` need the score order back; both are recovered from ``scores``
    when present, and fall back to ``predicted_step`` when not.
    """
    hits = {k: [] for k in (1, 2, 3)}
    rr, tolerance = [], {d: [] for d in TOLERANCES}
    for doc in docs:
        gold = _gold_steps(doc)
        if not gold:
            continue
        gold_step = gold[0]
        ranked = _ranked_steps(doc)
        rank = ranked.index(gold_step) + 1 if gold_step in ranked else None
        for k in hits:
            hits[k].append(1.0 if rank is not None and rank <= k else 0.0)
        rr.append(1.0 / rank if rank is not None and rank <= 3 else 0.0)
        top1 = doc.get("predicted_step")
        for d in TOLERANCES:
            tolerance[d].append(
                1.0 if top1 is not None and abs(int(top1) - gold_step) <= d else 0.0
            )
    out = {f"acc@{k}": float(np.mean(v)) if v else 0.0 for k, v in hits.items()}
    out["mrr@3"] = float(np.mean(rr)) if rr else 0.0
    out.update({f"tol_acc@{d}": float(np.mean(v)) if v else 0.0 for d, v in tolerance.items()})
    return out


def _ranked_steps(doc: dict) -> list[int]:
    """The method's candidates, best first."""
    scores = doc.get("scores")
    indices = doc.get("score_step_indices")
    if scores:
        if not indices or len(indices) != len(scores):
            indices = list(range(len(scores)))
        order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
        return [int(indices[i]) for i in order]
    top = doc.get("predicted_step")
    return [] if top is None else [int(top)]


def compute_metrics(docs: list[dict], score_key: str = "scores") -> dict:
    """Vendored ``compute_metrics`` (``evaluate.py:69-77``), same key names."""
    scorable = [d for d in docs if d.get("scores")]
    metrics = {}
    for key, prefix in (("topk_steps", "topk_detection"), ("conformal_steps", "conformal_detection")):
        found = _set_metrics(scorable, key)
        metrics.update({f"{prefix}_{k}": (None if found is None else found[k])
                        for k in ("precision", "recall", "f1", "hit_rate")})
    metrics.update(_ranking_metrics(scorable))
    metrics["auroc"] = step_auroc(scorable, score_key)
    metrics["auprc"] = step_auprc(scorable, score_key)
    metrics["n"] = len(scorable)
    metrics["n_files"] = len(docs)
    return metrics


def load_method_dir(method_dir: Path) -> list[dict]:
    """Every prediction file in one method directory, ``_run.json`` excluded."""
    docs = []
    for path in sorted(method_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        docs.append(json.loads(path.read_text()))
    return docs


COLUMNS = (
    "topk_detection_precision",
    "topk_detection_recall",
    "topk_detection_f1",
    "topk_detection_hit_rate",
    "conformal_detection_precision",
    "conformal_detection_recall",
    "conformal_detection_f1",
    "conformal_detection_hit_rate",
    "acc@1",
    "acc@2",
    "acc@3",
    "mrr@3",
    "tol_acc@1",
    "tol_acc@2",
    "tol_acc@3",
    "tol_acc@4",
    "tol_acc@5",
    "auroc",
    "auprc",
    "n",
)


def _cell(value) -> str:
    """A metric that does not apply is an empty cell, never the word "None"."""
    if value is None:
        return ""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main() -> None:
    p = argparse.ArgumentParser(prog="rb_shared.rb_metrics", description=__doc__.split("\n")[0])
    p.add_argument("--pred-root", required=True,
                   help="Dataset-level root, e.g. outputs-rb-nogt/ww")
    p.add_argument("--subset", action="append", default=[],
                   help="Subset name (repeatable; default: every subset present).")
    p.add_argument("--model", action="append", default=[],
                   help="Extractor name (repeatable; default: every model present).")
    p.add_argument("--methods", default=None,
                   help="Comma-separated method directories (default: every one present).")
    p.add_argument("--score-key", default="scores", choices=("scores", "scores_logit"),
                   help="Which stored per-step number AUROC/AUPRC rank. A method whose "
                        "`scores` are softmax probabilities is length-biased when pooled "
                        "across trajectories of different lengths; `scores_logit` is not.")
    p.add_argument("--out", default=None,
                   help="TSV destination (default: <pred-root>/reports/rb_metrics.tsv).")
    args = p.parse_args()

    root = Path(args.pred_root)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None
    subsets = args.subset or sorted(d.name for d in root.iterdir() if d.is_dir() and d.name != "reports")

    rows = []
    for subset in subsets:
        subset_dir = root / subset
        if not subset_dir.is_dir():
            continue
        models = args.model or sorted(d.name for d in subset_dir.iterdir() if d.is_dir())
        for model in models:
            model_dir = subset_dir / model
            if not model_dir.is_dir():
                continue
            # Default to every method directory present. A hard-coded list
            # silently reports nothing the moment a second baseline lands in
            # the same tree, which is exactly what StepFinder does.
            found = methods or sorted(
                d.name for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
            )
            for method in found:
                method_dir = model_dir / method
                if not method_dir.is_dir():
                    continue
                docs = load_method_dir(method_dir)
                if not docs:
                    continue
                metrics = compute_metrics(docs, args.score_key)
                rows.append({"subset": subset, "model": model, "method": method, **metrics})

    if not rows:
        print(f"no prediction directories found under {root}")
        return

    out = Path(args.out) if args.out else root / "reports" / "rb_metrics.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["subset", "model", "method", *COLUMNS]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(_cell(row.get(c)) for c in header))
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
