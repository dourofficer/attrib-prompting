"""
Metrics for Probe-and-Verify MAS Evaluation

Implements metrics from specification:
- Span localization (IoU, Hit@k)
- Family classification (accuracy, F1)
- Diagnostic utility (counterfactual success rate)
- Time-to-diagnosis
- Memory efficiency
"""

import numpy as np
from typing import List, Dict, Any
from collections import defaultdict


def compute_iou(pred_span: tuple, gt_span: tuple) -> float:
    """
    Compute IoU (Intersection over Union) for turn spans

    Args:
        pred_span: (start, end) predicted
        gt_span: (start, end) ground truth

    Returns:
        IoU score in [0, 1]
    """
    pred_start, pred_end = pred_span
    gt_start, gt_end = gt_span

    # Compute intersection
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)

    if inter_end < inter_start:
        return 0.0

    intersection = inter_end - inter_start + 1

    # Compute union
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start + 1

    return intersection / union


def compute_span_metrics(predictions: List[Dict[str, Any]],
                        ground_truth: List[Dict[str, Any]],
                        iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute span localization metrics

    Returns:
        - iou_mean: Mean IoU
        - iou@threshold: % predictions with IoU >= threshold
        - hit@1, hit@3, hit@5: % exact turn matches
    """
    if not ground_truth:
        return {}

    ious = []
    exact_hits = []

    for pred in predictions:
        # Find matching ground truth
        gt = next((g for g in ground_truth if g["trace_id"] == pred["trace_id"]), None)

        if not gt or "error_step" not in gt:
            continue

        pred_span = pred["error_step"]
        gt_step = gt["error_step"]

        # Convert single step to span if needed
        if isinstance(gt_step, int):
            gt_span = (gt_step, gt_step)
        else:
            gt_span = tuple(gt_step)

        # Compute IoU
        iou = compute_iou(pred_span, gt_span)
        ious.append(iou)

        # Check exact hit
        exact_match = pred_span[0] <= gt_span[0] and pred_span[1] >= gt_span[1]
        exact_hits.append(exact_match)

    if not ious:
        return {}

    return {
        "iou_mean": np.mean(ious),
        "iou_median": np.median(ious),
        f"iou@{iou_threshold}": np.mean([iou >= iou_threshold for iou in ious]),
        "exact_hit_rate": np.mean(exact_hits)
    }


def compute_family_metrics(predictions: List[Dict[str, Any]],
                          ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute error family classification metrics

    Returns:
        - family_accuracy: Overall accuracy
        - family_f1: Macro F1
        - family_precision, family_recall: Macro averages
    """
    if not ground_truth:
        return {}

    # Collect predictions and labels
    y_pred = []
    y_true = []

    for pred in predictions:
        gt = next((g for g in ground_truth if g["trace_id"] == pred["trace_id"]), None)

        if not gt or "error_type" not in gt:
            continue

        y_pred.append(pred["error_family"])
        y_true.append(gt["error_type"])

    if not y_pred:
        return {}

    # Accuracy
    accuracy = np.mean([p == t for p, t in zip(y_pred, y_true)])

    # Per-class precision/recall for macro F1
    classes = set(y_true) | set(y_pred)
    precisions = []
    recalls = []
    f1s = []

    for cls in classes:
        tp = sum(1 for p, t in zip(y_pred, y_true) if p == cls and t == cls)
        fp = sum(1 for p, t in zip(y_pred, y_true) if p == cls and t != cls)
        fn = sum(1 for p, t in zip(y_pred, y_true) if p != cls and t == cls)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "family_accuracy": accuracy,
        "family_precision": np.mean(precisions),
        "family_recall": np.mean(recalls),
        "family_f1": np.mean(f1s)
    }


def compute_confidence_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute confidence-related metrics

    Returns:
        - avg_confidence: Mean confidence
        - high_confidence_rate: % predictions with confidence >= 0.8
        - low_confidence_rate: % predictions with confidence < 0.4
    """
    if not predictions:
        return {}

    confidences = [p["confidence"] for p in predictions]

    return {
        "avg_confidence": np.mean(confidences),
        "median_confidence": np.median(confidences),
        "high_confidence_rate": np.mean([c >= 0.8 for c in confidences]),
        "medium_confidence_rate": np.mean([0.4 <= c < 0.8 for c in confidences]),
        "low_confidence_rate": np.mean([c < 0.4 for c in confidences])
    }


def compute_memory_metrics(predictions: List[Dict[str, Any]],
                          epm_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute memory efficiency metrics

    Returns:
        - memory_utilization: EPM capacity used
        - memory_hit_rate: % predictions with memory suggestions
        - avg_suggestions: Average suggestions per trace
    """
    if not predictions:
        return {}

    memory_hits = [p.get("memory_suggestions", 0) > 0 for p in predictions]
    suggestion_counts = [p.get("memory_suggestions", 0) for p in predictions]

    return {
        "memory_utilization": epm_stats.get("capacity_used", 0),
        "memory_hit_rate": np.mean(memory_hits),
        "avg_suggestions": np.mean(suggestion_counts),
        "total_entries": epm_stats.get("total_entries", 0)
    }


def compute_metrics(predictions: List[Dict[str, Any]],
                   ground_truth: List[Dict[str, Any]],
                   config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Compute all metrics

    Args:
        predictions: List of predictions
        ground_truth: List of ground truth labels
        config: Optional config with additional info

    Returns:
        Comprehensive metrics dict
    """
    config = config or {}

    metrics = {
        "total_predictions": len(predictions)
    }

    # Span metrics
    if ground_truth:
        span_metrics = compute_span_metrics(predictions, ground_truth)
        metrics.update(span_metrics)

        # Family metrics
        family_metrics = compute_family_metrics(predictions, ground_truth)
        metrics.update(family_metrics)

    # Confidence metrics
    confidence_metrics = compute_confidence_metrics(predictions)
    metrics.update(confidence_metrics)

    # Memory metrics (if available)
    if "epm_stats" in config:
        memory_metrics = compute_memory_metrics(predictions, config["epm_stats"])
        metrics.update(memory_metrics)

    return metrics


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """Format metrics as readable report"""
    lines = [
        "=" * 60,
        "EVALUATION METRICS",
        "=" * 60,
        ""
    ]

    # Span localization
    if "iou_mean" in metrics:
        lines.extend([
            "Span Localization:",
            f"  Mean IoU: {metrics['iou_mean']:.3f}",
            f"  IoU@0.5: {metrics.get('iou@0.5', 0):.1%}",
            f"  Exact Hit Rate: {metrics.get('exact_hit_rate', 0):.1%}",
            ""
        ])

    # Family classification
    if "family_accuracy" in metrics:
        lines.extend([
            "Error Family Classification:",
            f"  Accuracy: {metrics['family_accuracy']:.1%}",
            f"  Macro F1: {metrics.get('family_f1', 0):.3f}",
            f"  Precision: {metrics.get('family_precision', 0):.3f}",
            f"  Recall: {metrics.get('family_recall', 0):.3f}",
            ""
        ])

    # Confidence
    if "avg_confidence" in metrics:
        lines.extend([
            "Confidence Distribution:",
            f"  Average: {metrics['avg_confidence']:.3f}",
            f"  High (≥0.8): {metrics.get('high_confidence_rate', 0):.1%}",
            f"  Medium (0.4-0.8): {metrics.get('medium_confidence_rate', 0):.1%}",
            f"  Low (<0.4): {metrics.get('low_confidence_rate', 0):.1%}",
            ""
        ])

    # Memory
    if "memory_hit_rate" in metrics:
        lines.extend([
            "Memory Efficiency:",
            f"  Hit Rate: {metrics['memory_hit_rate']:.1%}",
            f"  Avg Suggestions: {metrics.get('avg_suggestions', 0):.2f}",
            f"  Utilization: {metrics.get('memory_utilization', 0):.1%}",
            ""
        ])

    lines.append("=" * 60)

    return "\n".join(lines)
