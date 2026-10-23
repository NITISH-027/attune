"""
ATTUNE Evaluation Metrics Suite
===============================
Mathematical metric functions adhering to docs/spec.md Section 5:
- Negative Transfer Rate (NTR) = FP / (FP + TN) on negative ground truth (gold=0)
- False Suppression Rate (FSR) = FN / (FN + TP) on positive ground truth (gold=1)
- Repeated-Correction Rate (RCR) = proportion of turns following explicit corrections
  where the system repeats the discredited preference (predicts 1 despite override)
- Standard Classification Metrics: Accuracy, Precision, Recall, F1
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_ntr(y_true: List[int], y_pred: List[int]) -> float:
    """
    Negative Transfer Rate (NTR):
    Conditional probability that the system inappropriately applies a preference
    when it should have been suppressed (over-personalization error on gold=0).
    NTR = FP / (FP + TN)
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    neg_mask = (y_true_arr == 0)
    total_neg = np.sum(neg_mask)
    if total_neg == 0:
        return 0.0

    # FP is where true is 0 but predicted is 1
    fp = np.sum((y_pred_arr[neg_mask] == 1))
    return float(fp / total_neg)


def compute_fsr(y_true: List[int], y_pred: List[int]) -> float:
    """
    False Suppression Rate (FSR):
    Conditional probability that the system mistakenly rejects a valid, helpful preference
    (under-personalization error on gold=1).
    FSR = FN / (TP + FN)
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    pos_mask = (y_true_arr == 1)
    total_pos = np.sum(pos_mask)
    if total_pos == 0:
        return 0.0

    # FN is where true is 1 but predicted is 0
    fn = np.sum((y_pred_arr[pos_mask] == 0))
    return float(fn / total_pos)


def compute_rcr(
    records: List[Dict[str, Any]],
    system_predict_fn: Optional[Callable[[Dict[str, Any]], int]] = None,
    y_pred: Optional[List[int]] = None,
) -> float:
    """
    Repeated-Correction Rate (RCR):
    Measures the proportion of turns following an explicit correction/override where
    the system fails to suppress the discredited preference and repeats it (predicts 1).
    Evaluated over instances where `recent_override` is not None or `scenario_type == 'explicit_reversal'`.
    """
    if len(records) == 0:
        return 0.0

    correction_indices = [
        i for i, r in enumerate(records)
        if r.get("recent_override") is not None or r.get("scenario_type") == "explicit_reversal"
    ]

    if not correction_indices:
        return 0.0

    if y_pred is not None:
        preds = [y_pred[i] for i in correction_indices]
    elif system_predict_fn is not None:
        preds = [system_predict_fn(records[i]) for i in correction_indices]
    else:
        raise ValueError("Either system_predict_fn or y_pred must be supplied to compute_rcr")

    repeated_mistakes = sum(1 for p in preds if p == 1)
    return float(repeated_mistakes / len(correction_indices))


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Computes standard binary classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def evaluate_system(
    records: List[Dict[str, Any]],
    y_pred: List[int],
    system_name: str = "System",
) -> Dict[str, Any]:
    """
    Computes a comprehensive evaluation dictionary for a given system output.
    """
    y_true = [r["gold_applicability"] for r in records]
    class_metrics = compute_classification_metrics(y_true, y_pred)
    ntr = compute_ntr(y_true, y_pred)
    fsr = compute_fsr(y_true, y_pred)
    rcr = compute_rcr(records, y_pred=y_pred)

    return {
        "system_name": system_name,
        "accuracy": class_metrics["accuracy"],
        "precision": class_metrics["precision"],
        "recall": class_metrics["recall"],
        "f1": class_metrics["f1"],
        "ntr": ntr,
        "fsr": fsr,
        "rcr": rcr,
        "num_records": len(records),
    }
