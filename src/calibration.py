"""
ATTUNE Probability Calibration and Threshold Optimization
=========================================================
Adheres to docs/spec.md Section 5.6 (ECE) and Section 6.3 (Graduation Gates):
- Temperature Scaling: Optimizes scalar temperature T on validation NLL loss.
- Expected Calibration Error (ECE): Computes 10-bin calibration gap.
- Threshold Optimization: Sweeps tau in [0.10, 0.90] to find tau_star that minimizes
  Negative Transfer Rate (NTR) subject to False Suppression Rate (FSR) <= 0.05.
"""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("attune_calibration")


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) according to docs/spec.md Section 5.6:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    n = len(probs)
    if n == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # In last bin, include upper boundary
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_conf = np.mean(probs[in_bin])
            bin_acc = np.mean(labels[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def fit_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    """
    Fits scalar temperature T on validation logits using NLL loss minimization.
    Updates model.temperature in-place and returns the fitted value.
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Raw uncalibrated logits (forward pass)
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # Temperature parameter to optimize (log-scale for stability: T = exp(s))
    log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32, requires_grad=True))
    optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def eval_loss():
        optimizer.zero_grad()
        temp = torch.exp(log_temp)
        scaled_logits = logits_tensor / temp
        loss = criterion(scaled_logits, labels_tensor)
        loss.backward()
        return loss

    try:
        optimizer.step(eval_loss)
    except Exception as exc:
        logger.warning(f"L-BFGS optimization encountered exception: {exc}. Retrying with Adam...")
        adam = torch.optim.Adam([log_temp], lr=0.05)
        for _ in range(50):
            adam.zero_grad()
            temp = torch.exp(log_temp)
            loss = criterion(logits_tensor / temp, labels_tensor)
            loss.backward()
            adam.step()

    optimal_temperature = float(torch.exp(log_temp).detach().item())
    optimal_temperature = max(0.01, min(10.0, optimal_temperature))

    # Update model parameter
    model.temperature.data = torch.tensor(optimal_temperature, dtype=torch.float32)
    logger.info(f"Fitted optimal temperature: {optimal_temperature:.4f}")
    return optimal_temperature


def find_optimal_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    target_fsr_max: float = 0.05,
    threshold_min: float = 0.10,
    threshold_max: float = 0.90,
    step: float = 0.02,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Sweeps candidate decision thresholds to find tau_star that minimizes Negative
    Transfer Rate (NTR) while maintaining False Suppression Rate (FSR) <= target_fsr_max.

    Returns:
        tau_star: Optimal threshold float
        best_metrics: Metrics at tau_star
        sweep_history: List of metric dicts for all tested thresholds
    """
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    total_pos = np.sum(pos_mask)
    total_neg = np.sum(neg_mask)

    assert total_pos > 0 and total_neg > 0, "Labels must contain both positive and negative instances."

    thresholds = np.arange(threshold_min, threshold_max + step / 2, step)
    sweep_history = []
    feasible_candidates = []

    for tau in thresholds:
        preds = (probs >= tau).astype(int)

        # NTR = FP / (FP + TN)
        fp = np.sum((preds == 1) & neg_mask)
        ntr = float(fp / total_neg) if total_neg > 0 else 0.0

        # FSR = FN / (FN + TP)
        fn = np.sum((preds == 0) & pos_mask)
        fsr = float(fn / total_pos) if total_pos > 0 else 0.0

        acc = float(np.mean(preds == labels))

        entry = {
            "threshold": round(float(tau), 4),
            "ntr": round(ntr, 4),
            "fsr": round(fsr, 4),
            "accuracy": round(acc, 4),
        }
        sweep_history.append(entry)

        if fsr <= target_fsr_max:
            feasible_candidates.append(entry)

    if feasible_candidates:
        # Among candidates with FSR <= 0.05, minimize NTR, tie-breaking on higher threshold
        best = min(feasible_candidates, key=lambda x: (x["ntr"], -x["threshold"]))
        tau_star = best["threshold"]
        logger.info(f"Optimal threshold found within FSR <= {target_fsr_max}: tau* = {tau_star:.4f} (NTR={best['ntr']:.4f}, FSR={best['fsr']:.4f})")
    else:
        # Fallback if no threshold satisfies strict FSR constraint: minimize NTR + 2*FSR
        best = min(sweep_history, key=lambda x: (x["ntr"] + 2 * x["fsr"]))
        tau_star = best["threshold"]
        logger.warning(f"No threshold satisfied FSR <= {target_fsr_max}. Best compromise: tau* = {tau_star:.4f} (NTR={best['ntr']:.4f}, FSR={best['fsr']:.4f})")

    return tau_star, best, sweep_history
