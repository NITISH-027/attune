"""
ATTUNE Gating Model Training & Calibration Pipeline
===================================================
CPU-optimized training script for the ATTUNE preference gating model.
Loads data/splits/train.jsonl and val.jsonl, trains the model for 4 epochs,
performs temperature scaling calibration, computes Expected Calibration Error (ECE),
finds the optimal threshold tau_star, and saves artifacts to models/:
- models/attune_gate_best.pt
- models/config.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import AttuneDataset, DEFAULT_MODEL_NAME, DEFAULT_MAX_LENGTH
from src.model import AttuneGatingModel
from src.calibration import fit_temperature, compute_ece, find_optimal_threshold
from eval.metrics import compute_classification_metrics, compute_ntr, compute_fsr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_attune")

DEFAULT_TRAIN_PATH = ROOT_DIR / "data" / "splits" / "train.jsonl"
DEFAULT_VAL_PATH = ROOT_DIR / "data" / "splits" / "val.jsonl"
MODELS_DIR = ROOT_DIR / "models"
MODEL_SAVE_PATH = MODELS_DIR / "attune_gate_best.pt"
CONFIG_SAVE_PATH = MODELS_DIR / "config.json"


def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu"):
    """Evaluates model on validation loader returning predictions, probabilities, and labels."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    temp = float(model.temperature.item())
    probs = 1.0 / (1.0 + np.exp(-logits / max(1e-3, temp)))
    preds = (probs >= 0.50).astype(int)

    metrics = compute_classification_metrics(labels.tolist(), preds.tolist())
    metrics["ntr"] = compute_ntr(labels.tolist(), preds.tolist())
    metrics["fsr"] = compute_fsr(labels.tolist(), preds.tolist())

    return metrics, probs, labels, logits


def train(
    train_path: Path = DEFAULT_TRAIN_PATH,
    val_path: Path = DEFAULT_VAL_PATH,
    batch_size: int = 16,
    epochs: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str = "cpu",
):
    print("=" * 80)
    print("ATTUNE PHASE 3: MODEL TRAINING & CALIBRATION PIPELINE")
    print("=" * 80)
    print(f"Device: {device} | Batch Size: {batch_size} | Max Length: {max_length}")
    print(f"Train File: {train_path} | Val File: {val_path}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Tokenizer & Datasets
    logger.info(f"Initializing tokenizer for '{DEFAULT_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

    train_dataset = AttuneDataset(train_path, tokenizer=tokenizer, max_length=max_length)
    val_dataset = AttuneDataset(val_path, tokenizer=tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 2. Instantiate Model & Optimizer
    logger.info("Instantiating AttuneGatingModel...")
    model = AttuneGatingModel(pretrained_model_name=DEFAULT_MODEL_NAME).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_model_state = None

    # 3. Training Loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_dataset)
        val_metrics, val_probs, val_labels, _ = evaluate(model, val_loader, device=device)

        logger.info(
            f"Epoch {epoch:2d}/{epochs} - Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"Val NTR: {val_metrics['ntr']:.4f} | Val FSR: {val_metrics['fsr']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time:.2f} seconds. Loading best model state...")
    model.load_state_dict(best_model_state)

    # 4. Calibration & Optimization
    print("\n" + "=" * 80)
    print("CALIBRATION & THRESHOLD OPTIMIZATION")
    print("=" * 80)

    # Pre-calibration ECE (T = 1.0)
    _, pre_probs, val_labels, _ = evaluate(model, val_loader, device=device)
    pre_ece = compute_ece(pre_probs, val_labels, n_bins=10)
    logger.info(f"Pre-Calibration ECE (T=1.000): {pre_ece:.4f}")

    # Temperature Scaling
    logger.info("Fitting optimal temperature parameter...")
    optimal_temp = fit_temperature(model, val_loader, device=device)

    # Post-calibration ECE
    _, post_probs, _, _ = evaluate(model, val_loader, device=device)
    post_ece = compute_ece(post_probs, val_labels, n_bins=10)
    logger.info(f"Post-Calibration ECE (T={optimal_temp:.4f}): {post_ece:.4f}")

    # Threshold Sweep
    logger.info("Sweeping decision thresholds for optimal tau* (target FSR <= 0.05)...")
    tau_star, best_threshold_metrics, sweep_history = find_optimal_threshold(
        post_probs,
        val_labels,
        target_fsr_max=0.05,
    )

    # 5. Save Artifacts
    logger.info(f"Saving best model weights to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    config_data = {
        "model_architecture": "AttuneGatingModel",
        "backbone_model": DEFAULT_MODEL_NAME,
        "max_length": max_length,
        "temperature": round(optimal_temp, 4),
        "tau_star": round(tau_star, 4),
        "target_fsr_max": 0.05,
        "pre_calibration_ece": round(pre_ece, 4),
        "post_calibration_ece": round(post_ece, 4),
        "optimal_threshold_metrics": best_threshold_metrics,
        "validation_records_count": len(val_dataset),
        "training_records_count": len(train_dataset),
        "training_time_seconds": round(total_training_time, 2),
    }

    with open(CONFIG_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Saved configuration and calibration metadata to {CONFIG_SAVE_PATH}")

    print("\n" + "=" * 80)
    print("ATTUNE PHASE 3 SUMMARY")
    print("=" * 80)
    print(f"Model saved to:               {MODEL_SAVE_PATH}")
    print(f"Config saved to:              {CONFIG_SAVE_PATH}")
    print(f"Optimal Temperature (T*):     {optimal_temp:.4f}")
    print(f"Pre-Calibration ECE:          {pre_ece:.4f}")
    print(f"Post-Calibration ECE:         {post_ece:.4f} (ECE reduced by {max(0, pre_ece - post_ece):.4f})")
    print(f"Optimal Decision Gate (tau*): {tau_star:.4f}")
    print(f"Metrics at tau*:              Accuracy: {best_threshold_metrics['accuracy']:.4f} | NTR: {best_threshold_metrics['ntr']:.4f} | FSR: {best_threshold_metrics['fsr']:.4f}")
    print("=" * 80 + "\n")

    return model, config_data


def main():
    parser = argparse.ArgumentParser(description="Train ATTUNE Gating Model.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
