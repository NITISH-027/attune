"""
ATTUNE Baseline Comparative Benchmark Runner
============================================
Evaluates all 4 comparative architectures under identical test conditions
using data/splits/val.jsonl (or test.jsonl) according to docs/spec.md:

1. System A: Zero-Shot Base (no memory, predicts 0)
2. System B: Fixed Profile (static memory, predicts 1)
3. System C: History Retrieval - RAG (MiniLM embedding cosine similarity >= 0.45)
4. System D: Rule-Based Logic Gate (Golden Override & Scope Keyword Match)

Computes:
- Negative Transfer Rate (NTR)
- False Suppression Rate (FSR)
- Repeated-Correction Rate (RCR)
- Classification Metrics (Accuracy, Precision, Recall, F1)
Generates Markdown table and saves results to eval/baseline_results.json.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from tabulate import tabulate

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.metrics import evaluate_system
from baselines.zero_shot import ZeroShotBaseline
from baselines.fixed_profile import FixedProfileBaseline
from baselines.history_rag import HistoryRAGBaseline
from baselines.rule_based import RuleBasedBaseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_benchmark")

DEFAULT_DATA_PATH = ROOT_DIR / "data" / "splits" / "val.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "eval" / "baseline_results.json"


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} evaluation records from {path}")
    return records


def run_benchmark(dataset_path: Path, output_path: Path):
    print("=" * 80)
    print("ATTUNE PHASE 2: COMPARATIVE BASELINE BENCHMARK EXECUTION")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")

    records = load_dataset(dataset_path)
    y_true = [r["gold_applicability"] for r in records]

    # Instantiate the 4 baseline systems
    systems = [
        ZeroShotBaseline(),
        FixedProfileBaseline(),
        HistoryRAGBaseline(threshold=0.45),
        RuleBasedBaseline(),
    ]

    results = []
    table_rows = []

    for system in systems:
        logger.info(f"Evaluating {system.name} on {len(records)} records...")
        t0 = time.time()
        y_pred = system.predict_batch(records)
        elapsed = time.time() - t0

        metrics = evaluate_system(records, y_pred, system_name=system.name)
        metrics["latency_seconds"] = round(elapsed, 4)
        metrics["latency_ms_per_query"] = round((elapsed / len(records)) * 1000, 2)
        results.append(metrics)

        table_rows.append({
            "Architecture": system.name,
            "Accuracy": f"{metrics['accuracy']:.3f}",
            "F1 Score": f"{metrics['f1']:.3f}",
            "NTR (Over-Personalization)": f"{metrics['ntr']:.3f}",
            "FSR (Under-Personalization)": f"{metrics['fsr']:.3f}",
            "RCR (Repeated Correction)": f"{metrics['rcr']:.3f}",
            "P99 / Turn Latency": f"{metrics['latency_ms_per_query']} ms",
        })

    # Display results table
    print("\n" + "=" * 80)
    print("COMPARATIVE BENCHMARK SCORECARD")
    print("=" * 80)
    print(tabulate(table_rows, headers="keys", tablefmt="github"))
    print("=" * 80 + "\n")

    # Save output JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_dataset": str(dataset_path),
                "num_records": len(records),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "systems": results,
            },
            f,
            indent=2,
        )

    logger.info(f"Successfully saved benchmark results to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ATTUNE comparative baseline benchmarks.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to evaluation JSONL split (defaults to data/splits/val.jsonl)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save benchmark JSON results (defaults to eval/baseline_results.json)",
    )
    args = parser.parse_args()

    run_benchmark(args.data_path, args.output_path)


if __name__ == "__main__":
    main()
