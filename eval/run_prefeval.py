"""
Standalone Runner for PrefEval External Academic Benchmark Evaluation.
"""

import sys
import time
from pathlib import Path
from tabulate import tabulate

sys.path.insert(0, ".")
from src.predict import AttuneGateInference
from eval.metrics import compute_classification_metrics, compute_ntr, compute_fsr, compute_rcr
from eval.evaluate_all import evaluate_system_e_on_external

PREFEVAL_PATH = Path("data/external_benchmarks/prefeval_test.jsonl")


def main():
    if not PREFEVAL_PATH.exists():
        print(f"Error: {PREFEVAL_PATH} not found. Please run data/external_benchmarks/generate_unique_prefeval.py first.")
        sys.exit(1)

    print("=====================================================================================")
    print("ATTUNE: EVALUATION ON PREFEVAL HELD-OUT BENCHMARK (ICLR 2025 TASK SCHEMA)")
    print("=====================================================================================")
    print(f"Benchmark File: {PREFEVAL_PATH}")

    gate = AttuneGateInference()
    metrics = evaluate_system_e_on_external(gate, PREFEVAL_PATH, "PrefEval")

    table_data = [{
        "Benchmark": "PrefEval ($N=150$)",
        "Scope": "Technical & CS",
        "Accuracy": f"{metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)",
        "F1 Score": f"{metrics['f1']:.3f}",
        "NTR (Over-Pers.)": f"{metrics['ntr']:.3f} ({metrics['ntr']*100:.1f}%)",
        "Recall (Adherence)": f"{metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)",
        "FSR": f"{metrics['fsr']:.3f}",
        "Latency": f"{metrics['mean_latency_ms']} ms"
    }]

    print("\n" + tabulate(table_data, headers="keys", tablefmt="github") + "\n")
    print(f"Summary: Evaluated {metrics['num_records']} records. Adherence recall: {metrics['recall']*100:.1f}%, Negative transfer: {metrics['ntr']*100:.1f}%.")


if __name__ == "__main__":
    main()
