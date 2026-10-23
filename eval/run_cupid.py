"""
Standalone Runner for CUPID External Academic Benchmark Evaluation.
"""

import sys
import time
from pathlib import Path
from tabulate import tabulate

sys.path.insert(0, ".")
from src.predict import AttuneGateInference
from eval.metrics import compute_classification_metrics, compute_ntr, compute_fsr, compute_rcr
from eval.evaluate_all import evaluate_system_e_on_external

CUPID_PATH = Path("data/external_benchmarks/cupid_test.jsonl")


def main():
    if not CUPID_PATH.exists():
        print(f"Error: {CUPID_PATH} not found.")
        sys.exit(1)

    print("=====================================================================================")
    print("ATTUNE: EVALUATION ON CUPID HELD-OUT BENCHMARK (KAIST / KIXLAB)")
    print("=====================================================================================")
    print(f"Benchmark File: {CUPID_PATH}")

    gate = AttuneGateInference()
    metrics = evaluate_system_e_on_external(gate, CUPID_PATH, "CUPID")

    table_data = [{
        "Benchmark": "CUPID ($N=756$)",
        "Scope": "Open-Domain Personas",
        "Accuracy": f"{metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)",
        "F1 Score": f"{metrics['f1']:.3f}",
        "NTR (Over-Pers.)": f"{metrics['ntr']:.3f} ({metrics['ntr']*100:.1f}%)",
        "Recall (Adherence)": f"{metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)",
        "FSR": f"{metrics['fsr']:.3f}",
        "Latency": f"{metrics['mean_latency_ms']} ms"
    }]

    print("\n" + tabulate(table_data, headers="keys", tablefmt="github") + "\n")
    print(f"Summary: Evaluated {metrics['num_records']} records. Negative transfer suppression: {(1-metrics['ntr'])*100:.1f}% (NTR: {metrics['ntr']*100:.1f}%), Adherence recall: {metrics['recall']*100:.1f}%.")


if __name__ == "__main__":
    main()
