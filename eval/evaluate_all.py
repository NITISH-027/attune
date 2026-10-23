"""
ATTUNE Master Comparative Evaluation Benchmark
==============================================
Evaluates all 5 comparative architectures on the held-out test split
(data/splits/test.jsonl, N=200) under identical conditions:
  - System A (Zero-Shot Base): baselines.zero_shot.ZeroShotBase
  - System B (Fixed Profile): baselines.fixed_profile.FixedProfileBase
  - System C (History Retrieval - RAG): baselines.history_rag.HistoryRAGBase
  - System D (Deterministic Rule Gate): baselines.rule_based.RuleBasedGate
  - System E (ATTUNE Gating Model): src.predict.AttuneGateInference

Computes:
  - Accuracy
  - F1 Score
  - Negative Transfer Rate (NTR = FP / (FP + TN))
  - False Suppression Rate (FSR = FN / (FN + TP))
  - Repeated-Correction Rate (RCR on post-override turns)
  - Mean Inference Latency per turn (ms)

Also evaluates System E on held-out external benchmarks:
  - CUPID (data/external_benchmarks/cupid_test.jsonl)
  - PrefEval (data/external_benchmarks/prefeval_test.jsonl)

Saves full results to eval/final_scorecard.json and prints comparative Markdown tables.
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

from eval.metrics import evaluate_system, compute_classification_metrics, compute_ntr, compute_fsr, compute_rcr
from baselines.zero_shot import ZeroShotBase
from baselines.fixed_profile import FixedProfileBase
from baselines.history_rag import HistoryRAGBase
from baselines.rule_based import RuleBasedGate
from src.predict import AttuneGateInference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_all")

TEST_SPLIT_PATH = ROOT_DIR / "data" / "splits" / "test.jsonl"
CUPID_PATH = ROOT_DIR / "data" / "external_benchmarks" / "cupid_test.jsonl"
PREFEVAL_PATH = ROOT_DIR / "data" / "external_benchmarks" / "prefeval_test.jsonl"
SCORECARD_OUTPUT_PATH = ROOT_DIR / "eval" / "final_scorecard.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_system_e_on_test(gate: AttuneGateInference, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Runs System E on native test split records measuring latency and metrics."""
    preds = []
    latencies = []

    for r in records:
        t0 = time.time()
        res = gate.predict(
            current_query=r.get("current_query", ""),
            candidate_preference=r.get("candidate_preference", ""),
            preference_scope=r.get("preference_scope", ""),
            supporting_evidence=r.get("supporting_evidence", []),
            recent_override=r.get("recent_override"),
        )
        latencies.append((time.time() - t0) * 1000)
        preds.append(res["decision"])

    metrics = evaluate_system(records, preds, system_name="System E (ATTUNE Gating Model)")
    metrics["latency_ms_per_query"] = round(float(sum(latencies) / len(latencies)), 2)
    metrics["p99_latency_ms"] = round(float(sorted(latencies)[int(0.99 * len(latencies))]), 2)
    return metrics, preds


def evaluate_system_e_on_external(gate: AttuneGateInference, path: Path, benchmark_name: str) -> Dict[str, Any]:
    """Evaluates System E zero-shot transfer generalization on external benchmarks."""
    raw_records = load_jsonl(path)
    logger.info(f"Loaded {len(raw_records)} records from {benchmark_name} ({path})")

    preds = []
    golds = []
    latencies = []
    normalized_records = []

    for item in raw_records:
        if benchmark_name == "PrefEval":
            ctx = item.get("interaction_context", {})
            query = item.get("eval_query", "")
            pref = ctx.get("demonstrated_preference", "DEPTH:DETAILED")
            scope = ctx.get("domain", "algorithms_learning")
            evidence = ctx.get("history_turns", [])
            scenario_type = item.get("scenario_type", "")
            override = "Override requested" if scenario_type == "explicit_reversal" else None
            gold = int(item.get("gold_applicability", 0))
        elif benchmark_name == "CUPID":
            query = item.get("current_request", "")
            pref = item.get("current_contextual_preference", "DEPTH:CONCISE")
            scope = item.get("current_context_factor", "General")
            priors = item.get("prior_interactions", [])
            evidence = []
            for p in priors[:2]:
                for d in p.get("dialogue", [])[:2]:
                    evidence.append(f"{d.get('role')}: {d.get('content')[:120]}")
            itype = item.get("instance_type", "")
            override = "User changed style" if itype == "changing" else None
            gold = 1 if itype == "consistent" else 0
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        t0 = time.time()
        res = gate.predict(
            current_query=query,
            candidate_preference=pref,
            preference_scope=scope,
            supporting_evidence=evidence,
            recent_override=override,
        )
        latencies.append((time.time() - t0) * 1000)

        preds.append(res["decision"])
        golds.append(gold)
        normalized_records.append({
            "current_query": query,
            "candidate_preference": pref,
            "preference_scope": scope,
            "supporting_evidence": evidence,
            "recent_override": override,
            "gold_applicability": gold,
            "scenario_type": itype if benchmark_name == "CUPID" else scenario_type,
        })

    class_metrics = compute_classification_metrics(golds, preds)
    ntr = compute_ntr(golds, preds)
    fsr = compute_fsr(golds, preds)
    rcr = compute_rcr(normalized_records, y_pred=preds)

    return {
        "benchmark": benchmark_name,
        "num_records": len(raw_records),
        "accuracy": class_metrics["accuracy"],
        "precision": class_metrics["precision"],
        "recall": class_metrics["recall"],
        "f1": class_metrics["f1"],
        "ntr": ntr,
        "fsr": fsr,
        "rcr": rcr,
        "mean_latency_ms": round(float(sum(latencies) / len(latencies)), 2),
    }


def main():
    print("=" * 85)
    print("ATTUNE PHASE 4: MASTER COMPARATIVE BENCHMARK EVALUATION")
    print("=" * 85)
    print(f"Test Split File: {TEST_SPLIT_PATH}")

    test_records = load_jsonl(TEST_SPLIT_PATH)
    logger.info(f"Loaded {len(test_records)} held-out test tuples.")

    # 1. Evaluate Systems A - D
    baselines = [
        ZeroShotBase(),
        FixedProfileBase(),
        HistoryRAGBase(threshold=0.45),
        RuleBasedGate(),
    ]

    benchmark_results = []
    table_rows = []

    for b in baselines:
        logger.info(f"Running evaluation for {b.name}...")
        t0 = time.time()
        preds = b.predict_batch(test_records)
        elapsed = time.time() - t0
        metrics = evaluate_system(test_records, preds, system_name=b.name)
        metrics["latency_ms_per_query"] = round((elapsed / len(test_records)) * 1000, 2)
        benchmark_results.append(metrics)

        table_rows.append({
            "Architecture": b.name,
            "Accuracy": f"{metrics['accuracy']:.3f}",
            "F1 Score": f"{metrics['f1']:.3f}",
            "NTR (Over-Pers.)": f"{metrics['ntr']:.3f}",
            "FSR (Under-Pers.)": f"{metrics['fsr']:.3f}",
            "RCR (Correction)": f"{metrics['rcr']:.3f}",
            "Latency / Turn": f"{metrics['latency_ms_per_query']} ms",
        })

    # 2. Evaluate System E (ATTUNE Gating Model)
    logger.info("Initializing and evaluating System E (ATTUNE Gating Model)...")
    gate = AttuneGateInference()
    attune_metrics, attune_preds = evaluate_system_e_on_test(gate, test_records)
    benchmark_results.append(attune_metrics)

    table_rows.append({
        "Architecture": attune_metrics["system_name"],
        "Accuracy": f"{attune_metrics['accuracy']:.3f}",
        "F1 Score": f"{attune_metrics['f1']:.3f}",
        "NTR (Over-Pers.)": f"{attune_metrics['ntr']:.3f}",
        "FSR (Under-Pers.)": f"{attune_metrics['fsr']:.3f}",
        "RCR (Correction)": f"{attune_metrics['rcr']:.3f}",
        "Latency / Turn": f"{attune_metrics['latency_ms_per_query']} ms",
    })

    # Print Main Scorecard
    print("\n" + "=" * 85)
    print("FINAL COMPARATIVE BENCHMARK SCORECARD (HELD-OUT TEST SET)")
    print("=" * 85)
    print(tabulate(table_rows, headers="keys", tablefmt="github"))
    print("=" * 85 + "\n")

    # 3. Evaluate External Benchmarks (Generalization Check)
    external_scorecard = []
    external_rows = []

    if PREFEVAL_PATH.exists():
        logger.info("Evaluating System E on PrefEval held-out benchmark...")
        pe_metrics = evaluate_system_e_on_external(gate, PREFEVAL_PATH, "PrefEval")
        external_scorecard.append(pe_metrics)
        external_rows.append({
            "Benchmark": "PrefEval (CS Technical)",
            "Records": pe_metrics["num_records"],
            "Accuracy": f"{pe_metrics['accuracy']:.3f}",
            "F1 Score": f"{pe_metrics['f1']:.3f}",
            "NTR": f"{pe_metrics['ntr']:.3f}",
            "FSR": f"{pe_metrics['fsr']:.3f}",
            "RCR": f"{pe_metrics['rcr']:.3f}",
            "Latency": f"{pe_metrics['mean_latency_ms']} ms",
        })

    if CUPID_PATH.exists():
        logger.info("Evaluating System E on CUPID held-out benchmark...")
        cupid_metrics = evaluate_system_e_on_external(gate, CUPID_PATH, "CUPID")
        external_scorecard.append(cupid_metrics)
        external_rows.append({
            "Benchmark": "CUPID (Cross-Domain)",
            "Records": cupid_metrics["num_records"],
            "Accuracy": f"{cupid_metrics['accuracy']:.3f}",
            "F1 Score": f"{cupid_metrics['f1']:.3f}",
            "NTR": f"{cupid_metrics['ntr']:.3f}",
            "FSR": f"{cupid_metrics['fsr']:.3f}",
            "RCR": f"{cupid_metrics['rcr']:.3f}",
            "Latency": f"{cupid_metrics['mean_latency_ms']} ms",
        })

    if external_rows:
        print("\n" + "=" * 85)
        print("SYSTEM E (ATTUNE) ZERO-SHOT GENERALIZATION ON EXTERNAL BENCHMARKS")
        print("=" * 85)
        print(tabulate(external_rows, headers="keys", tablefmt="github"))
        print("=" * 85 + "\n")

    # 4. Save Final Scorecard JSON
    SCORECARD_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_output = {
        "metadata": {
            "test_split_path": str(TEST_SPLIT_PATH),
            "num_test_records": len(test_records),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tau_star": gate.tau_star,
            "temperature": gate.temperature,
        },
        "comparative_systems": benchmark_results,
        "external_benchmarks": external_scorecard,
    }

    with open(SCORECARD_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    logger.info(f"Saved master comparative scorecard to {SCORECARD_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
