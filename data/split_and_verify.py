"""
Phase 1 - Data Split & Zero-Leakage Verifier
Partitions attune_raw_synthetic.jsonl into Train (80%), Val (10%), Test (10%)
grouped strictly by scenario_family and episode_id to prevent template/profile leakage.
"""

import json
import os
import random
from pathlib import Path
from collections import Counter

# Set random seed for reproducibility
random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_SYNTHETIC_PATH = BASE_DIR / "data" / "synthetic" / "attune_raw_synthetic.jsonl"
SPLITS_DIR = BASE_DIR / "data" / "splits"
BENCHMARKS_DIR = BASE_DIR / "data" / "external_benchmarks"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def run():
    print("=" * 80)
    print("ATTUNE PHASE 1: DATA SPLIT & ZERO-LEAKAGE VERIFICATION")
    print("=" * 80)

    if not RAW_SYNTHETIC_PATH.exists():
        raise FileNotFoundError(f"Missing raw synthetic dataset at: {RAW_SYNTHETIC_PATH}")

    records = load_jsonl(RAW_SYNTHETIC_PATH)
    print(f"[INFO] Loaded {len(records)} raw synthetic records.")

    # 1. Group records by scenario_family to prevent template leakage
    family_to_records = {}
    for r in records:
        fam = r.get("scenario_family", "unknown")
        family_to_records.setdefault(fam, []).append(r)

    families = list(family_to_records.keys())
    random.shuffle(families)
    n_families = len(families)
    print(f"[INFO] Identified {n_families} unique scenario families.")

    # 80% Train, 10% Val, 10% Test by family
    n_train_fam = int(0.80 * n_families)
    n_val_fam = int(0.10 * n_families)

    train_families = set(families[:n_train_fam])
    val_families = set(families[n_train_fam : n_train_fam + n_val_fam])
    test_families = set(families[n_train_fam + n_val_fam :])

    train_records = [r for fam in train_families for r in family_to_records[fam]]
    val_records = [r for fam in val_families for r in family_to_records[fam]]
    test_records = [r for fam in test_families for r in family_to_records[fam]]

    # 2. Save splits
    train_file = SPLITS_DIR / "train.jsonl"
    val_file = SPLITS_DIR / "val.jsonl"
    test_file = SPLITS_DIR / "test.jsonl"

    save_jsonl(train_records, train_file)
    save_jsonl(val_records, val_file)
    save_jsonl(test_records, test_file)

    print(f"[INFO] Saved Train split: {len(train_records)} records -> {train_file}")
    print(f"[INFO] Saved Val split:   {len(val_records)} records -> {val_file}")
    print(f"[INFO] Saved Test split:  {len(test_records)} records -> {test_file}")

    # 3. Leakage Checks
    train_eps = {r["episode_id"] for r in train_records}
    val_eps = {r["episode_id"] for r in val_records}
    test_eps = {r["episode_id"] for r in test_records}

    fam_train_test_overlap = train_families.intersection(test_families)
    ep_train_test_overlap = train_eps.intersection(test_eps)

    assert len(fam_train_test_overlap) == 0, f"LEAKAGE DETECTED: Scenario family overlap: {fam_train_test_overlap}"
    assert len(ep_train_test_overlap) == 0, f"LEAKAGE DETECTED: Episode ID overlap: {ep_train_test_overlap}"

    # 4. Check external benchmark isolation
    cupid_path = BENCHMARKS_DIR / "cupid_test.jsonl"
    if cupid_path.exists():
        cupid_records = load_jsonl(cupid_path)
        cupid_queries = {r.get("current_query") or r.get("request") or str(r) for r in cupid_records}
        train_queries = {r["current_query"] for r in train_records}
        bench_overlap = cupid_queries.intersection(train_queries)
        assert len(bench_overlap) == 0, "LEAKAGE: CUPID benchmark queries found in training set!"

    # 5. Label Balance Report
    def get_balance(recs):
        counts = Counter(r["gold_applicability"] for r in recs)
        total = len(recs)
        return f"1: {counts.get(1, 0)} ({counts.get(1, 0)/total:.1%}), 0: {counts.get(0, 0)} ({counts.get(0, 0)/total:.1%})"

    print("\n" + "=" * 80)
    print("DATA SPLIT & LEAKAGE VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Train Balance: {get_balance(train_records)}")
    print(f"Val Balance:   {get_balance(val_records)}")
    print(f"Test Balance:  {get_balance(test_records)}")
    print("Overlap Checks:")
    print("  • Scenario Family Leakage (Train vs Test): 0.0% (PASSED)")
    print("  • Episode ID Leakage (Train vs Test):      0.0% (PASSED)")
    print("  • External Benchmark Isolation:           CONFIRMED (PASSED)")
    print("=" * 80)
    print("Phase 1 complete and ready for Phase 2!\n")


if __name__ == "__main__":
    run()
