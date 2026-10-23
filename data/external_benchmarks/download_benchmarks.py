"""
========================================================================================
WARNING: THESE FILES ARE STRICTLY HELD-OUT EVALUATION BENCHMARKS.
DO NOT LOAD INTO ANY TRAINING SPLIT.
========================================================================================

ATTUNE Benchmark Acquisition Script:
Downloads or generates sample evaluation records from held-out external evaluation
benchmarks (CUPID and PrefEval) for evaluating preference gating and negative transfer.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_benchmarks")

BENCHMARK_DIR = Path(__file__).resolve().parent
CUPID_OUTPUT_PATH = BENCHMARK_DIR / "cupid_test.jsonl"
PREFEVAL_OUTPUT_PATH = BENCHMARK_DIR / "prefeval_test.jsonl"

STRICT_WARNING_BANNER = (
    "WARNING: THESE FILES ARE STRICTLY HELD-OUT EVALUATION BENCHMARKS. "
    "DO NOT LOAD INTO ANY TRAINING SPLIT."
)


def get_mock_cupid_records() -> List[Dict[str, Any]]:
    """Generates synthetic held-out CUPID test samples if external network is unavailable."""
    logger.info("Using representative mock loader for CUPID evaluation benchmark...")
    return [
        {
            "benchmark_source": "CUPID",
            "benchmark_id": f"cupid_eval_{i:04d}",
            "conversation_history": [
                {"role": "user", "content": "I prefer concise code without excessive comments."},
                {"role": "assistant", "content": "Understood. I will keep code implementations minimal."},
            ],
            "target_query": query,
            "latent_preference": pref,
            "held_out_category": cat,
            "gold_preference_active": active,
            "metadata": {
                "strict_held_out": True,
                "split": "test",
                "warning": STRICT_WARNING_BANNER,
            },
        }
        for i, (query, pref, cat, active) in enumerate(
            [
                (
                    "How do I reverse a singly linked list in Python?",
                    "DEPTH:CONCISE",
                    "algorithms_learning",
                    1,
                ),
                (
                    "Explain the full architectural trade-offs of microservices vs monoliths.",
                    "DEPTH:CONCISE",
                    "system_tradeoffs",
                    0,
                ),
                (
                    "Give me the regex to match an IPv4 address.",
                    "DEPTH:CONCISE",
                    "syntax_lookup",
                    1,
                ),
                (
                    "Can you walk through an in-depth proof of the CAP theorem?",
                    "DEPTH:CONCISE",
                    "system_tradeoffs",
                    0,
                ),
                (
                    "What is the git command to discard all unstaged local changes?",
                    "DEPTH:CONCISE",
                    "database_tools",
                    1,
                ),
                (
                    "Write an end-to-end tutorial with step-by-step traces for Raft consensus.",
                    "DEPTH:CONCISE",
                    "system_tradeoffs",
                    0,
                ),
            ]
            * 25
        )  # 150 records
    ]


def get_mock_prefeval_records() -> List[Dict[str, Any]]:
    """Generates synthetic held-out PrefEval test samples if external network is unavailable."""
    logger.info("Using representative mock loader for PrefEval evaluation benchmark...")
    return [
        {
            "benchmark_source": "PrefEval",
            "benchmark_id": f"prefeval_eval_{i:04d}",
            "interaction_context": {
                "domain": domain,
                "demonstrated_preference": pref,
                "history_turns": [
                    "User: Can we use formal proofs for these complexity analyses?",
                    "Assistant: Yes, I will provide rigorous axiomatic mathematical derivations.",
                ],
            },
            "eval_query": query,
            "gold_applicability": gold_app,
            "scenario_type": scen,
            "metadata": {
                "strict_held_out": True,
                "split": "test",
                "warning": STRICT_WARNING_BANNER,
            },
        }
        for i, (domain, pref, query, gold_app, scen) in enumerate(
            [
                (
                    "algorithms_learning",
                    "PEDAGOGY:FORMAL",
                    "Derive the recurrence relation for Strassen matrix multiplication using Master Theorem.",
                    1,
                    "same_context",
                ),
                (
                    "database_tools",
                    "PEDAGOGY:FORMAL",
                    "What CLI flag starts Postgres in the background with Docker?",
                    0,
                    "context_shift",
                ),
                (
                    "system_tradeoffs",
                    "PEDAGOGY:FORMAL",
                    "Prove mutual exclusion safety invariant in Peterson's algorithm.",
                    1,
                    "same_context",
                ),
                (
                    "syntax_lookup",
                    "PEDAGOGY:FORMAL",
                    "How to split a string by delimiter in Bash?",
                    0,
                    "context_shift",
                ),
                (
                    "algorithms_learning",
                    "PEDAGOGY:FORMAL",
                    "Explain quicksort conceptually with a simple everyday card shuffling analogy.",
                    0,
                    "explicit_reversal",
                ),
            ]
            * 30
        )  # 150 records
    ]


def download_or_load_cupid() -> List[Dict[str, Any]]:
    """Attempts to download CUPID from Hugging Face or falls back to mock loader."""
    try:
        from datasets import load_dataset

        logger.info("Attempting to load 'kixlab/cupid' test split from Hugging Face...")
        ds = load_dataset("kixlab/cupid", split="test")
        records = []
        for i, row in enumerate(ds):
            record = dict(row)
            record["_held_out_notice"] = STRICT_WARNING_BANNER
            records.append(record)
        logger.info(f"Successfully loaded {len(records)} records from HF 'kixlab/cupid'.")
        return records
    except Exception as exc:
        logger.warning(f"Could not load Hugging Face dataset 'kixlab/cupid' ({exc}). Falling back to mock dataset.")
        return get_mock_cupid_records()


def download_or_load_prefeval() -> List[Dict[str, Any]]:
    """Attempts to download PrefEval from Hugging Face or falls back to mock loader."""
    try:
        from datasets import load_dataset

        logger.info("Attempting to load 'prefeval' or similar dataset from Hugging Face...")
        # Try known identifiers or public mirrors
        for candidate in ["prefeval", "allenai/prefeval", "tasksource/prefeval"]:
            try:
                ds = load_dataset(candidate, split="test")
                records = [dict(row) for row in ds]
                for r in records:
                    r["_held_out_notice"] = STRICT_WARNING_BANNER
                logger.info(f"Successfully loaded {len(records)} records from HF '{candidate}'.")
                return records
            except Exception:
                continue
        raise RuntimeError("No matching Hugging Face dataset found for PrefEval.")
    except Exception as exc:
        logger.warning(f"Could not load Hugging Face dataset for PrefEval ({exc}). Falling back to mock dataset.")
        return get_mock_prefeval_records()


def save_jsonl(records: List[Dict[str, Any]], output_path: Path):
    """Saves records to JSONL file with assertion check."""
    assert output_path.parent == BENCHMARK_DIR, (
        f"Security violation: {output_path} is outside {BENCHMARK_DIR}!"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} records to {output_path}")


def main():
    print("=" * 80)
    print(STRICT_WARNING_BANNER)
    print("=" * 80)

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    cupid_data = download_or_load_cupid()
    save_jsonl(cupid_data, CUPID_OUTPUT_PATH)

    prefeval_data = download_or_load_prefeval()
    save_jsonl(prefeval_data, PREFEVAL_OUTPUT_PATH)

    print("\nBenchmark download completed successfully.")
    print(f"1. CUPID:    {CUPID_OUTPUT_PATH} ({len(cupid_data)} rows)")
    print(f"2. PrefEval: {PREFEVAL_OUTPUT_PATH} ({len(prefeval_data)} rows)")


if __name__ == "__main__":
    main()
