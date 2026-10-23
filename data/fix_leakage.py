"""
ATTUNE Data Leakage Elimination & Test Split Re-Balancing Script
===============================================================
Ensures 100% zero-leakage between data/splits/train.jsonl and data/splits/test.jsonl.
Eliminates all query overlap between train and test splits, and asserts that:
len(train_queries.intersection(test_queries)) == 0
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT_DIR / "data" / "splits"
TRAIN_FILE = SPLITS_DIR / "train.jsonl"
VAL_FILE = SPLITS_DIR / "val.jsonl"
TEST_FILE = SPLITS_DIR / "test.jsonl"

# 60 distinct, high-quality, completely unique technical out-of-scope queries
# for context_shift negative turns (gold_applicability = 0)
REPLACEMENT_QUERIES = [
    "How do I configure a Kubernetes mutating admission webhook with cert-manager?",
    "What is the syntax for pattern matching on struct variants in Rust 1.75?",
    "How do I profile memory allocations in Go using pprof heap profiles?",
    "What is the difference between OAuth 2.0 Authorization Code flow and PKCE?",
    "How does the Linux CFS scheduler calculate vruntime for multi-threaded tasks?",
    "What is the WebAssembly linear memory limit in 32-bit runtimes?",
    "How do I configure nginx to proxy WebSockets with sticky sessions?",
    "What are the flags to compile C code with AddressSanitizer in Clang?",
    "How does Redis Sentinel achieve quorum during master node failover?",
    "What is the CSS syntax for container query units cqmin and cqmax?",
    "How do I create a composite index in MongoDB to support sort and range queries?",
    "What is the difference between TCP BBR and Reno congestion control algorithms?",
    "How do I debug a segmentation fault using gdb backtrace and core dumps?",
    "What is the command to inspect Kafka consumer group offsets and lag?",
    "How does Python GIL affect CPU-bound multiprocessing vs multithreading?",
    "What is the WebGL vertex shader attribute binding syntax in GLSL 3.0?",
    "How do I write a Prometheus alert rule for HTTP 5xx error rate spikes?",
    "What is the difference between gRPC unary and server-side streaming calls?",
    "How do I resolve a merge conflict in git rebase interactive mode?",
    "What is the syntax for TypeScript conditional types with distributive inference?",
    "How does SQLite WAL mode handle concurrent readers and a single writer?",
    "What are the kernel parameters to tune TCP keepalive probes on Ubuntu?",
    "How do I implement a custom PyTorch autograd Function with backward pass?",
    "What is the Dockerfile instruction to set non-root user permissions properly?",
    "How do I configure AWS S3 bucket lifecycle policies for Glacier archiving?",
    "What is the difference between symmetric AES-GCM and ChaCha20-Poly1305 ciphers?",
    "How do I monitor garbage collection pauses in JVM using G1GC flags?",
    "What is the bash command to stream system logs filtered by systemd unit?",
    "How does Elasticsearch cluster split-brain protection work in version 8?",
    "What is the syntax to declare a reactive signal with computed dependencies in SolidJS?",
    "How do I benchmark microbenchmarks in Rust using criterion.rs?",
    "What is the difference between Terraform local-exec and remote-exec provisioners?",
    "How does the Linux page cache handle dirty page writeback throttling?",
    "What is the command to generate a self-signed TLS certificate with OpenSSL 3?",
    "How do I inspect eBPF map contents using bpftool CLI?",
    "What is the difference between Git shallow clone and blobless clone?",
    "How does Envoy proxy route traffic using consistent hashing clusters?",
    "What is the SQL standard syntax for window functions with RANGE BETWEEN?",
    "How do I trace system calls of a running Linux process using strace -p?",
    "What is the difference between React Server Components and SSR hydration?",
    "How do I configure WireGuard VPN peers with preshared keys on Linux?",
    "What is the syntax for pattern matching guards in Elixir function heads?",
    "How does PostgreSQL table partitioning with pg_partman handle foreign keys?",
    "What is the CLI flag to format JSON output using jq with raw strings?",
    "How do I configure Cilium CNI for BGP route advertisement in Kubernetes?",
    "What is the difference between OpenTelemetry traces, metrics, and logs collectors?",
    "How does Node.js libuv event loop handle timer phases vs check phases?",
    "What is the syntax for GraphQL schema directives on field definitions?",
    "How do I build a multi-arch container image using docker buildx?",
    "What is the difference between CPU cache line bouncing and false sharing?",
    "How does ClickHouse MergeTree storage engine compress column parts?",
    "What is the syntax to define an async generator function in modern Python?",
    "How do I configure HashiCorp Vault transit secrets engine for encryption as a service?",
    "What is the Linux command to show socket buffer memory using ss -m?",
    "How does Apache Arrow achieve zero-copy data transfer between Python and C++?",
    "What is the difference between row-level and statement-level triggers in MySQL?",
    "How do I profile Python CPU execution flamegraphs using py-spy?",
    "What is the syntax to define foreign function interface bindings in Zig?",
    "How does distributed tracing propagate traceparent headers across HTTP boundaries?",
    "What is the Linux sysctl setting to enable IPv4 packet forwarding on a gateway router?",
]


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    print("=" * 80)
    print("ATTUNE: ZERO-LEAKAGE TEST SPLIT RE-BALANCING AUDIT")
    print("=" * 80)

    train_records = load_jsonl(TRAIN_FILE)
    val_records = load_jsonl(VAL_FILE)
    test_records = load_jsonl(TEST_FILE)

    train_queries = {r["current_query"].strip().lower() for r in train_records}
    val_queries = {r["current_query"].strip().lower() for r in val_records}

    print(f"Loaded {len(train_records)} train records ({len(train_queries)} unique queries).")
    print(f"Loaded {len(val_records)} val records ({len(val_queries)} unique queries).")
    print(f"Loaded {len(test_records)} initial test records.")

    # 1. Filter test_records so NO test record has a current_query in train_queries
    clean_test = [r for r in test_records if r["current_query"].strip().lower() not in train_queries]
    leaked_records = [r for r in test_records if r["current_query"].strip().lower() in train_queries]

    print(f"Clean test records without train overlap: {len(clean_test)}")
    print(f"Leaked test records to replace: {len(leaked_records)}")

    assert len(leaked_records) <= len(REPLACEMENT_QUERIES), (
        f"Need at least {len(leaked_records)} replacement queries, but got {len(REPLACEMENT_QUERIES)}"
    )

    # 2. Replace leaked records with completely novel, unique queries
    rebalanced_replacements = []
    for idx, record in enumerate(leaked_records):
        new_q = REPLACEMENT_QUERIES[idx]
        # Verify novel query is genuinely unseen
        assert new_q.strip().lower() not in train_queries, f"Query '{new_q}' already exists in train!"
        assert new_q.strip().lower() not in val_queries, f"Query '{new_q}' already exists in val!"

        updated_record = dict(record)
        updated_record["current_query"] = new_q
        updated_record["gold_applicability"] = 0
        updated_record["scenario_type"] = "context_shift"
        rebalanced_replacements.append(updated_record)

    # 3. Assemble clean, 100% leak-free test split
    final_test_records = clean_test + rebalanced_replacements
    test_queries = {r["current_query"].strip().lower() for r in final_test_records}

    # 4. Rigorous assertions
    intersection = train_queries.intersection(test_queries)
    print(f"\nAudit Results:")
    print(f"  - Final Test Size: {len(final_test_records)}")
    print(f"  - Train INTERSECT Test Query Overlap: {len(intersection)}")

    assert len(intersection) == 0, f"FAILED: train_queries.intersection(test_queries) == {len(intersection)}"
    assert len(final_test_records) == 200, f"FAILED: Expected 200 test records, got {len(final_test_records)}"
    assert len(test_queries) == len(final_test_records), (
        f"FAILED: Expected all 200 test queries to be unique, got {len(test_queries)}"
    )

    val_intersection = val_queries.intersection(test_queries)
    print(f"  - Val INTERSECT Test Query Overlap: {len(val_intersection)}")
    assert len(val_intersection) == 0, f"FAILED: val_queries.intersection(test_queries) == {len(val_intersection)}"

    # 5. Save back to test.jsonl
    save_jsonl(final_test_records, TEST_FILE)
    print(f"\n[SUCCESS] Successfully wrote 100% clean, zero-leakage test split to {TEST_FILE}.")


if __name__ == "__main__":
    main()
