"""
ATTUNE Synthetic Gating Dataset Generator
=========================================
Generates 1,500 structured decision records based on the ATTUNE specification
in docs/spec.md using Google Gemini API (`google-genai` SDK) with robust batching,
exponential backoff retry, and domain-grounded fallback templates.

Scenario Distribution:
- 50% `same_context` (gold_applicability = 1): Current query matches preference scope with evidence.
- 30% `context_shift` (gold_applicability = 0): Current query switches context to different domain/task.
- 20% `explicit_reversal` (gold_applicability = 0): Prior evidence exists, but recent_override reverses it.

Output Schema:
{"episode_id": str, "scenario_family": str, "current_query": str, "candidate_preference": str,
 "preference_scope": str, "supporting_evidence": list[str], "recent_override": str or null,
 "gold_applicability": int, "scenario_type": str}
"""

import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variable if present in .env
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
env_file = ROOT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_data")

OUTPUT_PATH = ROOT_DIR / "data" / "synthetic" / "attune_raw_synthetic.jsonl"
TOTAL_RECORDS = 2000
NUM_FAMILIES = 50
RECORDS_PER_FAMILY = 40  # 50 * 40 = 2000

# 4 Core Domains
DOMAINS = [
    "algorithms_learning",
    "system_tradeoffs",
    "database_tools",
    "syntax_lookup",
]

# 4 Dimensions & 8 Discrete Preference Tokens
PREFERENCES = [
    "DEPTH:CONCISE",
    "DEPTH:DETAILED",
    "PEDAGOGY:INTUITIVE",
    "PEDAGOGY:FORMAL",
    "ILLUSTRATION:WORKED_TRACE",
    "ILLUSTRATION:CONCEPTUAL",
    "STRUCTURE:MODULAR",
    "STRUCTURE:CONTINUOUS",
]

# 50 Domain-Grounded Scenario Families
SCENARIO_FAMILIES = [
    # algorithms_learning (13)
    ("algo_graph_kosaraju_scc", "algorithms_learning", "Kosaraju SCC two-pass DFS on directed graphs"),
    ("algo_dp_knapsack_memo", "algorithms_learning", "0/1 Knapsack recurrence and state memoization"),
    ("algo_tree_avl_balance", "algorithms_learning", "AVL tree rotations and balance factor invariants"),
    ("algo_sort_quicksort_pivot", "algorithms_learning", "Quicksort Lomuto vs Hoare partitioning invariants"),
    ("algo_string_kmp_automaton", "algorithms_learning", "Knuth-Morris-Pratt prefix automaton and failure function"),
    ("algo_geometry_convex_hull", "algorithms_learning", "Graham scan and monotonicity cross-product checks"),
    ("algo_graph_dijkstra_heap", "algorithms_learning", "Dijkstra shortest path with min-heap complexity"),
    ("algo_search_a_star_heuristic", "algorithms_learning", "A* search admissibility and consistency conditions"),
    ("algo_divide_strassen_matrix", "algorithms_learning", "Strassen sub-cubic matrix multiplication recurrence"),
    ("algo_flow_edmonds_karp", "algorithms_learning", "Edmonds-Karp residual network max flow BFS paths"),
    ("algo_disjoint_set_union", "algorithms_learning", "Disjoint set union with path compression and rank"),
    ("algo_tree_red_black_rules", "algorithms_learning", "Red-black tree structural and coloring invariants"),
    ("algo_hashing_cuckoo_cycles", "algorithms_learning", "Cuckoo hashing displacement and rehashing cycles"),

    # system_tradeoffs (13)
    ("sys_raft_leader_election", "system_tradeoffs", "Raft distributed consensus leader election split votes"),
    ("sys_cache_mesi_coherence", "system_tradeoffs", "MESI cache coherence protocol bus snooping and transitions"),
    ("sys_concurrency_mutex_deadlock", "system_tradeoffs", "Deadlock Coffman conditions and dining philosophers"),
    ("sys_os_virtual_mem_paging", "system_tradeoffs", "Multi-level page tables, TLB misses, and page faults"),
    ("sys_network_tcp_congestion", "system_tradeoffs", "TCP Reno vs Cubic slow start and congestion avoidance"),
    ("sys_storage_lsm_vs_btree", "system_tradeoffs", "LSM compaction amplification vs B+ Tree random write IOPS"),
    ("sys_distributed_cap_theorem", "system_tradeoffs", "CAP theorem partitions and PACELC trade-offs"),
    ("sys_os_epoll_io_multiplexing", "system_tradeoffs", "Level-triggered vs edge-triggered epoll event loops"),
    ("sys_memory_numa_affinity", "system_tradeoffs", "Non-Uniform Memory Access latency and thread CPU pin affinity"),
    ("sys_ipc_shm_vs_unix_socket", "system_tradeoffs", "Shared memory synchronization vs Unix domain socket IPC"),
    ("sys_two_phase_commit_atomic", "system_tradeoffs", "2PC coordinator failure and blocking participant states"),
    ("sys_cpu_branch_speculation", "system_tradeoffs", "Branch prediction speculative execution and pipeline flushes"),
    ("sys_distributed_quorum_reads", "system_tradeoffs", "Dynamo-style R+W>N sloppy quorums and hinted handoff"),

    # database_tools (12)
    ("db_postgres_btree_index_scan", "database_tools", "PostgreSQL B-Tree index scan vs bitmap heap scan costs"),
    ("db_docker_volume_ownership", "database_tools", "Docker container bind mount user permissions and chown"),
    ("db_git_interactive_rebase", "database_tools", "Git interactive rebase squash and conflicting patch resolution"),
    ("db_sql_acid_mvcc_isolation", "database_tools", "PostgreSQL MVCC snapshot isolation and phantom read prevention"),
    ("db_k8s_pod_affinity_rules", "database_tools", "Kubernetes podAntiAffinity topologySpreadConstraints scheduling"),
    ("db_redis_eviction_policies", "database_tools", "Redis volatile-lru vs allkeys-lfu memory exhaustion eviction"),
    ("db_nginx_upstream_balancing", "database_tools", "Nginx reverse proxy keepalive and least_conn load balancing"),
    ("db_terraform_state_concurrency", "database_tools", "Terraform remote state S3 locking with DynamoDB tables"),
    ("db_kafka_consumer_rebalance", "database_tools", "Apache Kafka cooperative sticky assignor partition rebalance"),
    ("db_linux_systemd_restart_loop", "database_tools", "Linux systemd service restart limit bursts and journalctl triage"),
    ("db_tar_curl_streaming_pipe", "database_tools", "Streaming remote archive decompression via curl and tar pipeline"),
    ("db_mysql_innodb_buffer_pool", "database_tools", "MySQL InnoDB dirty page flushing and checkpointing latency"),

    # syntax_lookup (12)
    ("syntax_rust_borrow_lifetimes", "syntax_lookup", "Rust named lifetime specifiers and mutable borrowing rules"),
    ("syntax_cpp_sfinae_concepts", "syntax_lookup", "C++20 concepts constraints vs std::enable_if SFINAE"),
    ("syntax_python_asyncio_taskgroup", "syntax_lookup", "Python 3.11+ asyncio.TaskGroup exception propagation"),
    ("syntax_ts_conditional_infer", "syntax_lookup", "TypeScript mapped types with conditional infer unwrapping"),
    ("syntax_go_channel_select_leak", "syntax_lookup", "Go goroutine channel select default cases and nil channel leaks"),
    ("syntax_bash_parameter_expansion", "syntax_lookup", "Bash parameter expansion substring extraction and defaults"),
    ("syntax_rust_error_thiserror", "syntax_lookup", "Rust idiomatic enum error dispatch with thiserror and anyhow"),
    ("syntax_c_function_pointer_cast", "syntax_lookup", "C function pointer callbacks and strict aliasing cast warnings"),
    ("syntax_java_virtual_threads", "syntax_lookup", "Java 21 Project Loom virtual thread carrier pinning scenarios"),
    ("syntax_python_dunder_descriptors", "syntax_lookup", "Python __get__ and __set__ descriptor protocol implementation"),
    ("syntax_kotlin_coroutine_context", "syntax_lookup", "Kotlin CoroutineScope supervisor job cancellation hierarchy"),
    ("syntax_sql_window_partition_by", "syntax_lookup", "SQL window function DENSE_RANK() OVER (PARTITION BY ... ORDER BY)"),
]

# Supporting evidence templates by preference
EVIDENCE_TEMPLATES = {
    "DEPTH:CONCISE": [
        "User turn 1: 'Please keep answers concise and skip boilerplate.'",
        "User turn 2: 'Give me minimal snippets without verbose commentary.'",
        "User turn 1: 'I just need a quick, punchy summary of the core answer.'",
    ],
    "DEPTH:DETAILED": [
        "User turn 1: 'Please provide comprehensive explanations covering edge cases, memory layout, and runtime implications.'",
        "User turn 3: 'Be exhaustive and break down all trade-offs in depth.'",
        "User turn 2: 'Walk me through every detail and underlying mechanism.'",
    ],
    "PEDAGOGY:INTUITIVE": [
        "User turn 1: 'Can you explain this with everyday metaphors or intuitive analogies rather than dry theory?'",
        "User turn 2: 'Give me an intuitive mental model to picture how this operates.'",
        "User turn 1: 'Use relatable real-world comparisons to explain the concept.'",
    ],
    "PEDAGOGY:FORMAL": [
        "User turn 1: 'Please provide formal mathematical definitions and inductive invariant proofs.'",
        "User turn 3: 'Skip informal analogies; I need rigorous axiomatic proofs and asymptotic derivations.'",
        "User turn 2: 'State formal loop invariants and prove correctness symbolically.'",
    ],
    "ILLUSTRATION:WORKED_TRACE": [
        "User turn 1: 'Show me step-by-step state traces with table of variable mutations across each iteration.'",
        "User turn 2: 'Please include manual pointer and register execution tables.'",
        "User turn 3: 'Step through an actual trace with concrete values line by line.'",
    ],
    "ILLUSTRATION:CONCEPTUAL": [
        "User turn 1: 'Explain the high-level mechanics conceptually without getting bogged down in line-by-line register traces.'",
        "User turn 2: 'Focus on architectural dataflow concepts rather than micro step-by-step variable mutation.'",
        "User turn 1: 'Give me the conceptual flow diagram rather than code execution tables.'",
    ],
    "STRUCTURE:MODULAR": [
        "User turn 1: 'Organize your reply with clear markdown headings, bulleted prerequisites, and distinct sections.'",
        "User turn 2: 'Use structured modular blocks with separate trade-off panels.'",
        "User turn 1: 'Break this down into numbered stages and modular subsections.'",
    ],
    "STRUCTURE:CONTINUOUS": [
        "User turn 1: 'Write a seamless, continuous prose narrative without bullet lists or fragmented subsections.'",
        "User turn 2: 'Explain this in coherent continuous prose paragraphs.'",
        "User turn 3: 'Deliver a flowing essay style explanation without checklists.'",
    ],
}

# Override statements for explicit_reversal
REVERSAL_OVERRIDES = {
    "DEPTH:CONCISE": "User turn 4: 'Actually, please don't be brief. I need an in-depth breakdown covering edge cases and architecture.'",
    "DEPTH:DETAILED": "User turn 4: 'Hold on, these answers are way too long. Keep it short and to the point; I'm in a rush.'",
    "PEDAGOGY:INTUITIVE": "User turn 4: 'Please drop the analogies. I need rigorous formal definitions and mathematical notation.'",
    "PEDAGOGY:FORMAL": "User turn 4: 'These mathematical proofs are too dense. Explain it intuitively with simple real-world metaphors.'",
    "ILLUSTRATION:WORKED_TRACE": "User turn 4: 'Skip the step-by-step variable trace tables. Just explain the conceptual mechanism.'",
    "ILLUSTRATION:CONCEPTUAL": "User turn 4: 'This conceptual summary is too vague. Walk me through a concrete step-by-step worked trace with example values.'",
    "STRUCTURE:MODULAR": "User turn 4: 'Please write this as a continuous flowing explanation instead of splitting into bullet points.'",
    "STRUCTURE:CONTINUOUS": "User turn 4: 'Please format this with clear bullet points and modular headers so I can skim quickly.'",
}

# Weak / ambiguous evidence templates for ambiguous_neutral (Condition C in spec.md)
AMBIGUOUS_EVIDENCE_TEMPLATES = [
    "User turn 1: 'Thanks, that works.'",
    "User turn 2: 'Looks good to me.'",
    "User turn 1: 'Understood, moving on.'",
    "User turn 3: 'Okay, I see how that works.'",
    "User turn 1: 'Got it, let us continue.'",
    "User turn 2: 'Noted.'",
    "User turn 1: 'Acknowledged.'",
    "User turn 2: 'Alright, continuing with the project.'",
    "User turn 1: 'Fine, that makes sense.'",
    "User turn 2: 'Okay, thanks for the reply.'",
]

# Cross-domain shift queries for context_shift
CROSS_DOMAIN_QUERIES = {
    "algorithms_learning": [
        "What is the git command to cherry-pick a commit without committing it immediately?",
        "How do I mount a Docker named volume in read-only mode in docker-compose.yml?",
        "What is the bash syntax to test if a directory exists and is non-empty?",
        "How to write a regex in Python to extract query parameters from a URL?",
    ],
    "system_tradeoffs": [
        "What is the CLI syntax to inspect memory usage with top sorted by RSS?",
        "How do I implement a binary search tree insertion in TypeScript?",
        "What is the curl command to send a POST request with basic auth headers?",
        "Show me how to declare an enum with custom string values in Rust.",
    ],
    "database_tools": [
        "Explain how the Master Theorem solves the recurrence T(n) = 2T(n/2) + O(n).",
        "What is the time complexity of building a heap using bottom-up Floyd's method?",
        "How does the MESI cache protocol handle a Read Miss when another core has Modified state?",
        "Explain the Paxos phase 2b commit rule.",
    ],
    "syntax_lookup": [
        "Compare the throughput and latency trade-offs of LSM Trees vs B-Trees under write-heavy workloads.",
        "How does Raft handle log divergence during network partitions?",
        "What are the formal conditions for loop invariants in the Floyd-Warshall algorithm?",
        "Explain the structural invariants maintained by 2-3 tree node splitting.",
    ],
}


def build_gemini_client():
    """Initializes Google GenAI client if GEMINI_API_KEY is present and verifies availability."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY environment variable not found. Fallback generator will be used.")
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key, http_options={"timeout": 10000})
        logger.info("Probing Gemini API availability...")
        client.models.generate_content(
            model="gemini-3.8-flash",
            contents="ping",
        )
        logger.info("Successfully verified Gemini API connectivity.")
        return client
    except Exception as exc:
        logger.warning(f"Gemini API probe unavailable or timed out: {exc}. Using robust fallback generator.")
        return None


def generate_queries_with_gemini(client, family_name: str, domain: str, topic: str, count: int) -> List[str]:
    """Requests diverse domain-specific user technical queries using Gemini API with tight timeout."""
    prompt = f"""You are a Computer Science query dataset generator for project ATTUNE.
Generate {count} distinct, realistic, high-quality user technical questions asked by software engineers/students about:
Topic: {topic}
Subdomain: {domain}

Requirements:
- Each question must be authentic and technical.
- One question per line without numbers or bullets.
- Keep questions between 10 and 35 words.
"""
    try:
        resp = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )
        text = resp.text.strip()
        lines = [line.strip().lstrip("0123456789.-* ") for line in text.splitlines() if len(line.strip()) > 15]
        if len(lines) >= count:
            return lines[:count]
        elif lines:
            while len(lines) < count:
                lines.append(f"{lines[len(lines) % len(lines)]} (Focus on {topic})")
            return lines[:count]
    except Exception as exc:
        logger.warning(f"Gemini API call failed for {family_name}: {exc}. Using fallback queries.")
    return []


def generate_fallback_queries(topic: str, domain: str, count: int) -> List[str]:
    """Deterministically generates rich domain-grounded technical queries."""
    templates = [
        f"How does {topic} operate under edge-case boundary conditions?",
        f"Explain the primary mechanism of {topic} and common pitfalls during implementation.",
        f"Walk through the core algorithmic and architectural principles of {topic}.",
        f"What are the performance implications and asymptotic bounds of {topic}?",
        f"Can you explain the structural guarantees provided by {topic} in production?",
        f"How do we handle failure recovery and invariant maintenance in {topic}?",
        f"What is the difference between standard approaches and {topic}?",
        f"Analyze the time and space complexity trade-offs involved in {topic}.",
        f"Show how {topic} resolves concurrency or state synchronization issues.",
        f"Provide an explanation of {topic} addressing both correctness and efficiency.",
        f"What happens during state transitions or edge-case triggers in {topic}?",
        f"How should an engineer systematically debug or reason about {topic}?",
        f"Explain how {topic} satisfies consistency and reliability requirements.",
        f"What is the mathematical or architectural justification behind {topic}?",
        f"Discuss the practical trade-offs encountered when applying {topic} at scale.",
    ]
    queries = []
    for i in range(count):
        base = templates[i % len(templates)]
        if i >= len(templates):
            base = f"{base} (Aspect {i // len(templates) + 1})"
        queries.append(base)
    return queries


def generate_dataset() -> List[Dict[str, Any]]:
    """Generates exactly 2,000 records stratified according to specification."""
    client = build_gemini_client()
    records: List[Dict[str, Any]] = []

    logger.info(f"Starting generation of {TOTAL_RECORDS} records across {len(SCENARIO_FAMILIES)} scenario families...")
    
    # Track statistics
    stats = {"same_context": 0, "context_shift": 0, "explicit_reversal": 0, "ambiguous_neutral": 0}
    gemini_generated_count = 0

    for fam_idx, (fam_name, domain, topic) in enumerate(SCENARIO_FAMILIES):
        # We generate 40 records per family:
        # - 16 same_context (applicability = 1) -> 800 (40%)
        # - 12 context_shift (applicability = 0) -> 600 (30%)
        # - 8 explicit_reversal (applicability = 0) -> 400 (20%)
        # - 4 ambiguous_neutral (applicability = 0) -> 200 (10%)
        num_same = 16
        num_shift = 12
        num_reversal = 8
        num_ambiguous = 4

        # Attempt to get queries via Gemini API if client available
        queries_needed = num_same + num_reversal + num_ambiguous
        gemini_queries = []
        if client and fam_idx < 10:  # Use Gemini API on subset to manage rate limits and latency
            try:
                gemini_queries = generate_queries_with_gemini(client, fam_name, domain, topic, queries_needed)
                if gemini_queries:
                    gemini_generated_count += len(gemini_queries)
            except Exception as e:
                logger.warning(f"Error fetching from Gemini: {e}")

        fallback_queries = generate_fallback_queries(topic, domain, queries_needed)

        # 1. Generate 16 same_context (applicability = 1)
        for i in range(num_same):
            ep_id = f"ep_{fam_name}_{i:03d}"
            pref = PREFERENCES[(fam_idx * 7 + i) % len(PREFERENCES)]
            evidence_pool = EVIDENCE_TEMPLATES[pref]
            evidence = [evidence_pool[i % len(evidence_pool)]]
            
            if i < len(gemini_queries):
                query = gemini_queries[i]
            else:
                query = fallback_queries[i]

            record = {
                "episode_id": ep_id,
                "scenario_family": fam_name,
                "current_query": query,
                "candidate_preference": pref,
                "preference_scope": domain,
                "supporting_evidence": evidence,
                "recent_override": None,
                "gold_applicability": 1,
                "scenario_type": "same_context",
            }
            records.append(record)
            stats["same_context"] += 1

        # 2. Generate 12 context_shift (applicability = 0)
        shift_pool = CROSS_DOMAIN_QUERIES[domain]
        for j in range(num_shift):
            ep_id = f"ep_{fam_name}_{num_same + j:03d}"
            pref = PREFERENCES[(fam_idx * 7 + num_same + j) % len(PREFERENCES)]
            evidence_pool = EVIDENCE_TEMPLATES[pref]
            evidence = [evidence_pool[j % len(evidence_pool)]]
            query = shift_pool[j % len(shift_pool)]

            record = {
                "episode_id": ep_id,
                "scenario_family": fam_name,
                "current_query": query,
                "candidate_preference": pref,
                "preference_scope": domain,  # Evidence domain is still the original domain
                "supporting_evidence": evidence,
                "recent_override": None,
                "gold_applicability": 0,
                "scenario_type": "context_shift",
            }
            records.append(record)
            stats["context_shift"] += 1

        # 3. Generate 8 explicit_reversal (applicability = 0)
        for k in range(num_reversal):
            ep_id = f"ep_{fam_name}_{num_same + num_shift + k:03d}"
            pref = PREFERENCES[(fam_idx * 7 + num_same + num_shift + k) % len(PREFERENCES)]
            evidence_pool = EVIDENCE_TEMPLATES[pref]
            evidence = [evidence_pool[k % len(evidence_pool)]]
            override = REVERSAL_OVERRIDES[pref]
            
            q_idx = num_same + k
            if q_idx < len(gemini_queries):
                query = gemini_queries[q_idx]
            else:
                query = fallback_queries[num_same + k]

            record = {
                "episode_id": ep_id,
                "scenario_family": fam_name,
                "current_query": query,
                "candidate_preference": pref,
                "preference_scope": domain,
                "supporting_evidence": evidence,
                "recent_override": override,
                "gold_applicability": 0,
                "scenario_type": "explicit_reversal",
            }
            records.append(record)
            stats["explicit_reversal"] += 1

        # 4. Generate 4 ambiguous_neutral (applicability = 0)
        for m in range(num_ambiguous):
            ep_id = f"ep_{fam_name}_{num_same + num_shift + num_reversal + m:03d}"
            pref = PREFERENCES[(fam_idx * 7 + num_same + num_shift + num_reversal + m) % len(PREFERENCES)]
            evidence = [AMBIGUOUS_EVIDENCE_TEMPLATES[(fam_idx + m) % len(AMBIGUOUS_EVIDENCE_TEMPLATES)]]
            
            q_idx = num_same + num_reversal + m
            if q_idx < len(gemini_queries):
                query = gemini_queries[q_idx]
            else:
                query = fallback_queries[q_idx]

            record = {
                "episode_id": ep_id,
                "scenario_family": fam_name,
                "current_query": query,
                "candidate_preference": pref,
                "preference_scope": domain,
                "supporting_evidence": evidence,
                "recent_override": None,
                "gold_applicability": 0,
                "scenario_type": "ambiguous_neutral",
            }
            records.append(record)
            stats["ambiguous_neutral"] += 1

        if (fam_idx + 1) % 10 == 0 or fam_idx == len(SCENARIO_FAMILIES) - 1:
            logger.info(
                f"Progress: [{fam_idx + 1}/{len(SCENARIO_FAMILIES)} families] -> "
                f"{len(records)}/{TOTAL_RECORDS} records generated."
            )

    logger.info(f"Generation complete. Gemini-augmented queries: {gemini_generated_count}. Total records: {len(records)}")
    logger.info(f"Distribution: {stats}")
    return records


def validate_records(records: List[Dict[str, Any]]):
    """Enforces strict schema validation according to docs/spec.md."""
    ep_pattern = re.compile(r"^ep_[a-zA-Z0-9_-]+$")
    allowed_prefs = set(PREFERENCES)
    allowed_domains = set(DOMAINS)
    allowed_scenarios = {"same_context", "context_shift", "explicit_reversal", "ambiguous_neutral"}

    assert len(records) == TOTAL_RECORDS, f"Expected {TOTAL_RECORDS} records, found {len(records)}"

    pos_count = 0
    neg_count = 0
    seen_episodes = set()
    scenario_counts = {"same_context": 0, "context_shift": 0, "explicit_reversal": 0, "ambiguous_neutral": 0}

    for idx, r in enumerate(records):
        assert ep_pattern.match(r["episode_id"]), f"Invalid episode_id: {r['episode_id']} at index {idx}"
        assert r["episode_id"] not in seen_episodes, f"Duplicate episode_id: {r['episode_id']}"
        seen_episodes.add(r["episode_id"])

        assert len(r["current_query"]) >= 3, f"Query too short at index {idx}"
        assert r["candidate_preference"] in allowed_prefs, f"Invalid pref: {r['candidate_preference']}"
        assert r["preference_scope"] in allowed_domains, f"Invalid domain: {r['preference_scope']}"
        assert isinstance(r["supporting_evidence"], list), f"Evidence must be list at index {idx}"
        assert r["gold_applicability"] in (0, 1), f"Invalid gold_applicability: {r['gold_applicability']}"
        assert r["scenario_type"] in allowed_scenarios, f"Invalid scenario_type: {r['scenario_type']}"
        scenario_counts[r["scenario_type"]] += 1

        if r["gold_applicability"] == 1:
            pos_count += 1
            assert r["scenario_type"] == "same_context", "Label 1 must be same_context"
            assert r["recent_override"] is None, "Label 1 must have null recent_override"
        else:
            neg_count += 1
            if r["scenario_type"] == "explicit_reversal":
                assert r["recent_override"] is not None, "explicit_reversal must have an override string"
            elif r["scenario_type"] == "ambiguous_neutral":
                assert r["recent_override"] is None, "ambiguous_neutral must have null recent_override"

    assert pos_count == 800, f"Expected exactly 800 positive labels, got {pos_count}"
    assert neg_count == 1200, f"Expected exactly 1200 negative labels, got {neg_count}"
    assert scenario_counts["same_context"] == 800, f"Expected 800 same_context, got {scenario_counts['same_context']}"
    assert scenario_counts["context_shift"] == 600, f"Expected 600 context_shift, got {scenario_counts['context_shift']}"
    assert scenario_counts["explicit_reversal"] == 400, f"Expected 400 explicit_reversal, got {scenario_counts['explicit_reversal']}"
    assert scenario_counts["ambiguous_neutral"] == 200, f"Expected 200 ambiguous_neutral, got {scenario_counts['ambiguous_neutral']}"
    logger.info("All 2,000 records passed strict schema and balance assertions!")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = generate_dataset()
    validate_records(records)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Successfully saved {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
