"""
System D: Rule-Based Logic Gate Baseline
========================================
Operational Policy:
Executes a 2-tier deterministic heuristic:
1. Rule 1 (Golden Override Pre-Filter):
   If `recent_override` is not null / non-empty, the user has explicitly commanded a style change.
   Immediately returns 0 (suppress preference).
2. Rule 2 (Semantic Scope Match):
   Checks whether domain-specific keywords and conceptual vocabulary matching `preference_scope`
   are present in the active `current_query`.
   If a matching domain signature is detected, predicts applicability = 1; otherwise 0 (context shift).
"""

from typing import Any, Dict, List
import re

# Domain vocabulary signatures mapped from docs/spec.md
DOMAIN_VOCABULARY = {
    "algorithms_learning": {
        "algorithm", "complexity", "asymptotic", "recurrence", "dfs", "bfs", "tree", "graph",
        "dp", "knapsack", "sort", "quicksort", "pivot", "heap", "dijkstra", "avl", "rotation",
        "convex", "hull", "invariants", "partition", "automaton", "kmp", "search", "hash",
        "matrix", "strassen", "disjoint", "union", "red-black", "cuckoo", "big-o", "floyd",
        "kosaraju", "scc", "balance", "memoization", "prefix", "graham", "admissibility",
        "edmonds-karp", "flow", "master theorem"
    },
    "system_tradeoffs": {
        "system", "concurrency", "distributed", "cache", "mesi", "coherence", "deadlock",
        "mutex", "lock", "os", "paging", "virtual memory", "tlb", "tcp", "congestion",
        "lsm", "btree", "b-tree", "iops", "cap theorem", "pacelc", "epoll", "numa",
        "affinity", "ipc", "shared memory", "socket", "2pc", "two-phase", "branch",
        "speculation", "pipeline", "raft", "paxos", "quorum", "election", "split-brain",
        "reno", "cubic", "compaction", "multiplexing", "thread", "pinning"
    },
    "database_tools": {
        "database", "db", "sql", "postgres", "postgresql", "mysql", "innodb", "index",
        "scan", "docker", "container", "bind", "mount", "volume", "git", "rebase",
        "cherry-pick", "acid", "mvcc", "isolation", "k8s", "kubernetes", "pod", "redis",
        "eviction", "lru", "lfu", "nginx", "proxy", "load balancing", "terraform",
        "s3", "dynamodb", "kafka", "systemd", "journalctl", "curl", "tar", "archive",
        "heap scan", "dirty page", "checkpointing", "rebalance", "chown", "permissions"
    },
    "syntax_lookup": {
        "syntax", "rust", "borrow", "lifetime", "c++", "cpp", "sfinae", "concept",
        "enable_if", "python", "asyncio", "taskgroup", "typescript", "ts", "infer",
        "conditional", "go", "goroutine", "channel", "select", "bash", "parameter",
        "expansion", "flag", "enum", "thiserror", "anyhow", "function pointer", "callback",
        "java", "virtual thread", "loom", "descriptor", "__get__", "__set__", "kotlin",
        "coroutine", "window function", "dense_rank", "partition by", "aliasing"
    },
}


class RuleBasedBaseline:
    """Baseline D: Deterministic rule-based gate with Golden Override and Scope Match."""

    def __init__(self, name: str = "System D (Deterministic Rule Gate)"):
        self.name = name

    def predict(self, record: Dict[str, Any]) -> int:
        """
        Rule 1: Golden Override - If recent_override exists, return 0.
        Rule 2: Scope Match - Check if current_query matches preference_scope domain terms.
        """
        # Rule 1: Golden Override Pre-Filter
        override = record.get("recent_override")
        if override is not None and str(override).strip():
            return 0

        # Rule 2: Scope Match
        scope = record.get("preference_scope", "")
        query = record.get("current_query", "").lower()
        if not scope or not query:
            return 0

        # Tokenize query
        tokens = set(re.findall(r"\b[a-z0-9_-]+\b", query))
        scope_vocab = DOMAIN_VOCABULARY.get(scope, set())

        # Check for direct keyword overlap
        overlap = tokens.intersection(scope_vocab)
        if overlap:
            return 1

        # Also check for compound terms in text
        for term in scope_vocab:
            if " " in term and term in query:
                return 1

        return 0

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[int]:
        """Runs predictions sequentially across records."""
        return [self.predict(r) for r in records]


RuleBasedGate = RuleBasedBaseline
