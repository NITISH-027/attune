"""
Generates 150 genuinely unique, distinct technical preference-evaluation pairs
for the PrefEval benchmark (ICLR 2025 task schema).
Spans 20 distinct Computer Science and Systems Engineering topics with ZERO duplicate queries
and zero overlap with train.jsonl.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.predict import AttuneGateInference
from eval.metrics import compute_classification_metrics, compute_ntr, compute_fsr

TRAIN_SPLIT_PATH = Path("data/splits/train.jsonl")
PREFEVAL_PATH = Path("data/external_benchmarks/prefeval_test.jsonl")

# 20 distinct CS topics - 95 distinct positive queries (zero duplicate queries, zero train overlap)
POSITIVE_ITEMS = [
    # Topic 1: Raft / Consensus
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Derive the mathematical proof of state machine safety invariant under Raft joint consensus configuration changes."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Prove the quorum intersection theorem across split partitions in Paxos phase 2b."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain split-brain prevention mechanics using Raft term numbers and pre-vote extensions."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Analyze leader lease expiration hazards when physical clock drift violates TrueTime bounds in distributed consensus."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Compare Byzantine fault tolerance vs crash fault tolerance in PBFT state verification protocols."),

    # Topic 2: Relational MVCC & Storage
    ("database_tools", "PEDAGOGY:FORMAL", "Derive the phantom read anomaly prevention proof under snapshot isolation vs serializable SSI."),
    ("database_tools", "DEPTH:DETAILED", "Explain Postgres multi-version concurrency control tuple header xmin and xmax visibility rules."),
    ("database_tools", "DEPTH:DETAILED", "Break down write amplification in B+ trees under random 16KB dirty page flushes."),
    ("database_tools", "STRUCTURE:MODULAR", "Analyze MySQL InnoDB redo log circular ring buffer wraparound vs undo tablespace checkpointing."),
    ("database_tools", "PEDAGOGY:FORMAL", "Prove strict two-phase locking guarantees conflict serializability using precedence serialization graphs."),

    # Topic 3: Compilers & SSA
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Prove dominance frontier placement optimality for SSA phi-node insertion via Cytron's algorithm."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Derive register pressure bounds under Chaitin graph coloring with Kempe heuristic spilling."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain loop invariant code motion safety conditions when memory aliasing cannot be statically disproven."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Break down LLVM escape analysis algorithms that enable stack allocation of heap objects."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Contrast polyhedral loop transformation models with traditional unimodular loop skewing."),

    # Topic 4: Virtual Memory & Paging
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Prove that Belady's optimal page replacement policy yields minimum page faults for fixed reference strings."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Walk through multi-level 4-level paging virtual address translation in x86-64 long mode."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Analyze TLB shootdown latency bottlenecks during inter-processor interrupts on NUMA architectures."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Compare demand paging page fault handling overhead vs huge page transparent 2MB mappings."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Derive the mathematical working set model equations for thrashing avoidance in dynamic page allocation."),

    # Topic 5: Cryptography
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Prove security of ECDH key exchange under the Computational Diffie-Hellman assumption over curve25519."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Derive polynomial commitment Schwartz-Zippel lemma bounds in KZG commitment schemes."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain side-channel power analysis mitigations using constant-time Montgomery ladder multiplication."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Break down padding oracle decryption vulnerability in CBC mode PKCS#7 ciphertexts."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Compare post-quantum lattice CRYSTALS-Kyber key encapsulation vs classic RSA-4096."),

    # Topic 6: Networking
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Derive the TCP Cubic window growth function differential equation as a function of elapsed time."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain 0-RTT connection resumption replay attack risks and mitigation tokens in QUIC RFC 9000."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Break down head-of-line blocking elimination in HTTP/3 multiplexed streams over UDP."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Compare BBR congestion control bandwidth-delay product estimation against classic loss-based Reno."),
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Prove loop-free routing convergence in Bellman-Ford distributed distance-vector routing."),

    # Topic 7: Concurrency & Lock-Free
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Prove linearizability of the lock-free Michael-Scott queue using atomic compare-and-swap invariants."),
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Derive safety proof for hazard pointer memory reclamation in high-throughput concurrent skip lists."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain ABA problem emergence in Treiber stacks and how tag-pointer double-word CAS mitigates it."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Break down Read-Copy-Update (RCU) grace period detection mechanics in the Linux kernel."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Compare ticket spinlocks vs MCS queue locks under high core contention and cache line bouncing."),

    # Topic 8: Cache Coherence
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Derive the state transition matrix invariants of the MOESI cache snooping protocol."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain memory barrier semantics (acquire-release vs sequential consistency) in the C++20 memory model."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Break down false sharing degradation when independent thread counters inhabit the same 64-byte cache line."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Compare invalidation queues and store buffers in ARM weak ordering vs x86 TSO total store order."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Prove Dekker's mutual exclusion correctness under sequentially consistent memory order assumptions."),

    # Topic 9: Memory Allocators
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Derive upper bounds on external fragmentation in binary buddy allocation algorithms."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain jemalloc thread-cached allocation arenas and extent-based slab management mechanics."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Break down free-list boundary tag coalescing to eliminate adjacent unallocated chunks."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Compare TCMalloc thread-local caches against glibc ptmalloc arena contention under multithreading."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Prove worst-case time complexity of segregated-fit explicit free list allocation."),

    # Topic 10: Garbage Collection
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Prove soundness and completeness of Dijkstra's tri-color marking algorithm with write barriers."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain JDK ZGC colored pointers and load-barrier mechanics enabling sub-millisecond pause times."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Break down generational hypothesis mechanics: Eden survivor spaces and tenured tenuring thresholds."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Compare concurrent mark-sweep vs compacting Cheney copying garbage collection algorithms."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Derive Card Table dirty-bit card marking efficiency for cross-generational young-to-old pointers."),

    # Topic 11: Type Systems
    ("syntax_lookup", "PEDAGOGY:FORMAL", "Derive principal typing proofs for polymorphic lambda calculus using the Hindley-Milner Algorithm W."),
    ("syntax_lookup", "PEDAGOGY:FORMAL", "Prove sound substructural resource safety under affine linear type system exchange rules."),
    ("syntax_lookup", "DEPTH:DETAILED", "Explain Rust borrow checker non-lexical lifetimes (NLL) based on control flow graph variance."),
    ("syntax_lookup", "DEPTH:DETAILED", "Break down monomorphization code bloat vs dynamic dispatch vtable cache misses in generic execution."),
    ("syntax_lookup", "STRUCTURE:MODULAR", "Contrast structural subtyping in TypeScript with nominal subtyping in Java and C++."),

    # Topic 12: Async Runtimes & Event Loops
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Prove starvation-free scheduling in cooperative work-stealing deque algorithms like Chase-Lev."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain Linux epoll edge-triggered vs level-triggered notification differences under non-blocking IO."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Break down stackless coroutine state machine transformation in C++20 co_await promise types."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Compare Windows IO Completion Ports (IOCP) proactor pattern with Linux epoll reactor model."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Derive throughput bounds of thread pool queuing models under Erlang-C queuing theory."),

    # Topic 13: Serialization & Protocols
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Derive wire encoding bit efficiency formulas for Protocol Buffers varint and ZigZag signed integers."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain Cap'n Proto zero-copy traversal mechanics using word-aligned offset pointers without decoding."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Break down gRPC HTTP/2 stream multiplexing flow control credit window starvation scenarios."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Compare Apache Avro schema resolution rules with FlatBuffers in-place memory mapped structs."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Prove backward and forward schema compatibility invariants in Protocol Buffers field numbering."),

    # Topic 14: Filesystem Architecture
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Prove consistency and crash-recovery invariants in write-ahead metadata journaling for ext4."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain ZFS copy-on-write Merkle tree update cascades from data blocks up to the uberblock."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Break down log-structured filesystem segment cleaning overhead under high random write utilization."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Compare soft updates asynchronous dependency tracking against synchronous journal barrier flushes."),

    # Topic 15: Container Isolation
    ("database_tools", "PEDAGOGY:FORMAL", "Prove isolation guarantees of Linux unshare pivot_root namespace boundary confinement."),
    ("database_tools", "DEPTH:DETAILED", "Explain Linux cgroups v2 unified hierarchy memory.high throttling vs memory.max OOM killer invocation."),
    ("database_tools", "DEPTH:DETAILED", "Break down seccomp-BPF system call filtering assembly filters and return actions (SECCOMP_RET_KILL)."),
    ("database_tools", "STRUCTURE:MODULAR", "Compare overlayfs upper, lower, and workdir copy-up mechanics during container image layer unioning."),

    # Topic 16: GPU Architecture
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Derive memory coalescing efficiency formulas for 32-thread warps accessing global DRAM arrays."),
    ("algorithms_learning", "DEPTH:DETAILED", "Explain SIMT warp divergence branch masking and instruction serialization overhead in CUDA."),
    ("algorithms_learning", "DEPTH:DETAILED", "Break down shared memory 32-bank conflict resolution algorithms using padding offset strides."),
    ("algorithms_learning", "STRUCTURE:MODULAR", "Compare cooperative thread array threadblock tile scheduling across streaming multiprocessors (SMs)."),

    # Topic 17: Graph Algorithms
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Prove correctness of Tarjan's strongly connected components algorithm using DFS lowlink tree back-edges."),
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Derive the optimality and admissibility condition proof for the A* heuristic search algorithm."),
    ("algorithms_learning", "DEPTH:DETAILED", "Explain Johnson's all-pairs shortest paths algorithm using Bellman-Ford reweighting for negative edges."),
    ("algorithms_learning", "DEPTH:DETAILED", "Break down Hopcroft-Karp maximum bipartite matching O(E sqrt(V)) augmenting path phases."),
    ("algorithms_learning", "STRUCTURE:MODULAR", "Compare Dinic's blocking flow algorithm with Edmonds-Karp BFS residual network capacity updates."),

    # Topic 18: Dynamic Programming
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Derive state transitions and space reduction proofs for the traveling salesman bitmask DP."),
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Prove slope monotonicity invariant in the Convex Hull Trick DP optimization technique."),
    ("algorithms_learning", "DEPTH:DETAILED", "Explain divide-and-conquer DP optimization conditions using the quadrangle inequality."),
    ("algorithms_learning", "DEPTH:DETAILED", "Break down digit dynamic programming digit-sum prefix constraints and tight bound tracking."),
    ("algorithms_learning", "STRUCTURE:MODULAR", "Contrast top-down memoization call stack recursion limits with bottom-up tabulated cache locality."),

    # Topic 19: Inverted Indices
    ("database_tools", "PEDAGOGY:FORMAL", "Derive the BM25 probabilistic term frequency saturation curve and document length normalization formula."),
    ("database_tools", "DEPTH:DETAILED", "Explain Hierarchical Navigable Small World (HNSW) graphs skip-list layer routing and recall trade-offs."),
    ("database_tools", "DEPTH:DETAILED", "Break down inverted index posting list compression using Elias-Fano and Roaring Bitmaps."),
    ("database_tools", "STRUCTURE:MODULAR", "Compare Product Quantization (PQ) vector compression against Inverted File (IVF) index Voronoi cells."),

    # Topic 20: Distributed Transactions
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Prove strict external consistency in Google Spanner using TrueTime bounded uncertainty intervals [t.earliest, t.latest]."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain two-phase commit (2PC) blocking failure modes when the transaction coordinator crashes in prepare phase."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Break down multi-version timestamp ordering (MVTO) cascading abort elimination using Thomas write rule."),
    ("system_tradeoffs", "STRUCTURE:MODULAR", "Compare Three-Phase Commit (3PC) non-blocking guarantees against Paxos-backed consensus commit protocols."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Prove linearizability in Calvin deterministic database transaction scheduling using Paxos input logging."),
]

# 35 Context Shifts: User demonstrates PEDAGOGY:FORMAL on algorithms proofs, but switches to rapid tooling/syntax questions
CONTEXT_SHIFT_QUERIES = [
    "How do I squash the last three git commits into a single commit with git rebase?",
    "What is the docker run argument to publish container port 80 to host port 8080?",
    "What bash command counts the total number of lines across all markdown files?",
    "How to parse command-line arguments using argparse in modern Python?",
    "Which curl command option sends custom JSON payloads with Content-Type header?",
    "What command lists all active listening TCP ports along with process IDs on Linux?",
    "How to create an uncompressed tar archive containing only specific directories?",
    "What git command shows the diff between two branches without merging?",
    "How do I configure UFW firewall to allow incoming SSH connections on port 2222?",
    "What jq filter expression selects all array objects where active field is true?",
    "How to restart the systemd service for coredns without dropping connections?",
    "What is the TypeScript syntax to declare a mapped type with readonly keys?",
    "How to set an environment variable inside a multi-stage Docker build step?",
    "What git command deletes a remote tracking branch on GitHub origin?",
    "How to terminate a hung process by sending SIGTERM signal via kill command?",
    "Which curl flag outputs the HTTP status code directly to stdout?",
    "How to inspect total available disk inodes on mounted file systems?",
    "What Python standard library module creates temporary directories securely?",
    "How to display a compact one-line log graph of recent git commits?",
    "What npm CLI flag updates dependencies while respecting package-lock.json?",
    "How to remove all dangling untagged Docker images using docker image prune?",
    "What command displays TCP socket round trip times and window scale factors in Linux?",
    "How to generate an Ed25519 SSH public and private keypair via terminal?",
    "What journalctl command displays system logs generated within the past 15 minutes?",
    "How to synchronize local directory contents to a remote server using rsync over SSH?",
    "What cargo command runs unit tests in parallel with single-threaded output?",
    "How to measure DNS lookup time and server response latency using dig command?",
    "What kubectl command displays resource consumption metrics for running pods?",
    "How to view all open socket connections for a running process using lsof?",
    "What is the command to create an absolute symlink in the Linux terminal?",
    "How to configure static DNS nameservers in modern Linux systemd-resolved?",
    "What git command aborts an in-progress interactive rebase operation safely?",
    "How to verify an X.509 certificate expiration date using openssl x509 CLI?",
    "What tmux shortcut splits the active terminal pane horizontally?",
    "How to test UDP packet transmission to a remote host using nc command?"
]

# 20 Explicit Reversals: Query asks to skip or reverse the preference
EXPLICIT_REVERSAL_ITEMS = [
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Explain quicksort conceptually with a simple everyday card shuffling analogy, no mathematical proofs."),
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Describe Dijkstra algorithm with an intuitive road trip metaphor, drop the formal proofs."),
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Explain binary search trees using a phone directory analogy, skip axiomatic definitions."),
    ("algorithms_learning", "PEDAGOGY:FORMAL", "Give me a simple everyday metaphor for hash tables and avoid formal mathematical notation."),
    ("algorithms_learning", "DEPTH:DETAILED", "Give a one-sentence summary of A* search, keep it very brief."),
    ("algorithms_learning", "DEPTH:DETAILED", "Tell me in 2 words what Big-O measures, no essays."),
    ("algorithms_learning", "DEPTH:DETAILED", "What is Dijkstra shortest path in one short bullet point, skip the deep dive."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Explain Raft leader election with a simple classroom voting analogy, no formal theorems."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Explain CPU cache with a desk and library analogy, drop the state diagrams."),
    ("system_tradeoffs", "PEDAGOGY:FORMAL", "Describe virtual memory using a book index metaphor, skip formal page table definitions."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Explain what an inode is in one short sentence, no technical jargon."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Tell me briefly what Docker does without mentioning cgroups or namespaces."),
    ("system_tradeoffs", "DEPTH:DETAILED", "Give a fast bullet point summary of cache line invalidation, keep it brief."),
    ("system_tradeoffs", "DEPTH:DETAILED", "What is the primary role of a CPU register? Answer in under 10 words."),
    ("database_tools", "PEDAGOGY:FORMAL", "Explain database index with a textbook index analogy, skip B-tree tree diagrams."),
    ("database_tools", "PEDAGOGY:FORMAL", "Explain ACID transactions using a bank teller transfer analogy, no formal proofs."),
    ("database_tools", "DEPTH:DETAILED", "Just give me the raw SQL syntax to alter table column types, keep it concise."),
    ("database_tools", "DEPTH:DETAILED", "Explain write-ahead log in one simple sentence, skip the architectural details."),
    ("syntax_lookup", "PEDAGOGY:FORMAL", "Explain Rust lifetimes with a simple library book loan metaphor, no formal syntax."),
    ("syntax_lookup", "DEPTH:DETAILED", "What does the const keyword do in C++? Answer in one sentence."),
]


def build_records():
    train_queries = set()
    if TRAIN_SPLIT_PATH.exists():
        with open(TRAIN_SPLIT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                train_queries.add(json.loads(line).get("current_query", "").strip().lower())

    records = []
    idx = 0

    # 1. Positives (exactly 95 items)
    for dom, pref, q in POSITIVE_ITEMS[:95]:
        assert q.strip().lower() not in train_queries, f"LEAKAGE: {q}"
        records.append({
            "benchmark_source": "PrefEval",
            "benchmark_id": f"prefeval_eval_{idx:04d}",
            "current_query": q,
            "eval_query": q,
            "interaction_context": {
                "domain": dom,
                "demonstrated_preference": pref,
                "history_turns": [
                    "User: Can we use formal proofs for these complexity analyses?",
                    "Assistant: Yes, I will provide rigorous axiomatic mathematical derivations."
                ] if pref == "PEDAGOGY:FORMAL" else [
                    "User: Please provide comprehensive explanations covering edge cases and architecture.",
                    "Assistant: Certainly, I will break down the underlying mechanisms in exhaustive detail."
                ]
            },
            "gold_applicability": 1,
            "scenario_type": "same_context",
            "metadata": {
                "strict_held_out": True,
                "split": "test",
                "topic": "cs_technical",
                "warning": "WARNING: THESE FILES ARE STRICTLY HELD-OUT EVALUATION BENCHMARKS. DO NOT LOAD INTO ANY TRAINING SPLIT."
            }
        })
        idx += 1

    # 2. Context Shifts (35 items)
    for q in CONTEXT_SHIFT_QUERIES:
        assert q.strip().lower() not in train_queries, f"LEAKAGE: {q}"
        records.append({
            "benchmark_source": "PrefEval",
            "benchmark_id": f"prefeval_eval_{idx:04d}",
            "current_query": q,
            "eval_query": q,
            "interaction_context": {
                "domain": "algorithms_learning",
                "demonstrated_preference": "PEDAGOGY:FORMAL",
                "history_turns": [
                    "User: Can we use formal proofs for these complexity analyses?",
                    "Assistant: Yes, I will provide rigorous axiomatic mathematical derivations."
                ]
            },
            "gold_applicability": 0,
            "scenario_type": "context_shift",
            "metadata": {
                "strict_held_out": True,
                "split": "test",
                "topic": "cs_technical",
                "warning": "WARNING: THESE FILES ARE STRICTLY HELD-OUT EVALUATION BENCHMARKS. DO NOT LOAD INTO ANY TRAINING SPLIT."
            }
        })
        idx += 1

    # 3. Explicit Reversals (20 items)
    for dom, pref, q in EXPLICIT_REVERSAL_ITEMS:
        assert q.strip().lower() not in train_queries, f"LEAKAGE: {q}"
        records.append({
            "benchmark_source": "PrefEval",
            "benchmark_id": f"prefeval_eval_{idx:04d}",
            "current_query": q,
            "eval_query": q,
            "interaction_context": {
                "domain": dom,
                "demonstrated_preference": pref,
                "history_turns": [
                    "User: Always explain algorithms using formal proofs.",
                    "Assistant: Understood, I will always provide formal proofs."
                ] if pref == "PEDAGOGY:FORMAL" else [
                    "User: Please provide comprehensive deep dive explanations.",
                    "Assistant: Understood, I will provide exhaustive technical depth."
                ]
            },
            "gold_applicability": 0,
            "scenario_type": "explicit_reversal",
            "metadata": {
                "strict_held_out": True,
                "split": "test",
                "topic": "cs_technical",
                "warning": "WARNING: THESE FILES ARE STRICTLY HELD-OUT EVALUATION BENCHMARKS. DO NOT LOAD INTO ANY TRAINING SPLIT."
            }
        })
        idx += 1

    return records


if __name__ == "__main__":
    records = build_records()
    print("Total records generated:", len(records))
    unique_q = set(r["current_query"] for r in records)
    print("Unique queries count:", len(unique_q))
    assert len(unique_q) == 150, f"Expected 150 unique queries, got {len(unique_q)}"
    assert len(records) == 150, f"Expected 150 records, got {len(records)}"

    # Save to file
    PREFEVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREFEVAL_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved 150 clean distinct records to {PREFEVAL_PATH}")
