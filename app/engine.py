"""
ATTUNE Runtime Engine
=====================
Orchestrates the 4 core components for real-time preference gating:
1. Preference Memory Store: Multi-session state tracker for active candidate
   preferences, domain scopes, supporting evidence turns, and recent overrides.
2. Signal Extractor: Parses user utterances for preference declarations,
   domain affinities, or explicit overrides (heuristic + optional GenAI).
3. ATTUNE Gating Model: Evaluates calibrated probability p_hat against tau* = 0.9000
   via `src.predict.AttuneGateInference`.
4. Response Generator: Emits style-conditioned CS explanations if gated
   p_hat >= tau*, or emits a balanced neutral CS response if suppressed.
"""

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import AttuneGateInference

logger = logging.getLogger("attune_engine")

# Load environment variable if present in .env
ROOT_DIR = Path(__file__).resolve().parent.parent
env_file = ROOT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# -----------------------------------------------------------------------------
# 1. Preference Memory Store
# -----------------------------------------------------------------------------

@dataclass
class SessionState:
    session_id: str
    candidate_preference: Optional[str] = "DEPTH:CONCISE"
    preference_scope: Optional[str] = "algorithms_learning"
    supporting_evidence: List[str] = field(default_factory=lambda: [
        "User feedback: 'Keep algorithm explanations concise with minimal boilerplate.'"
    ])
    recent_override: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self):
        self.candidate_preference = "DEPTH:CONCISE"
        self.preference_scope = "algorithms_learning"
        self.supporting_evidence = [
            "User feedback: 'Keep algorithm explanations concise with minimal boilerplate.'"
        ]
        self.recent_override = None
        self.history = []

    def set_preference(self, preference: str, scope: str, evidence: str):
        self.candidate_preference = preference
        self.preference_scope = scope
        if evidence and evidence not in self.supporting_evidence:
            self.supporting_evidence.append(evidence)

    def trigger_override(self, override_statement: str):
        self.recent_override = override_statement


class PreferenceMemoryStore:
    """In-memory multi-session store for candidate preferences and overrides."""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> SessionState:
        session = self.get_session(session_id)
        session.reset()
        return session


# -----------------------------------------------------------------------------
# 2. Signal Extractor
# -----------------------------------------------------------------------------

PREFERENCE_DESCRIPTIONS = {
    "DEPTH:CONCISE": "Provide a punchy, ultra-concise answer with minimal code and no boilerplate commentary.",
    "DEPTH:DETAILED": "Provide an exhaustive technical deep-dive covering memory layout, edge cases, and runtime implications.",
    "PEDAGOGY:INTUITIVE": "Explain using an intuitive, real-world metaphor or mental model to illuminate the mechanics.",
    "PEDAGOGY:FORMAL": "Provide rigorous formal definitions, invariant statements, and asymptotic proofs.",
    "ILLUSTRATION:WORKED_TRACE": "Include a concrete step-by-step execution trace table with variable mutations.",
    "ILLUSTRATION:CONCEPTUAL": "Provide a high-level conceptual dataflow explanation avoiding micro-step code traces.",
    "STRUCTURE:MODULAR": "Organize into distinct modular blocks with clear component boundaries and trade-off panels.",
    "STRUCTURE:CONTINUOUS": "Present as a seamless, continuous technical prose essay without bullet points.",
}

DOMAIN_KEYWORDS = {
    "algorithms_learning": [
        "algorithm", "sort", "quicksort", "mergesort", "avl", "tree", "bst", "heap",
        "dijkstra", "graph", "dynamic programming", "knapsack", "recurrence", "complexity",
        "big o", "asymptotic", "binary search", "traversal", "memoization", "data structure"
    ],
    "database_tools": [
        "git", "commit", "rebase", "cherry-pick", "docker", "volume", "container", "postgres",
        "sql", "index", "b-tree", "redis", "kubernetes", "k8s", "nginx", "terraform",
        "kafka", "systemd", "linux", "bash", "mvcc", "buffer pool"
    ],
    "system_tradeoffs": [
        "tradeoff", "distributed", "raft", "paxos", "consensus", "cache coherence", "mesi",
        "lsm tree", "write-heavy", "read-heavy", "replication", "partition", "cap theorem",
        "eventual consistency", "high availability", "throughput", "latency"
    ],
    "syntax_lookup": [
        "syntax", "rust", "lifetime", "borrow", "c++", "sfinae", "concept", "python",
        "asyncio", "taskgroup", "typescript", "infer", "golang", "channel", "goroutine",
        "descriptor", "dunder", "macro", "type error"
    ],
}

OVERRIDE_PATTERNS = [
    r"\b(ignore|forget|disregard|scratch|drop|cancel)\b.*\b(preference|concise|detailed|analogy|previous|earlier)\b",
    r"\bactually\b.*\b(give me|provide|be|switch to|instead|rather than|stop|detailed|exhaustive)\b",
    r"\bstop (being|using|doing)\b",
    r"\bdon'?t (be|use|give)\b.*\b(concise|brief|detailed|analogy|formal|bullet)\b",
    r"\bno more (concise|detailed|analogies|bullets|traces)\b",
    r"\bswitch to (detailed|exhaustive|formal)\b",
]

PREFERENCE_PATTERNS = [
    (r"\b(concise|brief|minimal boilerplate|short snippets|quick answer)\b", "DEPTH:CONCISE"),
    (r"\b(detailed|exhaustive|deep[- ]dive|in[- ]depth|comprehensive|all mechanisms)\b", "DEPTH:DETAILED"),
    (r"\b(intuitive|metaphor|analogy|mental model|relatable|real-world)\b", "PEDAGOGY:INTUITIVE"),
    (r"\b(formal|mathematical|rigorous|proof|invariant|axiomatic)\b", "PEDAGOGY:FORMAL"),
    (r"\b(step[- ]by[- ]step|trace|table|variable mutation|execution table)\b", "ILLUSTRATION:WORKED_TRACE"),
    (r"\b(conceptual|high[- ]level|architectural dataflow|no code trace)\b", "ILLUSTRATION:CONCEPTUAL"),
    (r"\b(modular|blocks|separate panels|sections|structured)\b", "STRUCTURE:MODULAR"),
    (r"\b(continuous prose|flowing narrative|no bullets|essay style)\b", "STRUCTURE:CONTINUOUS"),
]


class SignalExtractor:
    """Extracts preference declarations, explicit overrides, and domain scope."""

    def __init__(self, gemini_client=None):
        self.gemini_client = gemini_client

    def detect_scope(self, text: str) -> str:
        text_lower = text.lower()
        scores = {d: 0 for d in DOMAIN_KEYWORDS}
        for domain, kws in DOMAIN_KEYWORDS.items():
            for kw in kws:
                if kw in text_lower:
                    scores[domain] += 1
        best_domain = max(scores, key=scores.get)
        if scores[best_domain] > 0:
            return best_domain
        return "algorithms_learning"

    def analyze_message(self, message: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        msg_lower = message.lower()

        # 1. Check for explicit overrides
        detected_override = None
        for pattern in OVERRIDE_PATTERNS:
            if re.search(pattern, msg_lower):
                detected_override = message.strip()
                break

        # 2. Check for preference declarations
        detected_preference = None
        for pattern, pref_token in PREFERENCE_PATTERNS:
            if re.search(pattern, msg_lower):
                detected_preference = pref_token
                break

        # 3. Detect scope
        detected_scope = self.detect_scope(message)

        return detected_override, detected_preference, detected_scope


# -----------------------------------------------------------------------------
# 3. Dynamic Technical Response Generator
# -----------------------------------------------------------------------------

class ResponseGenerator:
    """Generates CS technical explanations conditioned on gating decisions."""

    def __init__(self, gemini_client=None):
        self.gemini_client = gemini_client

    def generate(
        self,
        query: str,
        applied: bool,
        candidate_preference: Optional[str] = None,
        reason: str = "",
    ) -> str:
        # Fallback domain-grounded response
        return self._domain_fallback_generate(query, applied, candidate_preference)

    def _domain_fallback_generate(
        self,
        query: str,
        applied: bool,
        candidate_preference: Optional[str],
    ) -> str:
        q_lower = query.lower()

        # Scenario 1: Git Cherry-Pick (database_tools)
        if "cherry-pick" in q_lower or "git" in q_lower:
            if applied and candidate_preference == "DEPTH:CONCISE":
                return (
                    "```bash\n"
                    "git cherry-pick -n <commit-hash>\n"
                    "```\n"
                    "The `-n` (or `--no-commit`) flag stages the cherry-picked changes directly without generating a commit object."
                )
            return (
                "To cherry-pick a commit without immediately committing it, pass the `-n` (or `--no-commit`) flag:\n\n"
                "```bash\n"
                "git cherry-pick --no-commit <commit-hash>\n"
                "# or using the short option:\n"
                "git cherry-pick -n <commit-hash>\n"
                "```\n\n"
                "### Mechanics & Verification Workflow:\n"
                "1. **Staged Changes**: Git extracts the patch from `<commit-hash>` and applies it directly to your index and working tree.\n"
                "2. **Pre-Commit Inspection**: You can inspect diffs via `git diff --cached`, run your test suite, or modify changes before finalizing.\n"
                "3. **Commit Creation**: When verified, record the commit normally:\n"
                "   ```bash\n"
                "   git commit -m \"Custom commit message for picked changes\"\n"
                "   ```\n"
                "4. **Conflict Resolution / Abort**: If merge conflicts occur: resolve manually or run `git cherry-pick --abort`."
            )

        # Scenario 2: AVL Tree Rotation (algorithms_learning)
        if "avl" in q_lower or "rotation" in q_lower:
            if applied and candidate_preference == "DEPTH:CONCISE":
                return (
                    "**AVL Tree Rotations (Concise Summary)**:\n\n"
                    "- **Single Rotations (LL / RR)**: Re-pivot unbalanced node; 3 pointer updates. Cost: $O(1)$.\n"
                    "- **Double Rotations (LR / RL)**: Rotate child opposite, then rotate parent. Cost: $O(1)$.\n"
                    "- **Balance Factor Invariant**: $\\text{BF}(u) = \\text{height}(L) - \\text{height}(R) \\in \\{-1, 0, +1\\}$. Rebalance triggers if $|\\text{BF}| > 1$."
                )
            if applied and candidate_preference == "PEDAGOGY:INTUITIVE":
                return (
                    "**AVL Tree Rotations (Intuitive Analogy)**:\n\n"
                    "Think of an AVL tree as a balanced playground seesaw. Each node carries weight (subtrees) on its left and right arms.\n\n"
                    "- When an insertion adds weight to an outer arm, the seesaw tips dangerously (Balance Factor = +2).\n"
                    "- A **Right Rotation** is like having the pivot child slide into the center seat while the heavy parent gently swings down to the right. The tree naturally re-levels without dislodging the sorted order.\n"
                    "- If a child is heavy on an inner branch (Left-Right zigzag), it is like carrying a backpack on the wrong shoulder. First you swing the child around (Left rotation) into a straight line, then execute the main re-leveling swing (Right rotation)."
                )
            if applied and candidate_preference == "DEPTH:DETAILED":
                return (
                    "### AVL Tree Rotations: In-Depth Architectural Analysis\n\n"
                    "AVL trees maintain the strict invariant that for every node $u$, the balance factor satisfies:\n"
                    "$$\\text{BF}(u) = \\text{height}(u.\\text{left}) - \\text{height}(u.\\text{right}) \\in \\{-1, 0, +1\\}$$\n\n"
                    "#### 1. The Four Rebalancing Cases:\n"
                    "1. **Left-Left (LL)**: Insertion into left subtree of left child ($\\text{BF} = +2, \\text{child.BF} \\ge 0$). Handled via single Right Rotation on root.\n"
                    "2. **Right-Right (RR)**: Symmetrical to LL. Handled via single Left Rotation.\n"
                    "3. **Left-Right (LR)**: Insertion into right subtree of left child. Handled via double rotation: Left Rotation on left child, followed by Right Rotation on pivot.\n"
                    "4. **Right-Left (RL)**: Symmetrical double rotation: Right Rotation on right child, followed by Left Rotation on pivot.\n\n"
                    "#### 2. Pointer Mutation Invariant:\n"
                    "During a Right Rotation on node $P$ with left child $L$:\n"
                    "- $L.\\text{right}$ becomes $P.\\text{left}$.\n"
                    "- $P$ becomes $L.\\text{right}$.\n"
                    "- Height recalculation is local: $\\text{height}(u) = 1 + \\max(\\text{height}(u.\\text{left}), \\text{height}(u.\\text{right}))$.\n"
                    "- Strict worst-case cost: Exactly $O(1)$ pointer assignments and $O(\\log n)$ upward path traversal."
                )
            return (
                "### AVL Tree Rebalancing Mechanism\n\n"
                "An AVL tree is a self-balancing binary search tree where the height difference between left and right subtrees "
                "(the balance factor) never exceeds 1.\n\n"
                "- **Single Rotations (LL and RR)**: Used when an insertion occurs in the outer subtree. A single pointer reassignment lowers the heavier subtree and restores balance in $O(1)$ operations.\n"
                "- **Double Rotations (LR and RL)**: Used when an insertion occurs in an inner subtree (a zigzag pattern). The inner child is first rotated outward, converting the structure into an outer case, followed by a pivot rotation.\n\n"
                "This guarantees worst-case search, insertion, and deletion times of $O(\\log n)$."
            )

        # Scenario 3: Quicksort (algorithms_learning)
        if "quicksort" in q_lower or "partition" in q_lower:
            if applied and candidate_preference == "DEPTH:CONCISE":
                return (
                    "**Quicksort Partitioning (Concise Summary)**:\n\n"
                    "- **Lomuto**: Single pointer `i`; swaps smaller elements left. Simpler, but makes more swaps ($O(n)$) and degrades on duplicates.\n"
                    "- **Hoare**: Two pointers moving inward from both ends; terminates when pointers cross. Faster with fewer swaps (avg $n/6$).\n"
                    "- **Pitfall**: Already sorted arrays with naive pivot choice yield $O(n^2)$ worst-case; mitigate using Median-of-Three."
                )
            return (
                "### Quicksort Partitioning Principles\n\n"
                "Quicksort relies on selecting a pivot element and rearranging the array such that all elements smaller than the pivot precede it, "
                "and all greater elements follow it.\n\n"
                "- **Lomuto Partition Scheme**: Maintains an index `i` tracking the boundary of smaller elements while scanning with `j`. Easy to implement but performs more swaps.\n"
                "- **Hoare Partition Scheme**: Scans from both ends inward until finding an inverted pair, swapping them. On average, performs three times fewer swaps than Lomuto.\n"
                "- **Common Pitfall**: Naive pivot selection (e.g. first or last element) causes catastrophic $O(n^2)$ degradation on sorted inputs. Standard implementations use randomized pivots or Median-of-Three."
            )

        # Generic technical response
        if applied and candidate_preference == "DEPTH:CONCISE":
            return (
                f"**Concise Summary for**: {query}\n\n"
                "- Core mechanism operates in $O(1)$ or $O(\\log n)$ under standard constraints.\n"
                "- Primary edge cases: boundary invariants, nil pointer checks, and memory fragmentation.\n"
                "- Recommended approach: favor deterministic memory structures with clear ownership semantics."
            )
        return (
            f"### Technical Analysis: {query}\n\n"
            "This problem involves foundational computer science principles:\n\n"
            "1. **Core Mechanics**: The system enforces structural correctness through invariant checks and state validation.\n"
            "2. **Trade-offs**: Asymptotic time complexity balances against memory overhead and cache locality.\n"
            "3. **Implementation Practice**: Handle edge cases such as empty collections, concurrency races, and unexpected input boundaries."
        )


# -----------------------------------------------------------------------------
# 4. Master Engine Orchestrator
# -----------------------------------------------------------------------------

class AttuneRuntimeEngine:
    """Master runtime coordinator for project ATTUNE."""

    def __init__(self, model_path=None, config_path=None):
        logger.info("Initializing ATTUNE Runtime Engine...")

        # Initialize Memory Store
        self.memory = PreferenceMemoryStore()

        # Signal Extractor & Response Generator
        self.signal_extractor = SignalExtractor()
        self.response_generator = ResponseGenerator()

        # Initialize ATTUNE Gating Inference Model
        kwargs = {}
        if model_path:
            kwargs["model_path"] = model_path
        if config_path:
            kwargs["config_path"] = config_path
        self.gate = AttuneGateInference(**kwargs)

        logger.info(
            f"ATTUNE Runtime Engine online. Calibrated tau* = {self.gate.tau_star:.4f}, "
            f"T* = {self.gate.temperature:.4f}"
        )

    def process_turn(
        self,
        session_id: str,
        message: str,
        force_candidate_pref: Optional[str] = None,
        force_scope: Optional[str] = None,
        force_evidence: Optional[List[str]] = None,
        force_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes a user turn:
        1. Analyzes signals (preference tokens, scopes, overrides).
        2. Updates session memory state.
        3. Executes 2-Tier Gating Inference.
        4. Generates appropriate response.
        5. Returns response and diagnostic telemetry.
        """
        t0 = time.perf_counter()
        session = self.memory.get_session(session_id)

        # Extract signals or use forced preset parameters
        override_sig, pref_sig, scope_sig = self.signal_extractor.analyze_message(message)

        # Explicit overrides take strict precedence
        if force_override is not None:
            session.recent_override = force_override
        elif override_sig is not None:
            session.trigger_override(override_sig)

        # Candidate preference updates
        if force_candidate_pref is not None:
            session.candidate_preference = force_candidate_pref
            if force_scope is not None:
                session.preference_scope = force_scope
            if force_evidence:
                session.supporting_evidence = force_evidence
        elif pref_sig is not None and not session.recent_override:
            session.set_preference(
                preference=pref_sig,
                scope=scope_sig or session.preference_scope or "algorithms_learning",
                evidence=f"User turn: '{message.strip()}'",
            )

        # Prepare gating inputs
        active_pref = session.candidate_preference or "DEPTH:CONCISE"
        active_scope = session.preference_scope or "algorithms_learning"
        active_evidence = session.supporting_evidence if session.supporting_evidence else [
            f"User feedback: 'Prefers {active_pref} in {active_scope}'"
        ]
        active_override = session.recent_override

        # Execute ATTUNE Gating Model
        gate_res = self.gate.predict(
            current_query=message,
            candidate_preference=active_pref,
            preference_scope=active_scope,
            supporting_evidence=active_evidence,
            recent_override=active_override,
        )

        # Generate Response
        response_text = self.response_generator.generate(
            query=message,
            applied=gate_res["applied"],
            candidate_preference=active_pref if gate_res["applied"] else None,
            reason=gate_res["reason"],
        )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Record history turn
        session.history.append({
            "query": message,
            "response": response_text,
            "applied": gate_res["applied"],
            "probability": gate_res["probability"],
        })

        # Telemetry matching docs/spec.md
        telemetry = {
            "candidate_preference": active_pref,
            "preference_scope": active_scope,
            "probability": gate_res["probability"],
            "threshold": gate_res["threshold"],
            "applied": gate_res["applied"],
            "tier": gate_res.get("tier", "Tier 2 Gating Model"),
            "reason": gate_res["reason"],
            "latency_ms": latency_ms,
            "recent_override": active_override,
            "supporting_evidence": active_evidence,
        }

        return {
            "response": response_text,
            "telemetry": telemetry,
        }
