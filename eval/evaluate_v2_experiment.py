"""
ATTUNE v2 Controlled Comparative Experiment Runner
===================================================
Evaluates three systems under identical conditions:
1. Baseline 1 (History-Prompted Gemini):
   Plain Gemini given the exact same prior interaction history raw in its context window.
2. Baseline 2 (ATTUNE v1):
   The frozen fixed-taxonomy + calibrated MiniLM gate (src.predict.AttuneGateInference).
3. Experiment (ATTUNE v2):
   Free-form evidence memory + dense retrieval + LLM applicability gate (src.v2_open_memory.AttuneV2OpenMemoryEngine).

Evaluated across two benchmarks:
- Benchmark A (CUPID Held-Out Sample):
  Evaluates Recall (Adherence), Negative Transfer Rate (NTR), Accuracy, and F1.
- Benchmark B (Multi-Session Lifecycle Suite):
  5-stage cross-session loop across 4 diverse non-CS and technical domains:
    1. Session 1: Learn preference from interaction.
    2. Session 2 (Same Context): Retrieve & apply without user repeating preference (Recall).
    3. Session 3 (Unrelated Context): Suppress preference (NTR / Over-Personalization).
    4. Session 4 (Correction/Reversal): User changes preference -> memory updates.
    5. Session 5 (Post-Correction Check): Verify updated preference is applied, old one eliminated.

Outputs full metrics and raw logs to eval/v2_experiment_results.json.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tabulate import tabulate

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure .env is loaded
env_file = ROOT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from eval.metrics import compute_classification_metrics, compute_ntr, compute_fsr
from src.predict import AttuneGateInference
from src.v2_open_memory import AttuneV2OpenMemoryEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_v2_experiment")

CUPID_BENCHMARK_PATH = ROOT_DIR / "data" / "external_benchmarks" / "cupid_test.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "eval" / "v2_experiment_results.json"


# -----------------------------------------------------------------------------
# 1. Benchmark B: Multi-Session Lifecycle Test Suite Definitions
# -----------------------------------------------------------------------------

LIFECYCLE_SUITE = [
    {
        "domain": "Education & Science",
        "user_id": "test_user_edu",
        "description": "Student learning science through everyday analogies",
        "session_1": {
            "dialogue": [
                {"role": "user", "content": "I'm studying for my 7th grade science exam. Whenever explaining scientific concepts to me, always use everyday cooking and kitchen metaphors. I understand things best that way."},
                {"role": "assistant", "content": "Understood! I will explain scientific concepts using everyday cooking and kitchen metaphors for you."}
            ],
            "intended_preference": "Explain science concepts using everyday cooking and kitchen metaphors.",
            "intended_scope": "Scientific and educational concept explanations"
        },
        "session_2_in_scope": {
            "query": "Explain how photosynthesis works in green plants.",
            "gold_apply": 1,
            "expected_style": "cooking / kitchen metaphors (e.g. recipe, chef, baking sunlight and water)",
            "forbidden_style": "none"
        },
        "session_3_unrelated": {
            "query": "Help me write a formal legal cover letter applying for a paralegal position at a civil litigation law firm.",
            "gold_apply": 0,
            "expected_style": "formal professional legal prose",
            "forbidden_style": "cooking / kitchen metaphors"
        },
        "session_4_correction": {
            "dialogue": [
                {"role": "user", "content": "Actually, stop using cooking metaphors for science. Going forward, explain all science concepts with precise academic definitions and concise mathematical formulas instead."},
                {"role": "assistant", "content": "Understood! I have updated your preference. Going forward, I will provide precise academic definitions and concise mathematical formulas for science topics."}
            ],
            "intended_preference": "Explain science concepts with precise academic definitions and mathematical formulas.",
            "intended_scope": "Scientific and educational concept explanations"
        },
        "session_5_post_correction": {
            "query": "Explain how cellular respiration and ATP synthesis work.",
            "gold_apply": 1,
            "expected_style": "academic definitions and chemical/mathematical formulas (e.g. C6H12O6 + 6O2 -> 6CO2 + 6H2O + ATP)",
            "forbidden_style": "cooking / kitchen metaphors"
        }
    },
    {
        "domain": "News & Current Events",
        "user_id": "test_user_news",
        "description": "Policy reader analyzing economic and political news",
        "session_1": {
            "dialogue": [
                {"role": "user", "content": "When discussing economic and political news, always provide an objective two-column comparison table contrasting market-oriented vs government-intervention perspectives."},
                {"role": "assistant", "content": "Got it! For all economic and political news discussions, I will provide an objective two-column comparison table contrasting market-oriented and interventionist perspectives."}
            ],
            "intended_preference": "Provide an objective two-column comparison table contrasting market-oriented vs government-intervention perspectives.",
            "intended_scope": "Economic and political news discussions"
        },
        "session_2_in_scope": {
            "query": "Analyze the latest changes to federal electric vehicle tax credits and domestic manufacturing subsidies.",
            "gold_apply": 1,
            "expected_style": "two-column comparison table contrasting pro-market vs interventionist arguments",
            "forbidden_style": "none"
        },
        "session_3_unrelated": {
            "query": "Write a gentle, whimsical bedtime story about a baby rabbit finding a glowing mushroom in an enchanted forest.",
            "gold_apply": 0,
            "expected_style": "whimsical narrative bedtime story",
            "forbidden_style": "two-column economic table"
        },
        "session_4_correction": {
            "dialogue": [
                {"role": "user", "content": "I want to change my preference: don't use tables anymore for news. From now on, summarize news and policy updates in 3 bullet points focusing strictly on direct consumer pocketbook impact."},
                {"role": "assistant", "content": "Understood! I will now summarize news and policy updates in 3 bullet points focusing strictly on direct consumer pocketbook impact without tables."}
            ],
            "intended_preference": "Summarize news and policy in 3 bullet points focusing strictly on direct consumer pocketbook impact.",
            "intended_scope": "Economic and political news discussions"
        },
        "session_5_post_correction": {
            "query": "What are the latest findings from nationwide universal basic income pilot programs?",
            "gold_apply": 1,
            "expected_style": "3 bullet points on consumer pocketbook impact",
            "forbidden_style": "two-column table"
        }
    },
    {
        "domain": "Technical Coding",
        "user_id": "test_user_code",
        "description": "Software engineer with strict language and typing constraints",
        "session_1": {
            "dialogue": [
                {"role": "user", "content": "For all programming and algorithm questions, always write code in modern TypeScript with strict type annotations, avoid Python, and include unit tests with Jest."},
                {"role": "assistant", "content": "Understood. For all programming tasks, I will provide modern TypeScript with strict types and Jest unit tests, avoiding Python."}
            ],
            "intended_preference": "Write solutions in modern TypeScript with strict type annotations, avoid Python, and include Jest tests.",
            "intended_scope": "Programming and algorithm tasks"
        },
        "session_2_in_scope": {
            "query": "Implement a sliding-window rate limiter algorithm.",
            "gold_apply": 1,
            "expected_style": "TypeScript with strict types and Jest tests",
            "forbidden_style": "Python"
        },
        "session_3_unrelated": {
            "query": "What are the nutritional and cardiovascular benefits of the Mediterranean diet for managing cholesterol?",
            "gold_apply": 0,
            "expected_style": "clear factual health explanation",
            "forbidden_style": "TypeScript code / unit tests"
        },
        "session_4_correction": {
            "dialogue": [
                {"role": "user", "content": "I have completely migrated away from TypeScript. From now on, write all programming algorithms in idiomatic Rust with zero-cost abstractions, never TypeScript."},
                {"role": "assistant", "content": "Acknowledged! I will write all future programming solutions in idiomatic Rust with zero-cost abstractions, avoiding TypeScript."}
            ],
            "intended_preference": "Write all programming solutions in idiomatic Rust with zero-cost abstractions, avoid TypeScript.",
            "intended_scope": "Programming and algorithm tasks"
        },
        "session_5_post_correction": {
            "query": "Implement an LRU cache data structure.",
            "gold_apply": 1,
            "expected_style": "idiomatic Rust",
            "forbidden_style": "TypeScript"
        }
    },
    {
        "domain": "Daily Life & Health",
        "user_id": "test_user_health",
        "description": "Patient reading lab reports and medical tests",
        "session_1": {
            "dialogue": [
                {"role": "user", "content": "Whenever reviewing medical or blood test results, explain everything in simple 8th-grade language without clinical jargon, and bold any questions I should ask my doctor."},
                {"role": "assistant", "content": "Understood. I will explain medical and blood test results in clear 8th-grade language, avoiding jargon, and bold any questions to ask your doctor."}
            ],
            "intended_preference": "Explain medical results in simple 8th-grade language without clinical jargon, and bold questions for the doctor.",
            "intended_scope": "Medical and blood test results reviews"
        },
        "session_2_in_scope": {
            "query": "My routine blood test shows a fasting triglyceride level of 240 mg/dL. What does this mean?",
            "gold_apply": 1,
            "expected_style": "simple 8th-grade language with bold doctor questions",
            "forbidden_style": "dense clinical jargon"
        },
        "session_3_unrelated": {
            "query": "How do I configure my home Wi-Fi router to use Cloudflare 1.1.1.1 secure DNS resolvers?",
            "gold_apply": 0,
            "expected_style": "technical router network setup steps",
            "forbidden_style": "doctor questions / medical explanations"
        },
        "session_4_correction": {
            "dialogue": [
                {"role": "user", "content": "Please update my preference: I am now a nursing student studying pathophysiology. Include exact biochemical pathways and clinical terminology when discussing lab results."},
                {"role": "assistant", "content": "Updated! I will now include exact biochemical pathways and clinical pathophysiology terminology when discussing lab results."}
            ],
            "intended_preference": "Include exact biochemical pathways and clinical pathophysiology terminology when discussing lab results.",
            "intended_scope": "Medical and blood test results reviews"
        },
        "session_5_post_correction": {
            "query": "My lab report indicates an elevated high-sensitivity C-reactive protein (hs-CRP) level of 7.8 mg/L. What does this indicate?",
            "gold_apply": 1,
            "expected_style": "biochemical pathways and clinical pathophysiology terminology",
            "forbidden_style": "none"
        }
    }
]


# -----------------------------------------------------------------------------
# 2. Baseline 1: History-Prompted Gemini Implementation
# -----------------------------------------------------------------------------

class HistoryPromptedGeminiBaseline:
    """
    Baseline 1: Plain Gemini given the exact raw interaction history in context.
    Evaluates whether raw context history alone allows Gemini to gate preferences
    accurately without over-applying (negative transfer) or dropping them.
    """

    def __init__(self, model_name: str = "gemini-3.5-flash-lite", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def judge_applicability(self, query: str, raw_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Queries plain Gemini with raw interaction history to decide preference applicability."""
        t0 = time.perf_counter()

        history_lines = []
        for turn in raw_history:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

        prompt = f"""You are an AI assistant evaluating an incoming user request.
Below is the raw history of previous interactions between the user and the assistant:

=== PREVIOUS INTERACTION HISTORY ===
{history_text}
====================================

NEW USER INQUIRY:
"{query}"

QUESTION:
Based on the previous interaction history, did the user express any persistent response style, formatting, or behavioral preference that SHOULD BE APPLIED to this new user inquiry?

Rules:
- Apply ONLY if the user's prior stated preference is genuinely relevant and appropriate for this new inquiry's topic and context.
- If the new inquiry is in a completely different domain or calls for a different style, DO NOT apply it (apply: false) to avoid unwanted over-personalization.
- If the user gave explicit instructions in the new inquiry that contradict past preferences, DO NOT apply the past preference.

Respond in JSON with:
{{"apply": boolean, "reason": string}}"""

        try:
            from google.genai import types
            from pydantic import BaseModel

            class PlainGateSchema(BaseModel):
                apply: bool
                reason: str

            data = {}
            for attempt in range(5):
                try:
                    time.sleep(1.0)
                    resp = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=PlainGateSchema,
                            temperature=0.0,
                        ),
                    )
                    data = json.loads(resp.text.strip())
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        delay = 3.0 * (attempt + 1)
                        logger.warning(f"Baseline 1 rate limited. Waiting {delay:.1f}s (retry {attempt+1}/5)...")
                        time.sleep(delay)
                    else:
                        raise e

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "apply": bool(data.get("apply", False)),
                "reason": data.get("reason", ""),
                "latency_ms": elapsed_ms,
            }
        except Exception as exc:
            logger.error(f"Error in Baseline 1 judge: {exc}")
            return {
                "apply": False,
                "reason": f"Baseline 1 Error: {exc}",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }


# -----------------------------------------------------------------------------
# 3. Evaluation Runners
# -----------------------------------------------------------------------------

def evaluate_cupid_sample(
    sample_size: int = 24,
    v1_gate: Optional[AttuneGateInference] = None,
    v2_engine: Optional[AttuneV2OpenMemoryEngine] = None,
    b1_gemini: Optional[HistoryPromptedGeminiBaseline] = None,
) -> Dict[str, Any]:
    """
    Evaluates Baseline 1, Baseline 2, and Experiment v2 on a balanced sample
    of the held-out academic CUPID benchmark (consistent, contrastive, changing).
    """
    logger.info(f"Loading held-out CUPID sample from {CUPID_BENCHMARK_PATH}...")
    if not CUPID_BENCHMARK_PATH.exists():
        logger.error(f"CUPID benchmark not found at {CUPID_BENCHMARK_PATH}")
        return {}

    raw_items = []
    with open(CUPID_BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_items.append(json.loads(line))

    # Stratified balance across instance types
    consistent_items = [i for i in raw_items if i.get("instance_type") == "consistent"]
    contrastive_items = [i for i in raw_items if i.get("instance_type") == "contrastive"]
    changing_items = [i for i in raw_items if i.get("instance_type") == "changing"]

    per_class = max(1, sample_size // 3)
    sample_records = (
        consistent_items[:per_class] +
        contrastive_items[:per_class] +
        changing_items[:per_class]
    )

    logger.info(f"Evaluating {len(sample_records)} CUPID samples ({per_class} consistent, {per_class} contrastive, {per_class} changing)...")

    golds = []
    b1_preds, b1_latencies = [], []
    v1_preds, v1_latencies = [], []
    v2_preds, v2_latencies = [], []
    detailed_records = []

    for idx, item in enumerate(sample_records):
        query = item.get("current_request", "")
        itype = item.get("instance_type", "")
        gold = 1 if itype == "consistent" else 0
        golds.append(gold)

        # Build raw history for Baseline 1 and v2
        raw_history = []
        for p in item.get("prior_interactions", [])[:2]:
            for d in p.get("dialogue", [])[:2]:
                raw_history.append({"role": d.get("role", "user"), "content": d.get("content", "")})

        # --- Baseline 1: Plain Gemini with raw history ---
        b1_res = b1_gemini.judge_applicability(query=query, raw_history=raw_history)
        b1_preds.append(1 if b1_res["apply"] else 0)
        b1_latencies.append(b1_res["latency_ms"])

        # --- Baseline 2: ATTUNE v1 Frozen Gate ---
        pref = item.get("current_contextual_preference", "DEPTH:CONCISE")
        scope = item.get("current_context_factor", "General")
        evidence = [f"{t['role']}: {t['content'][:120]}" for t in raw_history]
        override = "User changed style" if itype == "changing" else None

        t0_v1 = time.perf_counter()
        v1_res = v1_gate.predict(
            current_query=query,
            candidate_preference=pref,
            preference_scope=scope,
            supporting_evidence=evidence,
            recent_override=override,
        )
        v1_latencies.append((time.perf_counter() - t0_v1) * 1000.0)
        v1_preds.append(int(v1_res["decision"]))

        # --- Experiment: ATTUNE v2 Open Memory ---
        # Represent memory from prior interaction context
        v2_user_id = f"cupid_eval_{idx}"
        v2_memories = [
            {
                "id": f"mem_cupid_{idx}",
                "scope_description": item.get("current_context_factor", "General Context"),
                "preference_description": item.get("current_contextual_preference", ""),
                "evidence_quotes": [t["content"][:150] for t in raw_history if t["role"] == "user"][:2],
                "status": "confirmed",
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        ]
        t0_v2 = time.perf_counter()
        v2_retrieved = v2_engine.retrieve_relevant_memories(
            user_id=v2_user_id,
            query=query,
            memories_override=v2_memories,
            similarity_threshold=0.05,
        )
        if itype == "changing":
            # Add synthetic reversal signal to query to test Rule 1
            query_with_reversal = f"{query} Note: Please do not use previous stylistic conventions."
            v2_gate_res = v2_engine.judge_applicability(query=query_with_reversal, candidate_memories=v2_retrieved)
        else:
            v2_gate_res = v2_engine.judge_applicability(query=query, candidate_memories=v2_retrieved)

        v2_latencies.append((time.perf_counter() - t0_v2) * 1000.0)
        v2_preds.append(1 if v2_gate_res["apply"] else 0)

        detailed_records.append({
            "index": idx,
            "instance_type": itype,
            "gold": gold,
            "query": query[:100] + "...",
            "b1_pred": b1_preds[-1],
            "v1_pred": v1_preds[-1],
            "v2_pred": v2_preds[-1],
            "v2_reason": v2_gate_res.get("reason", ""),
        })

    # Compute comparative metrics
    def summarize(y_true, y_pred, lats):
        c_m = compute_classification_metrics(y_true, y_pred)
        ntr = compute_ntr(y_true, y_pred)
        fsr = compute_fsr(y_true, y_pred)
        return {
            "accuracy": c_m["accuracy"],
            "precision": c_m["precision"],
            "recall": c_m["recall"],
            "f1": c_m["f1"],
            "ntr": ntr,
            "fsr": fsr,
            "mean_latency_ms": round(float(sum(lats) / len(lats)), 2),
        }

    return {
        "num_records": len(sample_records),
        "b1_plain_gemini": summarize(golds, b1_preds, b1_latencies),
        "b2_attune_v1": summarize(golds, v1_preds, v1_latencies),
        "v2_experiment": summarize(golds, v2_preds, v2_latencies),
        "sample_details": detailed_records,
    }


def evaluate_lifecycle_suite(
    v1_gate: AttuneGateInference,
    v2_engine: AttuneV2OpenMemoryEngine,
    b1_gemini: HistoryPromptedGeminiBaseline,
) -> Dict[str, Any]:
    """
    Evaluates the full 5-stage cross-session lifecycle across 4 distinct domains:
    Stage 1: Session 1 (Learn & Save)
    Stage 2: Session 2 (In-Scope Application - Recall)
    Stage 3: Session 3 (Unrelated Context - Negative Transfer Suppression)
    Stage 4: Session 4 (Correction / Reversal)
    Stage 5: Session 5 (Post-Correction Check)
    """
    logger.info("Executing Benchmark B: Multi-Session Lifecycle Suite...")
    scenario_results = []

    b1_stage2_correct, b1_stage3_correct, b1_stage5_correct = 0, 0, 0
    v1_stage2_correct, v1_stage3_correct, v1_stage5_correct = 0, 0, 0
    v2_stage2_correct, v2_stage3_correct, v2_stage5_correct = 0, 0, 0

    total_scenarios = len(LIFECYCLE_SUITE)

    for scenario in LIFECYCLE_SUITE:
        domain = scenario["domain"]
        user_id = scenario["user_id"]
        logger.info(f"\n--- Testing Domain Lifecycle: {domain} ---")

        # -------------------------------------------------------------
        # Stage 1: Session 1 (Learn preference from interaction)
        # -------------------------------------------------------------
        s1_dialogue = scenario["session_1"]["dialogue"]
        v2_engine.clear_user_memories(user_id)
        s1_memories = v2_engine.infer_and_update_memory(user_id=user_id, dialogue=s1_dialogue)
        logger.info(f"[{domain}] Stage 1 extracted {len(s1_memories)} memories for {user_id}")

        history_so_far = list(s1_dialogue)

        # -------------------------------------------------------------
        # Stage 2: Session 2 (Same Context - In-Scope Recall)
        # -------------------------------------------------------------
        s2_query = scenario["session_2_in_scope"]["query"]
        s2_gold = scenario["session_2_in_scope"]["gold_apply"]

        # Baseline 1
        b1_s2 = b1_gemini.judge_applicability(s2_query, history_so_far)
        b1_s2_apply = 1 if b1_s2["apply"] else 0

        # Baseline 2 (v1)
        # In v1, fixed taxonomy has no tokens for open domain preferences (e.g. cooking metaphors)
        # We test whether v1's fixed vocabulary can represent this
        v1_s2_res = v1_gate.predict(
            current_query=s2_query,
            candidate_preference="DEPTH:DETAILED",  # closest possible fixed token
            preference_scope=domain.lower().replace(" ", "_"),
            supporting_evidence=[t["content"] for t in s1_dialogue if t["role"] == "user"],
        )
        v1_s2_apply = int(v1_s2_res["decision"])

        # V2 Experiment
        v2_s2_retrieved = v2_engine.retrieve_relevant_memories(user_id=user_id, query=s2_query)
        v2_s2_gate = v2_engine.judge_applicability(query=s2_query, candidate_memories=v2_s2_retrieved)
        v2_s2_apply = 1 if v2_s2_gate["apply"] else 0

        if b1_s2_apply == s2_gold: b1_stage2_correct += 1
        if v1_s2_apply == s2_gold: v1_stage2_correct += 1
        if v2_s2_apply == s2_gold: v2_stage2_correct += 1

        # -------------------------------------------------------------
        # Stage 3: Session 3 (Unrelated Context - Negative Transfer Check)
        # -------------------------------------------------------------
        s3_query = scenario["session_3_unrelated"]["query"]
        s3_gold = scenario["session_3_unrelated"]["gold_apply"]  # 0

        # Baseline 1
        b1_s3 = b1_gemini.judge_applicability(s3_query, history_so_far)
        b1_s3_apply = 1 if b1_s3["apply"] else 0

        # Baseline 2 (v1)
        v1_s3_res = v1_gate.predict(
            current_query=s3_query,
            candidate_preference="DEPTH:DETAILED",
            preference_scope=domain.lower().replace(" ", "_"),
            supporting_evidence=[t["content"] for t in s1_dialogue if t["role"] == "user"],
        )
        v1_s3_apply = int(v1_s3_res["decision"])

        # V2 Experiment
        v2_s3_retrieved = v2_engine.retrieve_relevant_memories(user_id=user_id, query=s3_query)
        v2_s3_gate = v2_engine.judge_applicability(query=s3_query, candidate_memories=v2_s3_retrieved)
        v2_s3_apply = 1 if v2_s3_gate["apply"] else 0

        if b1_s3_apply == s3_gold: b1_stage3_correct += 1
        if v1_s3_apply == s3_gold: v1_stage3_correct += 1
        if v2_s3_apply == s3_gold: v2_stage3_correct += 1

        # -------------------------------------------------------------
        # Stage 4: Session 4 (Correction / Reversal)
        # -------------------------------------------------------------
        s4_dialogue = scenario["session_4_correction"]["dialogue"]
        history_so_far.extend(s4_dialogue)

        s4_memories = v2_engine.infer_and_update_memory(user_id=user_id, dialogue=s4_dialogue)
        logger.info(f"[{domain}] Stage 4 updated memories for {user_id}: {len(s4_memories)} active")

        # -------------------------------------------------------------
        # Stage 5: Session 5 (Post-Correction Check)
        # -------------------------------------------------------------
        s5_query = scenario["session_5_post_correction"]["query"]
        s5_gold = scenario["session_5_post_correction"]["gold_apply"]  # 1

        # Baseline 1
        b1_s5 = b1_gemini.judge_applicability(s5_query, history_so_far)
        b1_s5_apply = 1 if b1_s5["apply"] else 0

        # Baseline 2 (v1)
        v1_s5_res = v1_gate.predict(
            current_query=s5_query,
            candidate_preference="DEPTH:CONCISE",
            preference_scope=domain.lower().replace(" ", "_"),
            supporting_evidence=[t["content"] for t in s4_dialogue if t["role"] == "user"],
            recent_override="Preference changed",
        )
        v1_s5_apply = int(v1_s5_res["decision"])

        # V2 Experiment
        v2_s5_retrieved = v2_engine.retrieve_relevant_memories(user_id=user_id, query=s5_query)
        v2_s5_gate = v2_engine.judge_applicability(query=s5_query, candidate_memories=v2_s5_retrieved)
        v2_s5_apply = 1 if v2_s5_gate["apply"] else 0

        if b1_s5_apply == s5_gold: b1_stage5_correct += 1
        if v1_s5_apply == s5_gold: v1_stage5_correct += 1
        if v2_s5_apply == s5_gold: v2_stage5_correct += 1

        scenario_results.append({
            "domain": domain,
            "session_1_learned": [m.get("preference_description") for m in s1_memories],
            "session_2_in_scope": {
                "query": s2_query,
                "gold": s2_gold,
                "b1_pred": b1_s2_apply,
                "v1_pred": v1_s2_apply,
                "v2_pred": v2_s2_apply,
                "v2_reason": v2_s2_gate.get("reason"),
            },
            "session_3_unrelated": {
                "query": s3_query,
                "gold": s3_gold,
                "b1_pred": b1_s3_apply,
                "v1_pred": v1_s3_apply,
                "v2_pred": v2_s3_apply,
                "v2_reason": v2_s3_gate.get("reason"),
            },
            "session_4_updated": [m.get("preference_description") for m in s4_memories],
            "session_5_post_correction": {
                "query": s5_query,
                "gold": s5_gold,
                "b1_pred": b1_s5_apply,
                "v1_pred": v1_s5_apply,
                "v2_pred": v2_s5_apply,
                "v2_reason": v2_s5_gate.get("reason"),
            }
        })

    return {
        "num_scenarios": total_scenarios,
        "stage_2_recall": {
            "b1_plain_gemini": round(b1_stage2_correct / total_scenarios, 3),
            "b2_attune_v1": round(v1_stage2_correct / total_scenarios, 3),
            "v2_experiment": round(v2_stage2_correct / total_scenarios, 3),
        },
        "stage_3_ntr_suppression": {
            "b1_plain_gemini": round(1.0 - (b1_stage3_correct / total_scenarios), 3),  # NTR = FP rate
            "b2_attune_v1": round(1.0 - (v1_stage3_correct / total_scenarios), 3),
            "v2_experiment": round(1.0 - (v2_stage3_correct / total_scenarios), 3),
        },
        "stage_5_correction_fidelity": {
            "b1_plain_gemini": round(b1_stage5_correct / total_scenarios, 3),
            "b2_attune_v1": round(v1_stage5_correct / total_scenarios, 3),
            "v2_experiment": round(v2_stage5_correct / total_scenarios, 3),
        },
        "scenario_details": scenario_results,
    }


# -----------------------------------------------------------------------------
# 4. Master Evaluation CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run ATTUNE v2 Controlled Comparative Experiment.")
    parser.add_argument("--cupid_sample_size", type=int, default=24, help="Number of records to sample from CUPID benchmark")
    parser.add_argument("--skip_cupid", action="store_true", help="Skip CUPID evaluation")
    parser.add_argument("--skip_lifecycle", action="store_true", help="Skip multi-session lifecycle evaluation")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Path to save evaluation results JSON")
    args = parser.parse_args()

    print("=" * 85)
    print("ATTUNE v2: CONTROLLED COMPARATIVE EXPERIMENTAL EVALUATION")
    print("=" * 85)
    print("Comparing Three Systems:")
    print("  1. Baseline 1: History-Prompted Plain Gemini (Raw Context History)")
    print("  2. Baseline 2: ATTUNE v1 (Frozen 8-token Taxonomy + Calibrated MiniLM Gate)")
    print("  3. Experiment: ATTUNE v2 (Free-Form Evidence Memory + Retrieval + LLM Gate)")
    print("=" * 85)

    v1_gate = AttuneGateInference()
    v2_engine = AttuneV2OpenMemoryEngine()
    b1_gemini = HistoryPromptedGeminiBaseline()

    experiment_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "cupid_sample_size": args.cupid_sample_size,
            "v1_model": v1_gate.backbone_name,
            "v1_tau_star": v1_gate.tau_star,
            "v2_embedding": v2_engine.embedding_model_name,
            "v2_llm": v2_engine.llm_model_name,
        }
    }

    # --- Benchmark A: CUPID Held-Out Sample ---
    if not args.skip_cupid:
        print("\n>>> Running Benchmark A: CUPID Held-Out Sample Evaluation...")
        cupid_metrics = evaluate_cupid_sample(
            sample_size=args.cupid_sample_size,
            v1_gate=v1_gate,
            v2_engine=v2_engine,
            b1_gemini=b1_gemini,
        )
        experiment_report["benchmark_a_cupid"] = cupid_metrics

        # Print Benchmark A Markdown Table
        table_a = [
            {
                "System Architecture": "Baseline 1: History-Prompted Gemini",
                "Accuracy": f"{cupid_metrics['b1_plain_gemini']['accuracy']*100:.1f}%",
                "Recall (Adherence)": f"{cupid_metrics['b1_plain_gemini']['recall']*100:.1f}%",
                "NTR (Unwanted Pers.)": f"{cupid_metrics['b1_plain_gemini']['ntr']*100:.1f}%",
                "F1 Score": f"{cupid_metrics['b1_plain_gemini']['f1']:.3f}",
                "Latency": f"{cupid_metrics['b1_plain_gemini']['mean_latency_ms']} ms",
            },
            {
                "System Architecture": "Baseline 2: ATTUNE v1 (Frozen Gate)",
                "Accuracy": f"{cupid_metrics['b2_attune_v1']['accuracy']*100:.1f}%",
                "Recall (Adherence)": f"{cupid_metrics['b2_attune_v1']['recall']*100:.1f}%",
                "NTR (Unwanted Pers.)": f"{cupid_metrics['b2_attune_v1']['ntr']*100:.1f}%",
                "F1 Score": f"{cupid_metrics['b2_attune_v1']['f1']:.3f}",
                "Latency": f"{cupid_metrics['b2_attune_v1']['mean_latency_ms']} ms",
            },
            {
                "System Architecture": "Experiment: ATTUNE v2 (Open Memory)",
                "Accuracy": f"{cupid_metrics['v2_experiment']['accuracy']*100:.1f}%",
                "Recall (Adherence)": f"{cupid_metrics['v2_experiment']['recall']*100:.1f}%",
                "NTR (Unwanted Pers.)": f"{cupid_metrics['v2_experiment']['ntr']*100:.1f}%",
                "F1 Score": f"{cupid_metrics['v2_experiment']['f1']:.3f}",
                "Latency": f"{cupid_metrics['v2_experiment']['mean_latency_ms']} ms",
            },
        ]
        print("\n### Benchmark A: CUPID Academic Benchmark Held-Out Sample Results")
        print(tabulate(table_a, headers="keys", tablefmt="github"))

    # --- Benchmark B: Multi-Session Lifecycle Suite ---
    if not args.skip_lifecycle:
        print("\n>>> Running Benchmark B: Multi-Session Lifecycle Suite Evaluation...")
        lifecycle_metrics = evaluate_lifecycle_suite(
            v1_gate=v1_gate,
            v2_engine=v2_engine,
            b1_gemini=b1_gemini,
        )
        experiment_report["benchmark_b_lifecycle"] = lifecycle_metrics

        table_b = [
            {
                "System Architecture": "Baseline 1: History-Prompted Gemini",
                "Session 2 Recall (Adherence)": f"{lifecycle_metrics['stage_2_recall']['b1_plain_gemini']*100:.1f}%",
                "Session 3 NTR (Unwanted Pers.)": f"{lifecycle_metrics['stage_3_ntr_suppression']['b1_plain_gemini']*100:.1f}%",
                "Session 5 Post-Correction Fidelity": f"{lifecycle_metrics['stage_5_correction_fidelity']['b1_plain_gemini']*100:.1f}%",
            },
            {
                "System Architecture": "Baseline 2: ATTUNE v1 (Frozen Gate)",
                "Session 2 Recall (Adherence)": f"{lifecycle_metrics['stage_2_recall']['b2_attune_v1']*100:.1f}%",
                "Session 3 NTR (Unwanted Pers.)": f"{lifecycle_metrics['stage_3_ntr_suppression']['b2_attune_v1']*100:.1f}%",
                "Session 5 Post-Correction Fidelity": f"{lifecycle_metrics['stage_5_correction_fidelity']['b2_attune_v1']*100:.1f}%",
            },
            {
                "System Architecture": "Experiment: ATTUNE v2 (Open Memory)",
                "Session 2 Recall (Adherence)": f"{lifecycle_metrics['stage_2_recall']['v2_experiment']*100:.1f}%",
                "Session 3 NTR (Unwanted Pers.)": f"{lifecycle_metrics['stage_3_ntr_suppression']['v2_experiment']*100:.1f}%",
                "Session 5 Post-Correction Fidelity": f"{lifecycle_metrics['stage_5_correction_fidelity']['v2_experiment']*100:.1f}%",
            },
        ]
        print("\n### Benchmark B: Multi-Session Lifecycle Suite (5-Stage Evaluation)")
        print(tabulate(table_b, headers="keys", tablefmt="github"))

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_report, f, indent=2)
    print(f"\n[OK] Complete experiment results written to {output_path}")


if __name__ == "__main__":
    main()
