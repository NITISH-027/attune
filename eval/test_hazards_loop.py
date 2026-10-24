"""
Verification script for end-to-end validation of the learning -> saving -> retrieval -> application loop.

Tests all 4 feedback points:
1. Memory-writing step: "always include safety hazards" is extracted, saved to persistent memory, and surfaced in /api/memory.
2. In-context gap test: A subsequent query WITHOUT safety words ("explain recrystallization") retrieves and applies the saved preference.
3. Telemetry explanation: Verifies newly_learned_memory, retrieved_memories, and applied_memories in telemetry.
4. Performance claims: Verifies /api/status returns empirical v2 open-memory metrics and frozen v1 baseline.
"""

import sys
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"
SESSION_ID = "test_feedback_loop_session"

def make_request(endpoint: str, method: str = "GET", payload: dict = None):
    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8") if payload else None
    headers = {"Content-Type": "application/json"} if payload else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("=" * 70)
    print("RUNNING ATTUNE E2E FEEDBACK LOOP VERIFICATION")
    print("=" * 70)

    # 0. Check Status & Performance Claims (Feedback 4)
    print("\n--- Test 4: Performance Claims & Architectural Framing (/api/status) ---")
    status = make_request("/api/status")
    v2_metrics = status.get("v2_metrics", {})
    v1_baseline = status.get("v1_baseline", {})
    print(f"Engine Version: {status.get('engine_version')}")
    print(f"Architecture: {status.get('active_architecture')}")
    print(f"v2 CUPID Open-Domain Recall: {v2_metrics.get('cupid_open_domain_recall')}%")
    print(f"v1 CUPID Frozen Recall: {v2_metrics.get('v1_cupid_recall')}% ({v2_metrics.get('recall_gain')} gain)")
    print(f"Lifecycle Fidelity: {v2_metrics.get('lifecycle_correction_fidelity')}%")
    print(f"v1 Frozen Baseline: {v1_baseline.get('name')} (In-Domain Accuracy: {v1_baseline.get('in_domain_accuracy')*100}%, tau*={v1_baseline.get('tau_star')})")
    assert v2_metrics.get("cupid_open_domain_recall") == 83.3, "Expected 83.3% recall"
    assert v1_baseline.get("tau_star") == 0.9000, "Expected v1 tau* 0.9000"
    print("[PASS] Test 4 Passed: Status endpoint accurately separates v2 open-memory metrics from frozen v1 baseline.")

    # 1. Reset Session
    print("\n--- Resetting Test Session ---")
    reset_res = make_request("/api/reset", "POST", {"session_id": SESSION_ID})
    assert len(reset_res.get("preferences", [])) == 0, "Session should start clean with 0 preferences."
    print("[PASS] Session reset clean.")

    # 2. Turn 1: Neutral baseline query
    print("\n--- Turn 1: Neutral Query ('What is fractional distillation?') ---")
    turn1_res = make_request("/api/chat", "POST", {
        "message": "What is fractional distillation?",
        "session_id": SESSION_ID
    })
    t1_telemetry = turn1_res.get("telemetry", {})
    print(f"Gating Status: {t1_telemetry.get('tier')}")
    print(f"Applied: {t1_telemetry.get('applied')} | Reason: {t1_telemetry.get('reason')}")
    assert t1_telemetry.get("applied") is False, "Turn 1 should not apply any preference."
    print("[PASS] Turn 1 Passed: Neutral baseline, gate closed.")

    # 3. Turn 2: Explicit preference instruction (Feedback 1 & 3)
    print("\n--- Turn 2: Preference Instruction ('From now on, always include safety hazards and precautions when explaining chemistry procedures') ---")
    turn2_res = make_request("/api/chat", "POST", {
        "message": "From now on, always include safety hazards and precautions when explaining chemistry procedures",
        "session_id": SESSION_ID
    })
    t2_telemetry = turn2_res.get("telemetry", {})
    learned_mem = t2_telemetry.get("newly_learned_memory")
    print(f"Newly Learned Memory Recorded: {bool(learned_mem)}")
    if learned_mem:
        print(f"  Description: {learned_mem.get('preference_description')}")
        print(f"  Token/Dimension: {learned_mem.get('preference_token')} / {learned_mem.get('dimension')}")
        print(f"  Verbatim Evidence: {learned_mem.get('evidence_quotes')}")
    assert learned_mem is not None, "Turn 2 MUST extract and record a newly_learned_memory!"
    print("[PASS] Test 1 & 3 Passed (Learning & Telemetry): Preference extracted and signaled in telemetry.")

    # 4. Check Persistent Memory Store (/api/memory) (Feedback 1)
    print("\n--- Checking Reusable Personal Memory Store (/api/memory) ---")
    mem_res = make_request(f"/api/memory?session_id={SESSION_ID}")
    prefs = mem_res.get("preferences", [])
    print(f"Found {len(prefs)} persistent preferences in My Voice:")
    for p in prefs:
        print(f"  - [{p.get('dimension')}] {p.get('human_readable')} (Token: {p.get('preference_token')})")
    assert len(prefs) >= 1, "Preference MUST be saved and visible in /api/memory"
    has_hazard = any("hazard" in p.get("human_readable", "").lower() or "safety" in p.get("human_readable", "").lower() or p.get("preference_token") == "CONTENT:SAFETY_HAZARDS" for p in prefs)
    assert has_hazard, "Safety hazards preference MUST be present in memory list!"
    print("[PASS] Test 1 Passed (Memory Saved): Preference successfully persisted in reusable personal memory.")

    # 5. Turn 3: Follow-up query WITHOUT safety keywords (Feedback 2 & 3)
    print("\n--- Turn 3: Follow-up Query WITHOUT Style Keywords ('Explain the procedure for recrystallization of benzoic acid') ---")
    turn3_res = make_request("/api/chat", "POST", {
        "message": "Explain the procedure for recrystallization of benzoic acid",
        "session_id": SESSION_ID
    })
    t3_telemetry = turn3_res.get("telemetry", {})
    t3_response = turn3_res.get("response", "")
    print(f"Applied: {t3_telemetry.get('applied')}")
    print(f"Probability: {t3_telemetry.get('probability')}")
    print(f"Reason: {t3_telemetry.get('reason')}")
    retrieved = t3_telemetry.get("retrieved_memories", [])
    applied_mems = t3_telemetry.get("applied_memories", [])
    print(f"Retrieved Candidates: {len(retrieved)}")
    print(f"Applied Memories: {len(applied_mems)}")
    for a in applied_mems:
        print(f"  - {a.get('preference_description')}")

    print("\nResponse Snippet (checking for safety precautions):")
    snippet = t3_response[:300].replace('\n', ' ')
    print(f"  '{snippet}...'")

    # Assertions for Turn 3
    assert t3_telemetry.get("applied") is True, "Gate MUST OPEN for chemistry procedure topic matching saved scope!"
    assert len(applied_mems) >= 1, "Must have at least 1 applied memory"
    assert any("hazard" in resp_word or "safety" in resp_word or "precaution" in resp_word for resp_word in t3_response.lower().split()), "Model response MUST contain safety hazards/precautions!"
    print("[PASS] Test 2 Passed: Preference retrieved and applied from reusable memory when user DID NOT repeat style keyword.")

    print("\n" + "=" * 70)
    print("ALL 4 FEEDBACK TESTS PASSED ACCURATELY & VERIFIED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
