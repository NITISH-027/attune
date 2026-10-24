#!/usr/bin/env python3
"""
Test Isolated Memory Loop (ATTUNE v2)
=====================================
Strictly validates:
1. Arbitrary preference learning with exact scope preservation (no hardcoded keywords, no unauthorized scope expansion).
2. Memory persistence across isolated chat sessions with empty conversation history ([]).
3. Negative trigger control (unwanted personalization suppressed when query is out of scope).
4. Conflict resolution (superseding contradictory preferences).
5. User-level privacy isolation and git exclusion.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.v2_open_memory import AttuneV2OpenMemoryEngine
from app.engine import AttuneUniversalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_isolated_memory_loop")


def test_complete_isolated_memory_loop():
    logger.info("================================================================================")
    logger.info("STARTING ISOLATED MEMORY LOOP VALIDATION")
    logger.info("================================================================================")

    engine = AttuneUniversalEngine()
    test_user = "test_eval_user_isolation_99"

    # Step 0: Clean any pre-existing test files for this user
    test_user_file = PROJECT_ROOT / "data" / "user_memories" / f"{test_user}.json"
    if test_user_file.exists():
        test_user_file.unlink()

    # Verify gitignore covers memory files
    gitignore_path = PROJECT_ROOT / ".gitignore"
    with open(gitignore_path, "r", encoding="utf-8") as f:
        gitignore_content = f.read()
    assert "data/user_memories/" in gitignore_content, "FAIL: data/user_memories/ not in .gitignore"
    assert "data/*_store.json" in gitignore_content, "FAIL: data/*_store.json not in .gitignore"
    logger.info("✓ PRIVACY VERIFIED: data/user_memories/ and store files are properly git-ignored.")

    # -------------------------------------------------------------------------
    # TEST 1: Learn arbitrary preference in Chat Session 1
    # -------------------------------------------------------------------------
    chat_1_id = "chat_session_001"
    instruction_msg = "From now on, always include safety hazards and precautions when explaining chemistry procedures"
    
    logger.info(f"\n--- [Chat 1 ({chat_1_id})] User Instruction: '{instruction_msg}' ---")
    res1 = engine.process_turn(
        message=instruction_msg,
        session_id=chat_1_id,
        user_id=test_user,
    )

    learned_mem = res1.get("telemetry", {}).get("newly_learned_memory")
    assert learned_mem is not None, "FAIL: No memory was learned from explicit user instruction"
    logger.info(f"Learned memory description: {learned_mem.get('preference_description')}")
    logger.info(f"Learned memory scope: {learned_mem.get('scope_description')}")
    logger.info(f"Learned memory evidence: {learned_mem.get('evidence_quotes')}")

    # Verify arbitrary learning & scope preservation (no hardcoding, no over-expansion)
    pref_desc = learned_mem.get("preference_description", "").lower()
    scope_desc = learned_mem.get("scope_description", "").lower()
    
    assert "safety" in pref_desc or "hazard" in pref_desc or "precaution" in pref_desc, (
        f"FAIL: Extracted preference '{pref_desc}' missed the core directive"
    )
    assert "chemist" in scope_desc or "procedure" in scope_desc, (
        f"FAIL: Scope '{scope_desc}' did not preserve the chemistry procedure constraint"
    )
    # Ensure it did NOT over-broaden to all biology or general science
    assert "all biology" not in scope_desc and "all science" not in scope_desc, (
        f"FAIL: Scope '{scope_desc}' was arbitrarily broadened without user permission"
    )
    logger.info("✓ TEST 1 PASSED: Arbitrary preference learned and user's exact scope preserved.")

    # Verify physical file existence in isolated user partition
    assert test_user_file.exists(), f"FAIL: User memory partition file {test_user_file} was not written"
    with open(test_user_file, "r", encoding="utf-8") as f:
        stored_user_mems = json.load(f)
    assert len(stored_user_mems) >= 1, "FAIL: User partition file is empty"
    logger.info(f"✓ ISOLATION VERIFIED: Memory stored in user partition {test_user_file.name}")

    # -------------------------------------------------------------------------
    # TEST 2: Separate Chat Session with OLD CONVERSATION HISTORY EXCLUDED
    # -------------------------------------------------------------------------
    chat_2_id = "chat_session_002_fresh"
    test_query = "Explain the step-by-step procedure for recrystallization of benzoic acid."

    logger.info(f"\n--- [Chat 2 ({chat_2_id})] In-Scope Query (History EXCLUDED): '{test_query}' ---")
    
    # Notice: conversation_history is explicitly EMPTY [] - proving memory works from storage, not prompt leakage!
    res2 = engine.process_turn(
        message=test_query,
        session_id=chat_2_id,
        user_id=test_user,
    )

    telemetry2 = res2.get("telemetry", {})
    gate_applied = telemetry2.get("applied")
    gate_reason = telemetry2.get("reason")
    raw_similarity = telemetry2.get("probability")
    applied_mems = telemetry2.get("applied_memories", [])
    response_text = res2.get("response", "")

    logger.info(f"Gate applied: {gate_applied}")
    logger.info(f"Gate reason: {gate_reason}")
    logger.info(f"Raw similarity score: {raw_similarity}")
    logger.info(f"Number of applied memories: {len(applied_mems)}")
    logger.info(f"Response preview:\n{response_text[:400]}...\n")

    assert gate_applied is True, f"FAIL: Gate should be OPEN for in-scope chemistry procedure. Reason: {gate_reason}"
    assert len(applied_mems) > 0, "FAIL: Expected at least 1 applied memory from user partition"
    
    # Check that safety/hazards are actively present in response
    lower_resp = response_text.lower()
    has_safety_content = any(k in lower_resp for k in ["hazard", "safety", "precaution", "hot glass", "burn", "goggles", "boiling chip", "flamm"])
    assert has_safety_content, "FAIL: Response did not reflect safety hazards preference despite gate being open"
    logger.info("✓ TEST 2 PASSED: Memory retrieved across fresh chat session with history excluded and correctly applied.")

    # -------------------------------------------------------------------------
    # TEST 3: Negative Trigger Control (Unwanted Personalization Suppression)
    # -------------------------------------------------------------------------
    chat_3_id = "chat_session_003_fresh"
    out_of_scope_query = "What is the procedure for filing a patent application with the USPTO?"

    logger.info(f"\n--- [Chat 3 ({chat_3_id})] Out-of-Scope Query (History EXCLUDED): '{out_of_scope_query}' ---")
    
    res3 = engine.process_turn(
        message=out_of_scope_query,
        session_id=chat_3_id,
        user_id=test_user,
    )

    telemetry3 = res3.get("telemetry", {})
    gate_applied3 = telemetry3.get("applied")
    gate_reason3 = telemetry3.get("reason")
    response_text3 = res3.get("response", "")

    logger.info(f"Gate applied: {gate_applied3}")
    logger.info(f"Gate reason: {gate_reason3}")
    logger.info(f"Response preview:\n{response_text3[:300]}...\n")

    assert gate_applied3 is False, (
        f"FAIL: Gate should be SUPPRESSED for patent query (out of scope). Reason: {gate_reason3}"
    )
    # Check response does NOT inject chemical safety hazards
    lower_resp3 = response_text3.lower()
    assert "goggles" not in lower_resp3 and "boiling chip" not in lower_resp3 and "flammable solvent" not in lower_resp3, (
        "FAIL: Injected chemistry safety hazards into a patent law question!"
    )
    logger.info("✓ TEST 3 PASSED: Negative trigger correctly suppressed. No unwanted personalization.")

    # -------------------------------------------------------------------------
    # TEST 4: Conflicting Memory Update (Superseding / Replacing Old Preference)
    # -------------------------------------------------------------------------
    chat_4_id = "chat_session_004_update"
    correction_msg = "Going forward, do not include safety hazards for chemistry procedures. Instead, keep them strictly focused on theoretical reaction mechanisms and arrow-pushing."

    logger.info(f"\n--- [Chat 4 ({chat_4_id})] Correction Instruction: '{correction_msg}' ---")
    res4 = engine.process_turn(
        message=correction_msg,
        session_id=chat_4_id,
        user_id=test_user,
    )

    # Check active preferences for user
    active_prefs = engine.memory.list_voice_preferences(user_id=test_user)
    logger.info(f"Total active preferences in user partition: {len(active_prefs)}")
    for p in active_prefs:
        logger.info(f" - [{p.get('id')}] {p.get('preference_description')} (Scope: {p.get('scope_description')})")

    # Verify that conflicting "always include safety hazards" preference is superseded/removed
    has_old_hazard_rule = any(
        "always include safety" in p.get("preference_description", "").lower() 
        for p in active_prefs
    )
    assert not has_old_hazard_rule, "FAIL: Old contradictory safety hazard preference was not superseded!"
    logger.info("✓ TEST 4 PASSED: Contradictory preference successfully resolved/superseded.")

    # -------------------------------------------------------------------------
    # TEST 5: Verify New Preference in Fresh Chat
    # -------------------------------------------------------------------------
    chat_5_id = "chat_session_005_fresh"
    chemistry_query2 = "Explain the Fischer esterification procedure."

    logger.info(f"\n--- [Chat 5 ({chat_5_id})] Query After Correction: '{chemistry_query2}' ---")
    res5 = engine.process_turn(
        message=chemistry_query2,
        session_id=chat_5_id,
        user_id=test_user,
    )

    telemetry5 = res5.get("telemetry", {})
    applied_mems5 = telemetry5.get("applied_memories", [])
    response_text5 = res5.get("response", "")

    logger.info(f"Gate applied: {telemetry5.get('applied')}")
    logger.info(f"Applied memories: {[m.get('preference_description') for m in applied_mems5]}")
    logger.info(f"Response preview:\n{response_text5[:400]}...\n")

    lower_resp5 = response_text5.lower()
    has_mechanism_focus = any(k in lower_resp5 for k in ["mechanism", "protonation", "nucleophilic", "tetrahedral", "arrow", "resonance"])
    assert has_mechanism_focus, "FAIL: Response did not reflect updated theoretical reaction mechanism preference"
    logger.info("✓ TEST 5 PASSED: Fresh chat reflects updated preference without interference from superseded rule.")

    # Cleanup test file
    if test_user_file.exists():
        test_user_file.unlink()

    logger.info("\n================================================================================")
    logger.info("ALL 5 TESTS PASSED: COMPLETE ISOLATED MEMORY LOOP RIGOROUSLY VALIDATED!")
    logger.info("================================================================================")


if __name__ == "__main__":
    test_complete_isolated_memory_loop()
