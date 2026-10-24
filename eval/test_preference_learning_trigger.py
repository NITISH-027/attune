#!/usr/bin/env python3
"""
Test Suite: Attune Preference Learning Trigger & Persistence Integrity
======================================================================
Validates:
1. Exact failing sentence: "Whenever I ask for a comparison, include one drawback of each option."
2. Dynamic LLM evaluation without closed English cue-phrase list.
3. Persistence across isolated chat sessions with empty dialogue history ([]).
4. Application to in-scope comparison queries & suppression on unrelated queries.
5. Differently worded persistent preferences across unrelated topics.
6. Negative cases:
   - Ordinary questions
   - One-time task/formatting requests
   - Quoted third-party instructions
7. Extraction / Write failure: no false saved-memory promises.
"""

import sys
import os
import json
import logging
import unittest
from unittest.mock import patch
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.engine import AttuneUniversalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_preference_trigger")


class TestPreferenceLearningTrigger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = AttuneUniversalEngine()
        cls.test_user = "test_user_trigger_integrity_42"
        cls.user_mem_file = PROJECT_ROOT / "data" / "user_memories" / f"{cls.test_user}.json"

    def setUp(self):
        # Ensure clean state before each test
        if self.user_mem_file.exists():
            self.user_mem_file.unlink()
        self.engine.v2_engine.clear_user_memories(self.test_user)

    def tearDown(self):
        if self.user_mem_file.exists():
            self.user_mem_file.unlink()
        self.engine.v2_engine.clear_user_memories(self.test_user)

    def test_01_exact_failing_sentence_learning_and_cross_chat_isolation(self):
        """
        Regression Test for exact failing sentence:
        "Whenever I ask for a comparison, include one drawback of each option."
        """
        logger.info("\n=== TEST 1: Regression Test for Exact Failing Sentence ===")
        chat_1_id = "sess_reg_01"
        failing_msg = "Whenever I ask for a comparison, include one drawback of each option."

        # Turn 1: User gives enduring preference
        res1 = self.engine.process_turn(
            session_id=chat_1_id,
            user_id=self.test_user,
            message=failing_msg
        )

        telemetry1 = res1.get("telemetry", {})
        write_status1 = telemetry1.get("memory_write_status")
        new_mem1 = telemetry1.get("newly_learned_memory")

        logger.info(f"Turn 1 Write Status: {write_status1}")
        logger.info(f"Turn 1 Newly Learned: {new_mem1}")

        self.assertIn(write_status1, ["saved", "updated"], f"Expected 'saved' or 'updated', got {write_status1}")
        self.assertIsNotNone(new_mem1, "Expected newly_learned_memory to be present")

        pref_desc = new_mem1.get("preference_description", "").lower()
        scope_desc = new_mem1.get("scope_description", "").lower()
        evidence_quotes = new_mem1.get("evidence_quotes", [])

        logger.info(f"Extracted Description: {pref_desc}")
        logger.info(f"Extracted Scope: {scope_desc}")
        logger.info(f"Evidence Quotes: {evidence_quotes}")

        # Verify exact scope and preference content
        self.assertTrue(
            "drawback" in pref_desc or "downside" in pref_desc or "negative" in pref_desc or "disadvantage" in pref_desc,
            f"Preference description '{pref_desc}' missed 'drawback' requirement"
        )
        self.assertTrue(
            "compar" in scope_desc,
            f"Scope description '{scope_desc}' missed 'comparison' condition"
        )
        # Verify scope was not overly expanded
        self.assertFalse(
            "all questions" in scope_desc or "general science" in scope_desc or "all biology" in scope_desc,
            f"Scope was overly broadened: {scope_desc}"
        )
        # Verify real evidence quote
        self.assertTrue(len(evidence_quotes) > 0, "Expected non-empty evidence quotes")
        self.assertTrue(
            any("whenever" in eq.lower() and "drawback" in eq.lower() for eq in evidence_quotes),
            f"Evidence quotes do not match real user prompt: {evidence_quotes}"
        )

        # Verify GET /api/memory behavior for same user_id
        user_prefs = self.engine.memory.list_voice_preferences(user_id=self.test_user)
        logger.info(f"User Prefs in Storage: {len(user_prefs)} item(s)")
        self.assertEqual(len(user_prefs), 1, "Expected exactly 1 preference in user store")
        self.assertTrue("drawback" in user_prefs[0]["human_readable"].lower())

        # Turn 2: NEW CHAT with same user_id, completely empty history
        chat_2_id = "sess_reg_02_fresh"
        comp_query = "Compare PostgreSQL and MongoDB."
        logger.info(f"\n--- Fresh Chat ({chat_2_id}): '{comp_query}' ---")

        res2 = self.engine.process_turn(
            session_id=chat_2_id,
            user_id=self.test_user,
            message=comp_query
        )

        telemetry2 = res2.get("telemetry", {})
        gate_applied2 = telemetry2.get("applied")
        applied_mems2 = telemetry2.get("applied_memories", [])
        response2 = res2.get("response", "")

        logger.info(f"Gate applied: {gate_applied2}")
        logger.info(f"Applied memories: {[m.get('preference_description') for m in applied_mems2]}")
        logger.info(f"Response preview:\n{response2[:350]}...\n")

        self.assertTrue(gate_applied2, f"Gate should be OPEN for comparison query. Reason: {telemetry2.get('reason')}")
        self.assertTrue(len(applied_mems2) > 0, "Expected applied memories")
        # Verify response text includes drawback/disadvantage
        resp_lower = response2.lower()
        self.assertTrue(
            any(k in resp_lower for k in ["drawback", "downside", "disadvantage", "limitation", "cons"]),
            "Response text did not contain drawback/disadvantage for comparison options"
        )

        # Turn 3: Unrelated question in new chat must NOT apply comparison drawback preference
        chat_3_id = "sess_reg_03_unrelated"
        unrelated_query = "What is the speed of light in vacuum?"
        logger.info(f"\n--- Unrelated Query ({chat_3_id}): '{unrelated_query}' ---")

        res3 = self.engine.process_turn(
            session_id=chat_3_id,
            user_id=self.test_user,
            message=unrelated_query
        )

        telemetry3 = res3.get("telemetry", {})
        gate_applied3 = telemetry3.get("applied")
        logger.info(f"Gate applied: {gate_applied3}")
        logger.info(f"Reason: {telemetry3.get('reason')}")

        self.assertFalse(gate_applied3, "Gate should be SUPPRESSED for unrelated non-comparison question")
        logger.info("✓ TEST 1 PASSED: Regression for exact failing sentence verified and validated.")

    def test_02_differently_worded_preference_unrelated_topic(self):
        """
        Validates differently worded persistent preferences across unrelated topics
        without phrase-specific code.
        """
        logger.info("\n=== TEST 2: Differently Worded Persistent Preference ===")
        chat_id = "sess_diff_worded"
        msg = "For all financial calculations, show the intermediate tax formula before the final result."

        res = self.engine.process_turn(
            session_id=chat_id,
            user_id=self.test_user,
            message=msg
        )

        telemetry = res.get("telemetry", {})
        write_status = telemetry.get("memory_write_status")
        new_mem = telemetry.get("newly_learned_memory")

        logger.info(f"Write Status: {write_status}")
        logger.info(f"New Memory: {new_mem}")

        self.assertIn(write_status, ["saved", "updated"])
        self.assertIsNotNone(new_mem)
        pref_desc = new_mem.get("preference_description", "").lower()
        scope_desc = new_mem.get("scope_description", "").lower()

        self.assertTrue("tax" in pref_desc or "formula" in pref_desc)
        self.assertTrue("financ" in scope_desc or "calculation" in scope_desc)
        logger.info("✓ TEST 2 PASSED: Differently worded preference across unrelated topic learned.")

    def test_03_negative_cases(self):
        """
        Validates that negative cases do NOT become standing user preferences:
        A. Ordinary question
        B. One-time task/formatting request
        C. Quoted third-party instruction
        """
        logger.info("\n=== TEST 3: Negative Cases (Must NOT Become Preferences) ===")

        # Case A: Ordinary question
        msg_a = "What is the boiling point of ethanol at 1 atm?"
        res_a = self.engine.process_turn(
            session_id="sess_neg_a",
            user_id=self.test_user,
            message=msg_a
        )
        status_a = res_a.get("telemetry", {}).get("memory_write_status")
        new_mem_a = res_a.get("telemetry", {}).get("newly_learned_memory")
        logger.info(f"Ordinary Question Status: {status_a}")
        self.assertEqual(status_a, "no_update_needed", f"Ordinary question should have no_update_needed, got {status_a}")
        self.assertIsNone(new_mem_a)

        # Case B: One-time formatting request
        msg_b = "Compare React and Vue, and format the output as a numbered list just for this answer."
        res_b = self.engine.process_turn(
            session_id="sess_neg_b",
            user_id=self.test_user,
            message=msg_b
        )
        status_b = res_b.get("telemetry", {}).get("memory_write_status")
        new_mem_b = res_b.get("telemetry", {}).get("newly_learned_memory")
        logger.info(f"One-Time Request Status: {status_b}")
        self.assertEqual(status_b, "no_update_needed", f"One-time request should have no_update_needed, got {status_b}")
        self.assertIsNone(new_mem_b)

        # Case C: Quoted third-party instruction
        msg_c = "My manager said 'always use uppercase for SQL keywords', but what do you think?"
        res_c = self.engine.process_turn(
            session_id="sess_neg_c",
            user_id=self.test_user,
            message=msg_c
        )
        status_c = res_c.get("telemetry", {}).get("memory_write_status")
        new_mem_c = res_c.get("telemetry", {}).get("newly_learned_memory")
        logger.info(f"Quoted Third-Party Rule Status: {status_c}")
        self.assertEqual(status_c, "no_update_needed", f"Quoted third-party rule should have no_update_needed, got {status_c}")
        self.assertIsNone(new_mem_c)

        # Verify user storage remains completely empty
        user_prefs = self.engine.memory.list_voice_preferences(user_id=self.test_user)
        self.assertEqual(len(user_prefs), 0, "Negative cases must not populate user memory store")
        logger.info("✓ TEST 3 PASSED: All negative cases correctly rejected with no_update_needed.")

    def test_04_extraction_write_failure_integrity(self):
        """
        Validates that on an extraction/write failure:
        1. status is 'failed'
        2. No preference is added to user memory
        3. Response generation does not make false promises of saved memory
        """
        logger.info("\n=== TEST 4: Extraction/Write Failure Integrity ===")
        msg = "Whenever I ask about cars, always mention fuel efficiency first."

        # Simulate a disk write failure by patching save_user_memories to raise an exception
        with patch.object(self.engine.v2_engine, "save_user_memories", side_effect=IOError("Simulated disk write failure")):
            res = self.engine.process_turn(
                session_id="sess_fail_test",
                user_id=self.test_user,
                message=msg
            )

            telemetry = res.get("telemetry", {})
            write_status = telemetry.get("memory_write_status")
            write_error = telemetry.get("memory_write_error")
            new_mem = telemetry.get("newly_learned_memory")
            response = res.get("response", "")

            logger.info(f"Failure Test Write Status: {write_status}")
            logger.info(f"Failure Test Error: {write_error}")
            logger.info(f"Response:\n{response}\n")

            self.assertEqual(write_status, "failed")
            self.assertIsNone(new_mem, "newly_learned_memory must be None on write failure")
            self.assertIsNotNone(write_error)
            # Ensure no secret or full file path is exposed in telemetry error
            self.assertNotIn("D:\\", write_error)
            self.assertNotIn("C:\\", write_error)

            # Ensure response does NOT falsely claim it saved the memory
            resp_lower = response.lower()
            self.assertFalse(
                "i have saved" in resp_lower or "preference is saved" in resp_lower or "i will remember" in resp_lower,
                "Response falsely promised memory persistence when write failed!"
            )

        # Verify storage remains empty
        user_prefs = self.engine.memory.list_voice_preferences(user_id=self.test_user)
        self.assertEqual(len(user_prefs), 0, "Failed write must not leave partial memories in store")
        logger.info("✓ TEST 4 PASSED: Write failure integrity validated (no false promises).")


if __name__ == "__main__":
    unittest.main()
