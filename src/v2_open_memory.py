"""
ATTUNE v2: Open-Ended Natural-Language Preference Memory & Gating Loop
======================================================================
Removes the 8-token / 4-scope ceiling of v1 by representing user preferences
as free-form natural language backed by verbatim evidence quotes, dense
embedding retrieval (all-MiniLM-L6-v2), and LLM applicability gating.

Architecture:
- Step A: Learn & Save (infer_and_update_memory)
  Uses Gemini with structured JSON schema to extract/update free-form memories.
  Distinguishes one-time requests from explicit/repeated preferences.
  Supports memory correction/superseding when contradicted.
  Persists per-user memory store to disk (data/v2_memory_store.json).
- Step B: Retrieve (retrieve_relevant_memories)
  Dense vector retrieval over (scope_description + preference_description)
  using sentence-transformers/all-MiniLM-L6-v2.
- Step C: Judge Applicability (judge_applicability)
  Gemini-driven binary gating decision (apply: true/false, reason: str):
  * Rule 1: Current explicit instructions in the query strictly override any stored memory.
  * Rule 2: Apply only if the query genuinely matches scope and intent; otherwise suppress.
- Step D: Answer Generation (generate_response)
  Generates response conditioned ONLY on memories judged apply: true.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load .env if present
env_file = ROOT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

logger = logging.getLogger("attune_v2")

DEFAULT_V2_STORE_PATH = ROOT_DIR / "data" / "v2_memory_store.json"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


# -----------------------------------------------------------------------------
# Structured Schemas for Gemini Inference & Extraction
# -----------------------------------------------------------------------------

class MemoryItem(BaseModel):
    id: str = Field(description="Unique identifier for the memory, e.g. mem_science_analogy")
    scope_description: str = Field(description="Natural language condition/context describing when this preference applies")
    preference_description: str = Field(description="Natural language description of the requested style, format, or behavior")
    evidence_quotes: List[str] = Field(description="Verbatim user quotes from the dialogue demonstrating this preference")
    status: str = Field(description="'confirmed' if explicitly requested or repeated, 'candidate' if single weak hint", default="confirmed")


class MemoryOperationItem(BaseModel):
    action: str = Field(default="add", description="'add' to create, 'update' to supersede/change, 'delete' to remove")
    id: str = Field(default="", description="Unique identifier for the memory, e.g. mem_science_analogy")
    scope_description: str = Field(default="General", description="Natural language condition describing when this applies")
    preference_description: str = Field(default="", description="Natural language description of requested style")
    evidence_quotes: List[str] = Field(default_factory=list, description="Verbatim user quotes demonstrating preference")
    status: str = Field(default="confirmed", description="'confirmed' or 'candidate'")


class MemoryExtractionPayload(BaseModel):
    has_persistent_preference: bool = Field(description="True only if user demonstrated a persistent preference, False for one-time tasks")
    operations: List[MemoryOperationItem] = Field(
        default_factory=list,
        description="List of memory items with action and item details"
    )
    analysis: str = Field(description="Brief explanation distinguishing persistent preference vs one-time prompt")


class GatingDecisionSchema(BaseModel):
    apply: bool = Field(description="Whether to apply any stored memory to this query")
    reason: str = Field(description="Gating reason citing Rule 1 (override) or Rule 2 (scope match/suppression)")
    applicable_memory_ids: List[str] = Field(default_factory=list, description="IDs of memories that should be applied")


# -----------------------------------------------------------------------------
# V2 Open Memory Engine Implementation
# -----------------------------------------------------------------------------

class AttuneV2OpenMemoryEngine:
    """Complete Cross-Session Free-Form Preference Memory & Gating Engine."""

    def __init__(
        self,
        storage_path: Union[str, Path] = DEFAULT_V2_STORE_PATH,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm_model_name: str = DEFAULT_GEMINI_MODEL,
        api_key: Optional[str] = None,
    ):
        self.storage_path = Path(storage_path)
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        self._embedder = None
        self._gemini_client = None

        # Initialize storage file if missing
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_store({})

    # -------------------------------------------------------------------------
    # Lazy Loaders
    # -------------------------------------------------------------------------

    @property
    def embedder(self):
        """Lazy loads SentenceTransformer for dense retrieval."""
        if self._embedder is None:
            logger.info(f"Loading dense embedding backbone '{self.embedding_model_name}'...")
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @property
    def client(self):
        """Lazy loads Gemini client."""
        if self._gemini_client is None:
            if not self.api_key:
                logger.error("GEMINI_API_KEY is not configured! LLM calls will fail.")
            else:
                try:
                    from google import genai
                    self._gemini_client = genai.Client(api_key=self.api_key)
                    logger.info(f"Gemini client initialized with model '{self.llm_model_name}'.")
                except Exception as e:
                    logger.error(f"Failed to initialize google.genai: {e}", exc_info=True)
        return self._gemini_client

    def _safe_generate(self, contents, config=None, max_retries: int = 5):
        """Executes generate_content with automatic exponential backoff for rate limits."""
        for attempt in range(max_retries):
            try:
                time.sleep(1.0)  # Gentle pacing between requests
                kwargs = {"model": self.llm_model_name, "contents": contents}
                if config:
                    kwargs["config"] = config
                return self.client.models.generate_content(**kwargs)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    delay = 3.0 * (attempt + 1)
                    logger.warning(f"Gemini API rate limited (429). Waiting {delay:.1f}s (retry {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    raise e
        raise RuntimeError("Exceeded maximum retries for Gemini API call.")

    # -------------------------------------------------------------------------
    # Disk Persistence (data/v2_memory_store.json)
    # -------------------------------------------------------------------------

    def _read_store(self) -> Dict[str, List[Dict[str, Any]]]:
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading memory store {self.storage_path}: {e}")
        return {}

    def _write_store(self, store: Dict[str, List[Dict[str, Any]]]):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.storage_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=2, ensure_ascii=False)
            temp_path.replace(self.storage_path)
        except Exception as e:
            logger.error(f"Error writing memory store {self.storage_path}: {e}")

    def get_user_memories(self, user_id: str) -> List[Dict[str, Any]]:
        store = self._read_store()
        return store.get(user_id, [])

    def save_user_memories(self, user_id: str, memories: List[Dict[str, Any]]):
        store = self._read_store()
        store[user_id] = memories
        self._write_store(store)

    def clear_user_memories(self, user_id: str):
        store = self._read_store()
        if user_id in store:
            del store[user_id]
            self._write_store(store)

    # -------------------------------------------------------------------------
    # Step A: Learn & Save (infer_and_update_memory)
    # -------------------------------------------------------------------------

    def infer_and_update_memory(
        self,
        user_id: str,
        dialogue: Union[List[Dict[str, str]], str],
        existing_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyzes a completed dialogue session using Gemini with structured JSON.
        - Extracts free-form natural language preferences.
        - Backs entries with verbatim evidence quotes.
        - Distinguishes one-time situational instructions from persistent preferences.
        - Supports memory correction: updates or supersedes conflicting memories.
        - Persists to data/v2_memory_store.json.
        """
        if existing_memories is None:
            memories = self.get_user_memories(user_id)
        else:
            memories = list(existing_memories)

        # Format dialogue into text
        if isinstance(dialogue, list):
            dialogue_text = "\n".join(
                f"{turn.get('role', 'user').capitalize()}: {turn.get('content', '')}"
                for turn in dialogue
            )
        else:
            dialogue_text = str(dialogue)

        if not dialogue_text.strip():
            return memories

        prompt = f"""You are the ATTUNE v2 Cross-Session Preference Memory Engine.
Analyze the following completed conversation dialogue between a user and an assistant.

GOAL: Extract persistent, cross-session user preferences about response style, tone, format, or methodology.

STRICT RULES:
1. DISTINGUISH ONE-TIME REQUESTS FROM PERSISTENT PREFERENCES:
   - If the user asks for a specific one-off task (e.g. "write a poem about cats", "summarize this article in 3 lines", "help me fix this syntax error"), that is a ONE-TIME request. Do NOT save it as memory.
   - If the user states an ongoing preference (e.g. "always explain science using cooking metaphors", "I prefer balanced two-column tables for news", "never include boilerplate code", "from now on...", "going forward..."), or repeatedly asks for the same style across turns, that is a PERSISTENT preference. Save it.
2. VERBATIM EVIDENCE:
   - In `evidence_quotes`, include the EXACT words spoken by the user that prove this preference.
3. MEMORY CORRECTION & SUPERSEDING:
   - Here are the user's CURRENT active memories:
{json.dumps(memories, indent=2)}
   - If the user explicitly changes, contradicts, or reverses a prior preference (e.g. "Actually, stop using cooking metaphors, use academic formulas instead"), UPDATE or REPLACE the existing memory entry. Do NOT accumulate conflicting rules.
4. NATURAL LANGUAGE DESCRIPTIONS:
   - `scope_description`: Clear natural language describing when the preference applies (e.g., "When explaining scientific or educational concepts").
   - `preference_description`: Clear natural language describing the desired response style (e.g., "Use everyday cooking and kitchen metaphors").

DIALOGUE TO ANALYZE:
{dialogue_text}

Provide output strictly adhering to the JSON schema."""

        try:
            from google.genai import types

            response = self._safe_generate(
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=MemoryExtractionPayload,
                    temperature=0.1,
                ),
            )

            raw_text = response.text.strip()
            data = json.loads(raw_text)

            if not data.get("has_persistent_preference", False):
                logger.info(f"V2 Memory Engine: No persistent preference found in dialogue. Analysis: {data.get('analysis')}")
                return memories

            operations = data.get("operations", [])
            for op in operations:
                action = op.get("action", "add").lower()
                target_id = op.get("id") or op.get("target_id")

                if action in ["delete", "remove"]:
                    memories = [m for m in memories if m.get("id") != target_id]
                    logger.info(f"V2 Memory: Deleted memory {target_id}")

                elif action in ["update", "supersede", "modify"]:
                    updated = False
                    for m in memories:
                        if m.get("id") == target_id or (
                            op.get("scope_description") and
                            op.get("scope_description").lower() in m.get("scope_description", "").lower()
                        ):
                            m["scope_description"] = op.get("scope_description", m.get("scope_description"))
                            m["preference_description"] = op.get("preference_description", m.get("preference_description"))
                            m["evidence_quotes"] = op.get("evidence_quotes", m.get("evidence_quotes", []))
                            m["status"] = op.get("status", "confirmed")
                            m["updated_at"] = time.time()
                            updated = True
                            logger.info(f"V2 Memory: Updated memory {m.get('id')}")
                            break
                    if not updated and op.get("preference_description"):
                        # If not found, add as new
                        new_id = target_id or f"mem_{int(time.time() * 1000) % 100000}"
                        memories.append({
                            "id": new_id,
                            "scope_description": op.get("scope_description", "General"),
                            "preference_description": op.get("preference_description", ""),
                            "evidence_quotes": op.get("evidence_quotes", []),
                            "status": op.get("status", "confirmed"),
                            "created_at": time.time(),
                            "updated_at": time.time(),
                        })

                elif action == "add":
                    if op.get("preference_description"):
                        new_id = target_id or f"mem_{int(time.time() * 1000) % 100000}"
                        # Check if duplicate or contradicting existing scope
                        existing_idx = None
                        for i, m in enumerate(memories):
                            if m.get("scope_description", "").strip().lower() == op.get("scope_description", "").strip().lower():
                                existing_idx = i
                                break
                        if existing_idx is not None:
                            # Update existing rather than creating duplicate scope
                            memories[existing_idx] = {
                                "id": memories[existing_idx]["id"],
                                "scope_description": op.get("scope_description", memories[existing_idx]["scope_description"]),
                                "preference_description": op.get("preference_description"),
                                "evidence_quotes": op.get("evidence_quotes", []),
                                "status": op.get("status", "confirmed"),
                                "created_at": memories[existing_idx].get("created_at", time.time()),
                                "updated_at": time.time(),
                            }
                            logger.info(f"V2 Memory: Replaced matching scope memory {memories[existing_idx]['id']}")
                        else:
                            memories.append({
                                "id": new_id,
                                "scope_description": op.get("scope_description", "General"),
                                "preference_description": op.get("preference_description"),
                                "evidence_quotes": op.get("evidence_quotes", []),
                                "status": op.get("status", "confirmed"),
                                "created_at": time.time(),
                                "updated_at": time.time(),
                            })
                            logger.info(f"V2 Memory: Added new memory {new_id}")

            self.save_user_memories(user_id, memories)
            return memories

        except Exception as exc:
            logger.error(f"Error in infer_and_update_memory: {exc}", exc_info=True)
            return memories

    # -------------------------------------------------------------------------
    # Step B: Retrieve (retrieve_relevant_memories)
    # -------------------------------------------------------------------------

    def retrieve_relevant_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.05,
        memories_override: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves user's stored memories using dense vector similarity
        over `scope_description + preference_description` with all-MiniLM-L6-v2.
        """
        memories = memories_override if memories_override is not None else self.get_user_memories(user_id)
        if not memories:
            return []

        # If user has a small set of confirmed memories (<= top_k), return all of them
        # so the LLM Applicability Gate can perform full contextual gating without false drops
        if len(memories) <= top_k:
            scored_memories = []
            for m in memories:
                m_copy = dict(m)
                m_copy["similarity_score"] = float(m.get("similarity_score", 0.95))
                scored_memories.append(m_copy)
            return scored_memories

        # Construct textual representations for embedding
        mem_texts = [
            f"Scope: {m.get('scope_description', '')}. Preference: {m.get('preference_description', '')}"
            for m in memories
        ]

        query_emb = self.embedder.encode(query, normalize_embeddings=True)
        mem_embs = self.embedder.encode(mem_texts, normalize_embeddings=True)

        # Dot product of normalized vectors = cosine similarity
        similarities = np.dot(mem_embs, query_emb)

        scored_memories = []
        for i, score in enumerate(similarities):
            mem_copy = dict(memories[i])
            mem_copy["similarity_score"] = float(score)
            if score >= similarity_threshold:
                scored_memories.append(mem_copy)

        # If all were below threshold but memories exist, include top memory for LLM evaluation
        if not scored_memories and memories:
            best_idx = int(np.argmax(similarities))
            best_copy = dict(memories[best_idx])
            best_copy["similarity_score"] = float(similarities[best_idx])
            scored_memories.append(best_copy)

        scored_memories.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_memories[:top_k]

    # -------------------------------------------------------------------------
    # Step C: Judge Applicability (judge_applicability)
    # -------------------------------------------------------------------------

    def judge_applicability(
        self,
        query: str,
        candidate_memories: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Gemini-driven gating decision evaluating candidate memories against current query:
        - Rule 1: Current explicit instructions in query strictly override any stored memory.
                  Pure 1-token symbol/formula/lookup requests are excluded (suppressed).
        - Rule 2: Topic Scope Matching (NOT style-keyword matching). Preferences persist across turns
                  within the same topic scope without requiring the user to repeat style keywords.
        """
        if not candidate_memories:
            return {
                "apply": False,
                "reason": "No relevant stored memories retrieved (out of scope).",
                "applicable_memory_ids": [],
                "tier": "v2_empty_retrieval",
                "applied_memories": [],
            }

        # Build recent conversation context lines if available
        history_lines = []
        if conversation_history:
            for turn in conversation_history[-4:]:
                role = "User" if turn.get("role") == "user" else "Assistant"
                parts = turn.get("parts", [])
                if isinstance(parts, list) and parts:
                    txt = str(parts[0])
                else:
                    txt = str(turn.get("content", turn.get("message", "")))
                if len(txt) > 200:
                    txt = txt[:200] + "..."
                if txt.strip():
                    history_lines.append(f"{role}: {txt.strip()}")

        history_context = ""
        if history_lines:
            history_context = "RECENT CONVERSATION CONTEXT:\n" + "\n".join(history_lines) + "\n\n"

        prompt = f"""You are the ATTUNE v2 Context-Aware Preference Applicability Gate.
Evaluate whether any of the user's stored preferences should be applied to the current user query.

{history_context}CURRENT QUERY:
"{query}"

RETRIEVED CANDIDATE PREFERENCES:
{json.dumps(candidate_memories, indent=2)}

TWO GOLDEN GATING RULES & CORE PRINCIPLES:
1. PURPOSE OF STORED MEMORY:
   A stored preference exists precisely so the user does NOT have to repeat style words (like "in detail", "in short", "use analogies") on subsequent turns. Once established, the preference remains active and carries forward across all subsequent questions within that topic scope.

2. RULE 1 (CONTRADICTING OVERRIDE & PURE FORMULA/SYMBOL EXCLUSION):
   - CONTRADICTING OVERRIDE: ONLY suppress (apply: false) if the user's current query explicitly CONTRADICTS the candidate memory (e.g. stored memory is "in detail", but user says "give a 1-word answer"). If the current query requests, updates, or reinforces the same style (e.g., "in short explain" with the concise preference, or "explain in detail" with the detailed preference), you MUST return apply: true! DO NOT mark it false as redundant.
   - PURE FORMULA / SYMBOL EXCLUSION: If the user query is asking for a pure factual 1-token symbol, chemical formula, single number, or raw definition lookup (e.g., "What is the exact chemical formula for glucose?", "What is the atomic number of Gold?"), DO NOT apply stylistic elaboration or long prose (apply: false, SUPPRESSED). Return neutral direct factual answer.

3. RULE 2 (TOPIC SCOPE MATCHING, NOT STYLE-KEYWORD MATCHING):
   - Check if the TOPIC/SUBJECT of the current query (resolved in the context of recent conversation history, e.g. follow-up questions about sucrose after glucose, or asking "where sucrose is used") falls within the `scope_description` of the candidate memory (e.g., Science / Biology / Chemistry / Education / Technical explanations).
   - CRITICAL REQUIREMENT: Even if the current query is a normal question like "now for sucrose say" or "where sucrose is used" with NO style keywords in it, if its topic matches `scope_description` (and it is not a pure formula/symbol lookup), you MUST return apply: true!
   - DO NOT treat follow-up queries like "now for sucrose say" as incomplete fragments; understand them through the recent conversation context.
   - ONLY return apply: false (SUPPRESSED) if:
     a) The query switches to an unrelated topic outside `scope_description` (e.g., asking for movie recommendations or router configuration while memory is for science/education), OR
     b) The query triggers Rule 1 (explicit override or pure symbol/formula lookup).

Output JSON with:
- "apply": boolean (true if at least one memory should be applied, false if suppressed)
- "reason": concise explanation citing Rule 1 or Rule 2 (explain why the topic matches the scope, or why it was suppressed/overridden)
- "applicable_memory_ids": list of memory IDs to apply (empty if apply is false)"""

        try:
            from google.genai import types

            response = self._safe_generate(
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GatingDecisionSchema,
                    temperature=0.0,
                ),
            )

            decision_data = json.loads(response.text.strip())
            apply_decision = bool(decision_data.get("apply", False))
            reason = decision_data.get("reason", "Evaluated by v2 Applicability Gate.")
            applicable_ids = decision_data.get("applicable_memory_ids", [])

            applied_memories = [m for m in candidate_memories if m.get("id") in applicable_ids]
            if apply_decision and not applied_memories:
                applied_memories = candidate_memories[:1]

            return {
                "apply": apply_decision,
                "reason": reason,
                "applicable_memory_ids": applicable_ids,
                "tier": "v2_applicability_gate",
                "applied_memories": applied_memories,
            }

        except Exception as exc:
            logger.error(f"Error in judge_applicability: {exc}", exc_info=True)
            # Safe fallback: Suppress rather than over-apply
            return {
                "apply": False,
                "reason": f"Gating evaluation error: {str(exc)}",
                "applicable_memory_ids": [],
                "tier": "v2_fallback_suppression",
                "applied_memories": [],
            }

    # -------------------------------------------------------------------------
    # Step D: Answer Generation (generate_response)
    # -------------------------------------------------------------------------

    def generate_response(
        self,
        query: str,
        applied: bool,
        applied_memories: List[Dict[str, Any]],
        reason: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generates final answer conditioned ONLY on memories judged apply: true.
        If suppressed, generates direct, objective, factual response without unrequested styles.
        """
        if not self.client:
            return "**[Gemini API Offline]**: Cannot generate response without GEMINI_API_KEY."

        if applied and applied_memories:
            pref_descriptions = "\n".join(
                f"- Preference: {m.get('preference_description')} (Scope: {m.get('scope_description')})"
                for m in applied_memories
            )
            system_instruction = (
                "You are ATTUNE v2, an adaptive, context-aware AI assistant.\n"
                "The user has active, approved preferences applicable to this inquiry:\n"
                f"{pref_descriptions}\n"
                "Adopt these preferences naturally and thoroughly in your response."
            )
        else:
            system_instruction = (
                "You are ATTUNE v2, an objective, concise, and helpful AI assistant.\n"
                f"GATING POLICY: Any prior stylistic or formatting preferences are SUPPRESSED for this query ({reason}).\n"
                "Answer the user's request directly, factually, and concisely without assuming unrequested styles, metaphors, or formatting."
            )

        from google.genai import types

        contents = []
        if conversation_history:
            for turn in conversation_history[-6:]:
                role = turn.get("role", "user")
                if role not in ["user", "model"]:
                    role = "user"
                parts = turn.get("parts", [])
                text_parts = []
                for p in parts:
                    if isinstance(p, dict) and "text" in p:
                        text_parts.append(str(p["text"]))
                    elif isinstance(p, str):
                        text_parts.append(p)
                    else:
                        text_parts.append(str(p))
                turn_text = "\n".join(text_parts).strip()
                if turn_text:
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=turn_text)]
                        )
                    )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)]
            )
        )

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
            resp = self._safe_generate(
                contents=contents,
                config=config,
            )
            return resp.text.strip() if resp and resp.text else "No response returned."
        except Exception as exc:
            logger.error(f"Error in generate_response: {exc}", exc_info=True)
            return f"**[Gemini Generation Error]**: {str(exc)}"

    # -------------------------------------------------------------------------
    # Full Turn Processing (for runtime & evaluation)
    # -------------------------------------------------------------------------

    def process_turn(
        self,
        user_id: str,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Executes full v2 lifecycle for an incoming turn."""
        t0 = time.perf_counter()

        # Step B: Dense retrieval
        candidates = self.retrieve_relevant_memories(user_id=user_id, query=query)

        # Step C: Gating decision
        gate_res = self.judge_applicability(query=query, candidate_memories=candidates)

        applied = gate_res["apply"]
        reason = gate_res["reason"]
        applied_memories = gate_res["applied_memories"]

        # Step D: Response generation
        response_text = self.generate_response(
            query=query,
            applied=applied,
            applied_memories=applied_memories,
            reason=reason,
            conversation_history=conversation_history,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "response": response_text,
            "applied": applied,
            "reason": reason,
            "candidate_memories": candidates,
            "applied_memories": applied_memories,
            "latency_ms": latency_ms,
            "user_memories": self.get_user_memories(user_id),
        }
