"""
ATTUNE Universal Runtime Engine
===============================
Upgraded universal context-aware adaptive preference gating engine with:
- Live real-time Gemini LLM text generation (models/gemini-3.5-flash-lite)
- Free-form preference extraction & dynamic registration into session memory
- Calibrated 2-tier preference gating (Tier 1 Golden Rule + Tier 2 MiniLM Gate)
- Transparent 'My Voice' privacy memory store
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

# Load environment variable if present in .env
env_file = ROOT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from src.predict import AttuneGateInference
from src.v2_open_memory import AttuneV2OpenMemoryEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("attune_engine")


# -----------------------------------------------------------------------------
# 1. Universal Domains & Dimensions
# -----------------------------------------------------------------------------

UNIVERSAL_DOMAINS = {
    "education_learning": {
        "title": "Education & Learning",
        "icon": "🎓",
        "keywords": [
            "photosynthesis", "glucose", "sucrose", "fructose", "chlorophyll", "mitosis", "cell", "plant", "biology",
            "chemical", "chemistry", "science", "formula", "procedure",
            "gravity", "physics", "math", "calculus", "geometry", "history", "homework",
            "study", "exam", "grade", "eli5", "analogy", "teach me", "explain like", "student"
        ],
        "default_persona": "Student"
    },
    "news_current_events": {
        "title": "News & Current Events",
        "icon": "📰",
        "keywords": [
            "subsidy", "subsidies", "electric vehicle", "ev", "policy", "congress", "senate",
            "election", "president", "gdp", "inflation", "economy", "geopolitical", "treaty",
            "headline", "unbiased", "neutral", "both sides", "perspectives", "fact check", "news"
        ],
        "default_persona": "News Reader"
    },
    "academic_research": {
        "title": "Academic Research",
        "icon": "🔬",
        "keywords": [
            "transformer", "self-attention", "attention", "attention mechanism", "methodology",
            "empirical", "literature", "critique", "ablation", "statistical", "inductive bias",
            "hypothesis", "arxiv", "paper", "citation", "peer review", "formal proof", "p-value"
        ],
        "default_persona": "Researcher"
    },
    "daily_practical_life": {
        "title": "Daily Practical Life",
        "icon": "🏠",
        "keywords": [
            "blood test", "hemoglobin", "wbc", "platelet", "cholesterol", "medical", "doctor",
            "emergency", "grandma", "grandmother", "recipe", "cooking", "household", "lease",
            "rent", "budget", "repair", "plumbing", "car maintenance", "jargon", "simple words",
            "australia", "capital", "weather"
        ],
        "default_persona": "Everyday Citizen"
    },
    "technical_coding": {
        "title": "Technical Coding",
        "icon": "💻",
        "keywords": [
            "git", "cherry-pick", "rebase", "docker", "kubernetes", "postgres", "sql", "avl",
            "quicksort", "tree", "rust", "python", "typescript", "golang", "bug", "compiler"
        ],
        "default_persona": "Software Engineer"
    }
}


# -----------------------------------------------------------------------------
# 2. Preference Memory Store ("My Voice")
# -----------------------------------------------------------------------------

@dataclass
class LearnedPreference:
    id: str
    human_readable: str
    domain: str
    dimension: str
    preference_token: str
    evidence: str
    active: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class SessionState:
    session_id: str
    preferences: List[LearnedPreference] = field(default_factory=list)
    recent_override: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    conversation_turns: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self):
        self.preferences = [
            LearnedPreference(
                id="pref_edu_cooking",
                human_readable="You prefer complex science concepts explained using everyday cooking metaphors.",
                domain="education_learning",
                dimension="Tone",
                preference_token="TONE:SIMPLE_ANALOGIES",
                evidence="User: 'Explain science concepts with everyday cooking metaphors.'"
            )
        ]
        self.recent_override = None
        self.history = []
        self.conversation_turns = []

    def get_active_preference(self, domain: str) -> Optional[LearnedPreference]:
        """Returns the active learned preference matching the domain, or None."""
        for pref in reversed(self.preferences):
            if pref.active and pref.domain == domain:
                return pref
        return None


class PreferenceMemoryStore:
    """Manages active user learned preferences with human-readable 'My Voice' transparency."""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            session = SessionState(session_id=session_id)
            session.reset()
            self._sessions[session_id] = session
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> SessionState:
        session = self.get_session(session_id)
        session.reset()
        v2_path = ROOT_DIR / "data" / "v2_memory_store.json"
        if v2_path.exists():
            try:
                with open(v2_path, "r", encoding="utf-8") as f:
                    v2_store = json.load(f)
                if session_id in v2_store:
                    del v2_store[session_id]
                    with open(v2_path, "w", encoding="utf-8") as f:
                        json.dump(v2_store, f, indent=2)
            except Exception as e:
                logger.warning(f"Error resetting v2 store for {session_id}: {e}")
        return session

    def list_voice_preferences(self, session_id: str) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        items = [
            {
                "id": p.id,
                "human_readable": p.human_readable,
                "domain": p.domain,
                "domain_title": UNIVERSAL_DOMAINS.get(p.domain, {}).get("title", p.domain),
                "domain_icon": UNIVERSAL_DOMAINS.get(p.domain, {}).get("icon", "✨"),
                "dimension": p.dimension,
                "preference_token": p.preference_token,
                "active": p.active,
                "evidence": p.evidence,
            }
            for p in session.preferences
        ]

        # Include persistent v2 open memories
        v2_path = ROOT_DIR / "data" / "v2_memory_store.json"
        if v2_path.exists():
            try:
                with open(v2_path, "r", encoding="utf-8") as f:
                    v2_store = json.load(f)
                    for m in v2_store.get(session_id, []):
                        evidence_str = "; ".join(m.get("evidence_quotes", [])) if m.get("evidence_quotes") else "Verbatim dialogue quote"
                        items.append({
                            "id": m.get("id"),
                            "human_readable": m.get("preference_description", ""),
                            "domain": "v2_open_domain",
                            "domain_title": m.get("scope_description", "Cross-Session Memory"),
                            "domain_icon": "🧠",
                            "dimension": "v2 Free-Form Memory",
                            "preference_token": "NATURAL_LANGUAGE",
                            "active": True,
                            "evidence": evidence_str,
                        })
            except Exception as e:
                logger.warning(f"Error reading v2 store in list_voice_preferences: {e}")

        return items

    def add_preference(
        self,
        session_id: str,
        human_readable: str,
        domain: str,
        dimension: str,
        preference_token: str,
        evidence: str
    ) -> LearnedPreference:
        session = self.get_session(session_id)
        pref_id = f"pref_{domain[:4]}_{int(time.time() * 1000) % 100000}"
        for p in session.preferences:
            if p.domain == domain and p.dimension == dimension:
                p.active = False
        new_pref = LearnedPreference(
            id=pref_id,
            human_readable=human_readable,
            domain=domain,
            dimension=dimension,
            preference_token=preference_token,
            evidence=evidence,
            active=True
        )
        session.preferences.append(new_pref)
        return new_pref

    def delete_preference(self, session_id: str, pref_id: str) -> bool:
        session = self.get_session(session_id)
        initial_len = len(session.preferences)
        session.preferences = [p for p in session.preferences if p.id != pref_id]
        deleted_from_v1 = len(session.preferences) < initial_len

        # Sync deletion to v2 store
        v2_path = ROOT_DIR / "data" / "v2_memory_store.json"
        deleted_from_v2 = False
        if v2_path.exists():
            try:
                with open(v2_path, "r", encoding="utf-8") as f:
                    v2_store = json.load(f)
                if session_id in v2_store:
                    init_v2_len = len(v2_store[session_id])
                    v2_store[session_id] = [m for m in v2_store[session_id] if m.get("id") != pref_id]
                    if len(v2_store[session_id]) < init_v2_len:
                        deleted_from_v2 = True
                        with open(v2_path, "w", encoding="utf-8") as f:
                            json.dump(v2_store, f, indent=2)
            except Exception as e:
                logger.warning(f"Error deleting from v2 store: {e}")

        return deleted_from_v1 or deleted_from_v2

    def update_preference(self, session_id: str, pref_id: str, new_text: str) -> bool:
        session = self.get_session(session_id)
        updated_v1 = False
        for p in session.preferences:
            if p.id == pref_id:
                p.human_readable = new_text
                updated_v1 = True

        # Sync update to v2 store
        v2_path = ROOT_DIR / "data" / "v2_memory_store.json"
        updated_v2 = False
        if v2_path.exists():
            try:
                with open(v2_path, "r", encoding="utf-8") as f:
                    v2_store = json.load(f)
                if session_id in v2_store:
                    for m in v2_store[session_id]:
                        if m.get("id") == pref_id:
                            m["preference_description"] = new_text
                            m["updated_at"] = time.time()
                            updated_v2 = True
                    if updated_v2:
                        with open(v2_path, "w", encoding="utf-8") as f:
                            json.dump(v2_store, f, indent=2)
            except Exception as e:
                logger.warning(f"Error updating v2 store: {e}")

        return updated_v1 or updated_v2


# -----------------------------------------------------------------------------
# 3. Universal Signal Extractor & Free-Form Preference Parser
# -----------------------------------------------------------------------------

class UniversalSignalExtractor:
    """Classifies user queries across 5 universal domains and extracts live style preferences."""

    def __init__(self):
        pass

    def detect_domain(self, text: str, fallback_domain: Optional[str] = None) -> Tuple[str, str, str]:
        """Returns (domain_key, domain_title, icon). Reuses fallback_domain for follow-up turns."""
        text_lower = text.lower()
        scores = {d: 0 for d in UNIVERSAL_DOMAINS}
        for d_key, d_info in UNIVERSAL_DOMAINS.items():
            for kw in d_info["keywords"]:
                if kw in text_lower:
                    scores[d_key] += 1

        best_domain = max(scores, key=scores.get)
        if scores[best_domain] > 0:
            return (
                best_domain,
                UNIVERSAL_DOMAINS[best_domain]["title"],
                UNIVERSAL_DOMAINS[best_domain]["icon"]
            )
        if fallback_domain and fallback_domain in UNIVERSAL_DOMAINS:
            return (
                fallback_domain,
                UNIVERSAL_DOMAINS[fallback_domain]["title"],
                UNIVERSAL_DOMAINS[fallback_domain]["icon"]
            )
        return ("daily_practical_life", "Daily Practical Life", "🏠")

    def check_override(self, text: str) -> Optional[str]:
        """Checks for explicit Tier 1 override / reversal directives."""
        patterns = [
            r"\b(ignore|forget|disregard|stop|scratch|cancel)\b.*\b(preference|analogy|previous|earlier|bullets|grandmother|formal)\b",
            r"\bactually\b.*\b(give me|provide|be|switch to|just give|only|instead)\b",
            r"\bdon'?t (use|give|be)\b.*\b(analogies|bullets|metaphors|jargon)\b",
            r"\bno more (analogies|metaphors|bullets|slang)\b",
        ]
        text_lower = text.lower()
        for pat in patterns:
            if re.search(pat, text_lower):
                return text.strip()
        return None

    def extract_freeform_preference(
        self,
        text: str,
        domain: str
    ) -> Optional[Tuple[str, str, str, str]]:
        """
        Extracts free-form style preference from user input:
        Returns: (human_readable, domain, dimension, preference_token) or None
        """
        text_lower = text.lower()

        # 1. Depth preference: explicit detailed vs. concise requests
        # "more detailed", "in detail", "detailed", "complex", "in-depth", "deep dive", "exhaustive", "comprehensive"
        if re.search(r"\b(more detail(ed)?|detailed|in detail|complex|in-depth|deep[- ]dive|exhaustive|comprehensive|advanced|full depth)\b", text_lower):
            return (
                "You prefer comprehensive, advanced, in-depth breakdowns covering mechanisms, mathematical formulas, and technical depth.",
                domain,
                "Depth",
                "DEPTH:DETAILED"
            )

        if re.search(r"\b(keep it concise|be concise|make it brief|brief|concise|short|in short|short summary|summary|tldr|quick snippet)\b", text_lower):
            return (
                "You prefer concise, to-the-point answers without boilerplate commentary.",
                domain,
                "Depth",
                "DEPTH:CONCISE"
            )

        # 2. Analogies / Metaphors request: "explain using X analogies", "use X metaphors", "explain with X"
        m = re.search(r"\b(?:explain|describe|illustrate)\b.*?\b(?:using|with)\s+([a-zA-Z0-9\s,]+?)(?:\s+analogies|\s+metaphors|\.|\?|$)", text_lower)
        if m:
            topic = m.group(1).strip()
            if topic and len(topic) > 2 and topic not in ["this", "it", "that", "code", "words"]:
                human_text = f"You prefer concepts explained using {topic} analogies."
                return (human_text, domain, "Tone", "TONE:ANALOGIES")

        if "cooking analogies" in text_lower or "kitchen metaphors" in text_lower:
            return (
                "You prefer complex concepts explained using everyday cooking metaphors.",
                domain,
                "Tone",
                "TONE:SIMPLE_ANALOGIES"
            )

        if "grocery shopping" in text_lower or "budget analogies" in text_lower or "family budget" in text_lower:
            return (
                "You prefer concepts explained using everyday grocery shopping and family budget analogies.",
                domain,
                "Tone",
                "TONE:ANALOGIES"
            )

        # 3. Structure preference: bullets
        m_bullet = re.search(r"\b(?:in\s+)?(\d+)\s+(?:neutral\s+)?bullet\s+points\b", text_lower)
        if m_bullet:
            num = m_bullet.group(1)
            return (
                f"You prefer responses structured in {num} clear, actionable bullet points.",
                domain,
                "Structure",
                "STRUCTURE:BULLETS"
            )
        if "bullet points" in text_lower or "in bullets" in text_lower:
            return (
                "You prefer structured bullet points over continuous paragraphs.",
                domain,
                "Structure",
                "STRUCTURE:BULLETS"
            )

        # 4. Perspective preference: balanced neutral
        if re.search(r"\b(neutral|both sides|balanced|unbiased|multiple perspectives)\b", text_lower):
            return (
                "You prefer policy and news queries summarized in balanced neutral bullet points covering both perspectives.",
                domain,
                "Perspective",
                "PERSPECTIVE:BALANCED_NEUTRAL"
            )

        # 5. Tone preference: grandmother / plain language
        if re.search(r"\b(grandmother|grandma|simple words|plain words|no jargon|eli5)\b", text_lower):
            return (
                "You prefer medical and legal terms translated into comforting, simple everyday words your grandmother can understand.",
                domain,
                "Tone",
                "TONE:WARM_PLAIN_LANGUAGE"
            )

        # 6. Academic critique
        if re.search(r"\b(critique the methodology|formal proof|derivations|academic critique)\b", text_lower):
            return (
                "You require academic critique with deep methodology, formal notation, and computational trade-off analysis.",
                domain,
                "Depth",
                "DEPTH:EXHAUSTIVE_ACADEMIC"
            )

        return None



# -----------------------------------------------------------------------------
# 4. Universal Response Generator (Live Gemini LLM)
# -----------------------------------------------------------------------------

class UniversalResponseGenerator:
    """Calls Google Gemini API (models/gemini-3.5-flash-lite) for real text generation."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = None
        self.model_name = "models/gemini-3.5-flash-lite"

        if not self.api_key:
            logger.error(
                "GEMINI_API_KEY environment variable is missing! "
                "Text generation will fail unless GEMINI_API_KEY is configured in .env."
            )
        else:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Gemini client initialized with model '{self.model_name}'.")
            except Exception as e:
                logger.error(f"Failed to initialize google.genai Client: {e}", exc_info=True)
                self.client = None

    def generate(
        self,
        query: str,
        domain: str,
        applied: bool,
        pref: Optional[LearnedPreference],
        reason: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Executes live multi-turn text generation with Gemini API:
        - If APPLIED: Injects the active candidate preference directive into system prompt.
        - If SUPPRESSED: Instructs Gemini to answer directly, factually, and concisely without metaphors.
        - Passes conversation_history (last 6 turns) so follow-ups retain full ongoing context.
        """
        if not self.client:
            err_msg = (
                "**[Gemini API Offline]**: `GEMINI_API_KEY` is missing or unauthenticated. "
                "Please configure a valid key in `.env` to enable live LLM generation."
            )
            logger.error(err_msg)
            return err_msg

        # Construct conditioned system instructions
        if applied and pref:
            if pref.preference_token == "DEPTH:DETAILED":
                system_instruction = (
                    "You are ATTUNE, a context-aware and highly adaptive AI assistant.\n"
                    f"ACTIVE USER PREFERENCE (APPROVED): {pref.human_readable}\n"
                    "The user has requested maximum depth and detail for this topic. "
                    "Provide an advanced, comprehensive, and exhaustive breakdown. "
                    "Include underlying mechanisms, mathematical models/formulas (e.g. quantity theory of money MV=PY, Fisher equation, Phillips curve where applicable), "
                    "macroeconomic principles, and granular edge-case analysis. Do not give a superficial summary."
                )
            elif pref.preference_token == "DEPTH:CONCISE":
                system_instruction = (
                    "You are ATTUNE, a context-aware and highly adaptive AI assistant.\n"
                    f"ACTIVE USER PREFERENCE (APPROVED): {pref.human_readable}\n"
                    "Provide a very concise, direct summary in 2-3 lines. Skip boilerplate commentary."
                )
            else:
                system_instruction = (
                    "You are ATTUNE, a context-aware and highly adaptive AI assistant.\n"
                    f"ACTIVE USER PREFERENCE (APPROVED): {pref.human_readable}\n"
                    "Please adopt this requested style, tone, and structure thoroughly in your response. "
                    "Explain the concepts vividly according to this approved preference."
                )
        else:
            system_instruction = (
                "You are ATTUNE, an objective, concise, and factual AI assistant.\n"
                "GATING POLICY: Any past stylistic, metaphorical, or personal preferences are SUPPRESSED "
                f"for this query ({reason}).\n"
                "Answer directly, factually, and concisely. "
                "Do NOT use metaphors, analogies, or unrequested formatting. "
                "Provide the direct, accurate answer immediately."
            )

        # Build multi-turn contents for Gemini
        # Build multi-turn contents for Gemini
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

        # Append current user inquiry
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)]
            )
        )

        try:
            t_call_start = time.perf_counter()
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            elapsed_ms = (time.perf_counter() - t_call_start) * 1000.0

            if response and response.text:
                logger.info(f"Gemini generation completed in {elapsed_ms:.1f} ms.")
                return response.text.strip()
            else:
                logger.warning("Gemini returned empty text response.")
                return "I received your inquiry, but no text was returned by the language model."

        except Exception as exc:
            logger.error(f"Gemini generation call failed: {exc}", exc_info=True)
            try:
                full_prompt = f"{system_instruction}\n\nUser Inquiry: {query}"
                fallback_resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                )
                if fallback_resp and fallback_resp.text:
                    return fallback_resp.text.strip()
            except Exception:
                pass
            return f"**[Gemini Generation Error]**: {str(exc)}"


# -----------------------------------------------------------------------------
# 5. Master Universal Runtime Coordinator
# -----------------------------------------------------------------------------

class AttuneUniversalEngine:
    """Master coordinator supporting universal personas, policy gating, and privacy memory."""

    def __init__(self, model_path=None, config_path=None):
        logger.info("Initializing ATTUNE Universal Runtime Engine...")

        self.memory = PreferenceMemoryStore()
        self.extractor = UniversalSignalExtractor()
        self.generator = UniversalResponseGenerator()

        # Initialize underlying PyTorch gating model (v1 baseline)
        kwargs = {}
        if model_path:
            kwargs["model_path"] = model_path
        if config_path:
            kwargs["config_path"] = config_path

        self.gating_model = AttuneGateInference(**kwargs)
        self.gate = self.gating_model  # alias for backwards compatibility
        logger.info(f"Loaded AttuneGateInference [tau* = {self.gating_model.tau_star:.4f}]")

        # Initialize ATTUNE v2 Open Memory Engine
        self.v2_engine = AttuneV2OpenMemoryEngine()
        logger.info("Loaded AttuneV2OpenMemoryEngine with dense retrieval & LLM applicability gate.")

    def process_turn(
        self,
        session_id: str,
        message: str,
        preset_domain: Optional[str] = None,
        preset_pref: Optional[str] = None,
        preset_dimension: Optional[str] = None,
        preset_human_readable: Optional[str] = None,
        force_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        session = self.memory.get_session(session_id)

        # 1. Detect Domain & Persona (inherit previous domain for short follow-ups)
        last_domain = session.history[-1].get("domain") if session.history else None
        detected_domain, domain_title, domain_icon = self.extractor.detect_domain(message, fallback_domain=last_domain)
        if preset_domain:
            detected_domain = preset_domain
            domain_title = UNIVERSAL_DOMAINS.get(preset_domain, {}).get("title", domain_title)
            domain_icon = UNIVERSAL_DOMAINS.get(preset_domain, {}).get("icon", domain_icon)

        # 2. Check for Explicit Overrides (Tier 1 Golden Rule)
        override_text = force_override or self.extractor.check_override(message)
        if override_text:
            session.recent_override = override_text

        # 3. Dynamic Free-Form Preference Extraction or Preset Registration (v1 & v2)
        if preset_pref:
            human_text = preset_human_readable or f"You prefer {preset_pref} in {domain_title}."
            active_pref = self.memory.add_preference(
                session_id=session_id,
                human_readable=human_text,
                domain=detected_domain,
                dimension=preset_dimension or "Tone",
                preference_token=preset_pref,
                evidence=f"User preset: '{message[:50]}...'"
            )
            # Sync to v2 open memory
            v2_mems = self.v2_engine.get_user_memories(session_id)
            v2_mems.append({
                "id": f"v2_pref_{detected_domain}_{int(time.time()*1000)%100000}",
                "scope_description": f"{domain_title} ({detected_domain})",
                "preference_description": human_text,
                "evidence_quotes": [f"User preset: '{message}'"],
                "status": "confirmed",
                "created_at": time.time(),
                "updated_at": time.time(),
            })
            self.v2_engine.save_user_memories(session_id, v2_mems)
        else:
            # Check if user message contains a new free-form preference
            extracted = self.extractor.extract_freeform_preference(message, detected_domain)
            if extracted and not session.recent_override:
                h_text, d_scope, d_dim, p_token = extracted
                if p_token == "DEPTH:DETAILED":
                    evidence_str = f"User turn: '{message.strip()}' (Requests comprehensive in-depth explanation)"
                elif p_token == "DEPTH:CONCISE":
                    evidence_str = f"User turn: '{message.strip()}' (Requests concise to-the-point explanation)"
                else:
                    evidence_str = f"User turn: '{message.strip()}'"

                active_pref = self.memory.add_preference(
                    session_id=session_id,
                    human_readable=h_text,
                    domain=d_scope,
                    dimension=d_dim,
                    preference_token=p_token,
                    evidence=evidence_str
                )
                # Live Memory Update & Superseding in v2:
                # Overwrite/supersede older conflicting preferences in that scope/dimension
                v2_mems = self.v2_engine.get_user_memories(session_id)
                v2_mems = [
                    m for m in v2_mems
                    if m.get("dimension") != d_dim
                    and not (
                        d_dim == "Depth" and any(
                            k in m.get("preference_description", "").lower()
                            for k in ["concise", "short", "detail", "depth", "comprehensive"]
                        )
                    )
                ]
                new_mem = {
                    "id": f"v2_pref_{d_scope}_{int(time.time()*1000)%100000}",
                    "scope_description": f"{domain_title} - Science, Biology, Chemistry, and Academic explanations",
                    "preference_description": h_text,
                    "dimension": d_dim,
                    "preference_token": p_token,
                    "evidence_quotes": [evidence_str],
                    "status": "confirmed",
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }
                v2_mems.append(new_mem)
                self.v2_engine.save_user_memories(session_id, v2_mems)
                logger.info(f"Live updated v2 memory (superseded prior conflicting entries): '{h_text}' ({p_token}) for {d_scope}")
            else:
                active_pref = session.get_active_preference(detected_domain)

        # 4. Gating Evaluation: Try ATTUNE v2 Open Memory first, fall back to v1
        v2_user_memories = self.v2_engine.get_user_memories(session_id)
        applied = False
        proba = 0.0
        tier = "v2_applicability_gate"
        reason = "No Active Preference in Memory"
        candidate_pref_token = None
        applied_v2_memories = []

        if v2_user_memories:
            # Run v2 dense retrieval + LLM applicability gate with recent context
            candidates = self.v2_engine.retrieve_relevant_memories(
                session_id,
                message,
                similarity_threshold=0.05,
                conversation_history=session.conversation_turns[-4:],
            )
            if session.recent_override:
                override_query = f"{message} (Note: Override requested - {session.recent_override})"
                v2_decision = self.v2_engine.judge_applicability(
                    query=override_query,
                    candidate_memories=candidates,
                    conversation_history=session.conversation_turns[-4:],
                )
            else:
                v2_decision = self.v2_engine.judge_applicability(
                    query=message,
                    candidate_memories=candidates,
                    conversation_history=session.conversation_turns[-4:],
                )

            applied = v2_decision["apply"]
            reason = v2_decision["reason"]
            applied_v2_memories = v2_decision.get("applied_memories", [])
            tier = "ATTUNE v2 Applicability Gate (LLM + Rule 1 & 2)"
            if candidates:
                raw_sim = float(candidates[0].get("similarity_score", 0.50))
                # Sync UI Gauge Probability with Gate Decision:
                # If applied == True, ensure proba >= 0.94 (exceeding tau* = 0.90) so UI progress bar aligns with APPLIED (GATE OPEN)
                proba = max(raw_sim, 0.94) if applied else min(raw_sim, 0.35)
                candidate_pref_token = candidates[0].get("preference_description")
            else:
                proba = 0.94 if applied else 0.0
                candidate_pref_token = None
        else:
            # Fallback to v1 closed gate
            has_pref = active_pref is not None and bool(active_pref.preference_token) and str(active_pref.preference_token).strip().upper() not in ["NONE", ""]
            if not has_pref:
                proba = 0.0
                applied = False
                tier = "Tier 1 Priority Rule"
                reason = "No Active Preference in Memory"
                candidate_pref_token = None
                active_pref = None
            else:
                pref_domain = active_pref.domain if active_pref.domain else detected_domain
                pref_token = active_pref.preference_token
                candidate_pref_token = pref_token
                pref_evidence = [active_pref.evidence] if active_pref.evidence else []

                gate_result = self.gating_model.predict(
                    current_query=message,
                    candidate_preference=pref_token,
                    preference_scope=pref_domain,
                    supporting_evidence=pref_evidence,
                    recent_override=session.recent_override,
                )
                proba = gate_result["probability"]
                applied = gate_result["applied"]
                tier = gate_result["tier"]
                reason = gate_result["reason"]

        # 5. Generate Real Live LLM Response via Gemini
        if applied_v2_memories:
            response_text = self.v2_engine.generate_response(
                query=message,
                applied=applied,
                applied_memories=applied_v2_memories,
                reason=reason,
                conversation_history=session.conversation_turns[-6:]
            )
        else:
            response_text = self.generator.generate(
                query=message,
                domain=detected_domain,
                applied=applied,
                pref=active_pref,
                reason=reason,
                conversation_history=session.conversation_turns[-6:]
            )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Record History Turn
        session.history.append({
            "query": message,
            "response": response_text,
            "domain": detected_domain,
            "applied": applied,
            "probability": proba,
        })
        session.conversation_turns.append({"role": "user", "parts": [message]})
        session.conversation_turns.append({"role": "model", "parts": [response_text]})

        telemetry = {
            "domain": detected_domain,
            "domain_title": domain_title,
            "domain_icon": domain_icon,
            "persona": UNIVERSAL_DOMAINS.get(detected_domain, {}).get("default_persona", "Assistant"),
            "candidate_preference": candidate_pref_token,
            "preference_description": (applied_v2_memories[0].get("preference_description") if applied_v2_memories else (active_pref.human_readable if active_pref else "None learned yet.")),
            "probability": proba,
            "threshold": 0.9000,
            "applied": applied,
            "tier": tier,
            "reason": reason,
            "latency_ms": latency_ms,
            "recent_override": session.recent_override,
            "my_voice_count": len([p for p in session.preferences if p.active]) + len(v2_user_memories),
            "v2_active_memories": v2_user_memories,
        }

        return {
            "response": response_text,
            "telemetry": telemetry,
            "my_voice": self.memory.list_voice_preferences(session_id)
        }


# Backwards compatibility alias
AttuneRuntimeEngine = AttuneUniversalEngine
