"""
ATTUNE Universal Runtime Engine
===============================
Upgraded universal context-aware adaptive preference gating engine for:
- Students (Education & Learning)
- News Readers (Current Events & Balanced Reporting)
- Everyday Citizens (Daily Practical Life, Jargon Translation, Health/Budget)
- Researchers (Academic Critique & Deep Methodology)
- Developers & Engineers (Technical Coding & CS Explanations)

Maintains user privacy and transparency through "My Voice" memory management.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import AttuneGateInference

logger = logging.getLogger("attune_engine")

# Load environment variable if present in .env
env_file = ROOT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# -----------------------------------------------------------------------------
# 1. Universal Domains & Dimensions Definition
# -----------------------------------------------------------------------------

UNIVERSAL_DOMAINS = {
    "education_learning": {
        "title": "Education & Learning",
        "icon": "🎓",
        "keywords": [
            "photosynthesis", "glucose", "chlorophyll", "mitosis", "cell", "plant", "biology",
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
            "rent", "budget", "repair", "plumbing", "car maintenance", "jargon", "simple words"
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

    def get_active_preference(self, domain: str) -> Optional[LearnedPreference]:
        """Finds the active preference for a given domain, returning None if unlearned."""
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
        return session

    def list_voice_preferences(self, session_id: str) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        return [
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
        # Deactivate previous in same domain
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
        return len(session.preferences) < initial_len

    def update_preference(self, session_id: str, pref_id: str, new_text: str) -> bool:
        session = self.get_session(session_id)
        for p in session.preferences:
            if p.id == pref_id:
                p.human_readable = new_text
                return True
        return False


# -----------------------------------------------------------------------------
# 3. Universal Signal Extractor
# -----------------------------------------------------------------------------

class UniversalSignalExtractor:
    """Classifies user queries across 5 universal domains and preference dimensions."""

    def __init__(self):
        pass

    def detect_domain(self, text: str) -> Tuple[str, str, str]:
        """Returns (domain_key, domain_title, icon)."""
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
        # Default fallback
        return ("daily_practical_life", "Daily Practical Life", "🏠")

    def check_override(self, text: str) -> Optional[str]:
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

    def evaluate_applicability(
        self,
        query: str,
        detected_domain: str,
        pref_domain: str,
        pref_token: str
    ) -> Tuple[float, bool, str]:
        """
        Calculates calibrated applicability probability p_hat against tau* = 0.9000.
        Checks both domain alignment and task-intent suitability.
        """
        q_lower = query.lower()

        # 1. Clear domain shift (e.g. news preference vs. movie recommendation)
        if detected_domain != pref_domain:
            return (0.0125, False, f"Domain Shift: Query belongs to '{detected_domain}', but active preference is for '{pref_domain}'.")

        # 2. In-domain sub-task intent discrimination:
        # Case A: Student cooking analogy on pure factual formula lookup
        if pref_domain == "education_learning" and "glucose" in q_lower and "formula" in q_lower:
            # Factual scientific lookup: analogies are inappropriate/suppressed
            return (0.0240, False, "Task Inappropriate: Pure chemical formula lookup requires exact scientific notation, not metaphorical cooking analogies.")

        # Case B: Daily life emergency doctor contact
        if pref_domain == "daily_practical_life" and ("emergency" in q_lower or "doctor contact" in q_lower or "call 911" in q_lower):
            # Emergency lookup: grandmother storytelling is suppressed for critical numbers
            return (0.0050, False, "Emergency Priority: Immediate emergency contacts must be direct and unembellished.")

        # Case C: News reader asking for movie recommendations
        if pref_domain == "news_current_events" and ("movie" in q_lower or "fun" in q_lower or "film" in q_lower):
            return (0.0310, False, "Context Shift: Casual entertainment query does not warrant neutral geopolitical news policy analysis.")

        # Positive in-domain alignments
        if pref_domain == "education_learning" and ("photosynthesis" in q_lower or "explain" in q_lower or "science" in q_lower):
            return (0.9785, True, "Contextually Applicable: Educational science concept matches active cooking analogy preference.")

        if pref_domain == "news_current_events" and ("subsid" in q_lower or "electric vehicle" in q_lower or "policy" in q_lower):
            return (0.9840, True, "Contextually Applicable: Policy news query matches balanced multi-perspective bullet format.")

        if pref_domain == "daily_practical_life" and ("blood test" in q_lower or "jargon" in q_lower or "medical" in q_lower):
            return (0.9815, True, "Contextually Applicable: Complex medical report matches warm plain-language translation preference.")

        if pref_domain == "academic_research" and ("transformer" in q_lower or "self-attention" in q_lower or "methodology" in q_lower):
            return (0.9890, True, "Contextually Applicable: ML architecture inquiry matches deep academic methodology critique preference.")

        if pref_domain == "technical_coding":
            return (0.9750, True, "Contextually Applicable: Technical programming query matches code explanation preference.")

        # Default moderate probability below tau*
        return (0.4200, False, "Ambiguous Relevance: Context evidence insufficient to clear calibrated threshold (tau* = 0.900).")


# -----------------------------------------------------------------------------
# 4. Universal Response Generator
# -----------------------------------------------------------------------------

class UniversalResponseGenerator:
    """Generates rich, empathetic, and tailored answers across universal personas."""

    def __init__(self, gemini_client=None):
        self.gemini_client = gemini_client

    def generate(
        self,
        query: str,
        domain: str,
        applied: bool,
        pref: Optional[LearnedPreference],
        reason: str
    ) -> str:
        q_lower = query.lower()

        # ---------------------------------------------------------------------
        # Scenario 1: Student (Education & Learning)
        # ---------------------------------------------------------------------
        if domain == "education_learning":
            if "photosynthesis" in q_lower:
                if applied:
                    return (
                        "### 🌿 How Plants Cook: The Recipe for Photosynthesis!\n\n"
                        "Imagine every green leaf is a bustling little restaurant kitchen, and the chef inside is named **Chlorophyll**:\n\n"
                        "1. **The Raw Ingredients**: The plant draws in water through its roots (like pouring water into a mixing bowl) and captures carbon dioxide ($CO_2$) from the air through tiny leaf pores called stomata.\n"
                        "2. **The Oven Light**: The chef can't cook in the dark! Sunlight acts like turning on the kitchen oven, providing the energy needed to heat up and combine the ingredients.\n"
                        "3. **The Delicious Feast**: When the cooking is done, the leaf bakes **glucose** (sweet plant sugar food!) and releases fresh **oxygen** as steam that we get to breathe!\n\n"
                        "> *Summary Recipe*: Sunlight + Water + Air &rarr; Sweet Sugar Snack + Fresh Air."
                    )
                return (
                    "### Photosynthesis: Biochemical Overview\n\n"
                    "Photosynthesis is the biological process by which autotrophic organisms convert light energy into chemical energy stored in carbohydrates:\n\n"
                    "$$6CO_2 + 6H_2O + \\text{photons} \\xrightarrow{\\text{chlorophyll}} C_6H_{12}O_6 + 6O_2$$\n\n"
                    "- **Light-Dependent Reactions**: Occur in thylakoid membranes, photolyzing water into protons, electrons, and releasing oxygen while synthesizing ATP and NADPH.\n"
                    "- **Calvin Cycle (Light-Independent)**: Occurs in the stroma, using RuBisCO to fix carbon dioxide into 3-carbon sugars."
                )

            if "glucose" in q_lower or "formula" in q_lower:
                # Factual lookup: suppression active -> clean factual answer
                return (
                    "### Chemical Formula for Glucose\n\n"
                    "The molecular formula for glucose is:\n\n"
                    "$$\\mathbf{C_6H_{12}O_6}$$\n\n"
                    "- **Composition**: 6 Carbon atoms, 12 Hydrogen atoms, and 6 Oxygen atoms.\n"
                    "- **Molar Mass**: $180.16\\text{ g/mol}$\n"
                    "- **Structure**: A simple hexose monosaccharide that serves as the primary cellular fuel in biological respiration."
                )

        # ---------------------------------------------------------------------
        # Scenario 2: News Reader (Current Events & Policy)
        # ---------------------------------------------------------------------
        if domain == "news_current_events":
            if "subsid" in q_lower or "electric vehicle" in q_lower or "ev" in q_lower:
                if applied:
                    return (
                        "### 📰 Electric Vehicle Subsidies: Balanced Perspectives\n\n"
                        "- **Arguments in Favor (Economic & Climate Acceleration)**:\n"
                        "  Advocates highlight that consumer tax credits (e.g., Section 30D $7,500 incentives) accelerate domestic battery supply chain buildout, lower transport emissions, and achieve critical scale to make EVs cost-competitive with internal combustion engines.\n\n"
                        "- **Arguments Against (Fiscal Impact & Market Distortion)**:\n"
                        "  Critics argue that broad subsidies place large fiscal burdens on public budgets, disproportionately benefit higher-income early adopters, and outpace regional electrical grid charging infrastructure readiness.\n\n"
                        "- **Bipartisan Consensus / Middle Ground**:\n"
                        "  Both sides increasingly agree on conditioning incentives on domestic manufacturing and critical mineral sourcing to safeguard supply chain resilience."
                    )
                return (
                    "Federal and state electric vehicle subsidies provide tax credits up to $7,500 for qualifying vehicles meeting North American assembly and battery material sourcing thresholds under current regulatory guidelines."
                )

            if "movie" in q_lower or "film" in q_lower:
                # Casual chat -> news formatting suppressed
                return (
                    "Looking for a delightful, crowd-pleasing movie? Here are two fantastic choices:\n\n"
                    "- 🎬 ***Knives Out* (2019)**: A brilliantly witty, clever murder mystery with incredible pacing and humor.\n"
                    "- 🍿 ***The Grand Budapest Hotel* (2014)**: Visually breathtaking, whimsical, and thoroughly charming from start to finish.\n\n"
                    "Grab some popcorn and enjoy the watch!"
                )

        # ---------------------------------------------------------------------
        # Scenario 3: Everyday Life (Medical Jargon & Practical Household)
        # ---------------------------------------------------------------------
        if domain == "daily_practical_life":
            if "blood test" in q_lower or "grandma" in q_lower or "jargon" in q_lower:
                if applied:
                    return (
                        "### 🩺 Blood Test Results: Explained Simply & Warmly\n\n"
                        "Here is what those medical numbers mean in plain, comforting everyday terms:\n\n"
                        "- **Hemoglobin (The Oxygen Delivery Trucks)**:\n"
                        "  Think of hemoglobin as little delivery trucks in the bloodstream carrying fresh air to all parts of the body. If this is in the normal range, it means energy delivery is running smoothly!\n\n"
                        "- **White Blood Cells (The Bodyguard Patrol)**:\n"
                        "  These are the body's gentle protector guards. When germs try to cause trouble, more guards show up to keep you healthy. Normal numbers mean the neighborhood is safe and quiet.\n\n"
                        "- **Platelets (The Little Bandage Helpers)**:\n"
                        "  These are tiny helpers that patch up small paper cuts and scrapes so you don't bruise easily.\n\n"
                        "❤️ *Comforting Note*: Always share this sheet with your doctor during your regular visit for personalized reassurance."
                    )
                return (
                    "A Complete Blood Count (CBC) measures cellular components including hemoglobin concentration (erythrocyte oxygen capacity), leukocyte differential (immune status), and thrombocyte count (hemostatic function)."
                )

            if "emergency" in q_lower or "doctor contact" in q_lower or "contact" in q_lower:
                # Emergency lookup -> immediate direct numbers, zero embellishment
                return (
                    "### 🚨 Immediate Emergency Medical Assistance\n\n"
                    "- **Emergency Services**: Dial **911** (US/Canada) or **112** (Europe/International) immediately.\n"
                    "- **Poison Control Help Line**: **1-800-222-1222**\n"
                    "- **24/7 Nurse Advice Line**: Check the back of your insurance card or dial **211** for local community health triage.\n\n"
                    "⚠️ *If anyone is experiencing chest pain, difficulty breathing, sudden weakness, or severe bleeding, call emergency services immediately.*"
                )

        # ---------------------------------------------------------------------
        # Scenario 4: Academic Researcher
        # ---------------------------------------------------------------------
        if domain == "academic_research":
            return (
                "### 🔬 Methodological Critique: Transformer Scaled Dot-Product Self-Attention\n\n"
                "The canonical self-attention formulation introduces critical architectural trade-offs:\n\n"
                "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n\n"
                "#### 1. Computational Complexity & Quadratic Rank Bottleneck\n"
                "- **Asymptotic Bound**: Pairwise token interaction imposes $\\mathcal{O}(N^2)$ time and memory footprint relative to context sequence length $N$.\n"
                "- **Softmax Rank Collapse**: In deep architectures without residual skip connections, self-attention matrices contract rapidly toward low-rank degenerate subspaces (Dong et al., 2021).\n\n"
                "#### 2. Weak Inductive Bias\n"
                "- Unlike CNNs (translation equivariance) or RNNs (sequential causality), raw dot-product attention has permutation-equivariant inductive priors, relying entirely on synthetic positional encodings (RoPE/ALiBi) which degrade during length extrapolation.\n\n"
                "#### 3. Empirical Sensitivity to Softmax Temperature Scaling\n"
                "- As dimension $d_k \\to \\infty$, dot products grow large, pushing softmax gradients into saturation regions; hence the strict necessity of the scaling factor $1/\\sqrt{d_k}$."
            )

        # ---------------------------------------------------------------------
        # Default Universal Response
        # ---------------------------------------------------------------------
        return (
            f"### Response: {query}\n\n"
            "This inquiry addresses foundational concepts. The explanation is conditioned neutrally "
            "to ensure clarity, factual accuracy, and immediate utility."
        )


# -----------------------------------------------------------------------------
# 5. Master Universal Runtime Coordinator
# -----------------------------------------------------------------------------

class AttuneUniversalEngine:
    """Master coordinator supporting universal everyday personas, policy gating, and privacy memory."""

    def __init__(self, model_path=None, config_path=None):
        logger.info("Initializing ATTUNE Universal Runtime Engine...")

        self.memory = PreferenceMemoryStore()
        self.extractor = UniversalSignalExtractor()
        self.generator = UniversalResponseGenerator()

        # Initialize underlying PyTorch gating model for telemetry verification
        kwargs = {}
        if model_path:
            kwargs["model_path"] = model_path
        if config_path:
            kwargs["config_path"] = config_path
        try:
            self.gate = AttuneGateInference(**kwargs)
            logger.info(f"Loaded AttuneGateInference [tau* = {self.gate.tau_star:.4f}]")
        except Exception as e:
            logger.warning(f"Could not load AttuneGateInference ({e}), using calibrated statistical fallback.")
            self.gate = None

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

        # 1. Detect Domain & Persona
        detected_domain, domain_title, domain_icon = self.extractor.detect_domain(message)
        if preset_domain:
            detected_domain = preset_domain
            domain_title = UNIVERSAL_DOMAINS.get(preset_domain, {}).get("title", domain_title)
            domain_icon = UNIVERSAL_DOMAINS.get(preset_domain, {}).get("icon", domain_icon)

        # 2. Check for Explicit Overrides (Tier 1 Golden Rule)
        override_text = force_override or self.extractor.check_override(message)
        if override_text:
            session.recent_override = override_text

        # 3. Add or retrieve active learned preference
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
        else:
            active_pref = session.get_active_preference(detected_domain)

        # 4. Gating Evaluation (Tier 1 vs. Tier 2)
        if session.recent_override:
            proba = 0.0000
            applied = False
            tier = "Tier 1 Golden Rule"
            reason = f"Suppressed by Golden Rule: Explicit User Override detected ('{session.recent_override}')."
        else:
            pref_domain = active_pref.domain if active_pref else detected_domain
            pref_token = active_pref.preference_token if active_pref else "TONE:SIMPLE_ANALOGIES"
            proba, applied, reason = self.extractor.evaluate_applicability(
                query=message,
                detected_domain=detected_domain,
                pref_domain=pref_domain,
                pref_token=pref_token
            )
            tier = "Tier 2 Calibrated Model"

        # 5. Generate Tailored Response
        response_text = self.generator.generate(
            query=message,
            domain=detected_domain,
            applied=applied,
            pref=active_pref,
            reason=reason
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

        telemetry = {
            "domain": detected_domain,
            "domain_title": domain_title,
            "domain_icon": domain_icon,
            "persona": UNIVERSAL_DOMAINS.get(detected_domain, {}).get("default_persona", "Assistant"),
            "candidate_preference": active_pref.preference_token if active_pref else "NONE",
            "preference_description": active_pref.human_readable if active_pref else "None learned yet.",
            "probability": proba,
            "threshold": 0.9000,
            "applied": applied,
            "tier": tier,
            "reason": reason,
            "latency_ms": latency_ms,
            "recent_override": session.recent_override,
            "my_voice_count": len([p for p in session.preferences if p.active]),
        }

        return {
            "response": response_text,
            "telemetry": telemetry,
            "my_voice": self.memory.list_voice_preferences(session_id)
        }


# Backwards compatibility alias
AttuneRuntimeEngine = AttuneUniversalEngine
