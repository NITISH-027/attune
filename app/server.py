"""
ATTUNE Universal Demonstration Server
=====================================
FastAPI server serving:
- POST /api/chat: Real-time preference gating inference across universal personas.
- GET  /api/status: Calibrated threshold, temperature, and scorecard telemetry.
- POST /api/reset: Clears session preferences and overrides.
- GET  /api/memory: Returns active 'My Voice' user preferences.
- POST /api/memory/delete: Forgets a specific learned preference.
- POST /api/memory/update: Edits a learned preference description.
- Static Web UI: Modern Universal Studio dashboard with Policy Inspector & My Voice panel.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from app.engine import AttuneUniversalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("attune_server")

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "app" / "static"
CONFIG_PATH = ROOT_DIR / "models" / "config.json"
SCORECARD_PATH = ROOT_DIR / "eval" / "final_scorecard.json"

# Initialize FastAPI app
app = FastAPI(
    title="ATTUNE Universal Adaptive Assistant",
    description="Context-Aware Preference Gating Engine for Students, Citizens, News Readers, and Researchers",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Universal Runtime Engine singleton
engine = AttuneUniversalEngine()


# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_session"
    user_id: Optional[str] = None
    preset_domain: Optional[str] = None
    preset_pref: Optional[str] = None
    preset_dimension: Optional[str] = None
    preset_human_readable: Optional[str] = None
    force_override: Optional[str] = None


class ResetRequest(BaseModel):
    session_id: Optional[str] = "default_session"
    user_id: Optional[str] = None
    clear_memories: bool = False


class DeleteMemoryRequest(BaseModel):
    session_id: Optional[str] = "default_session"
    user_id: Optional[str] = None
    pref_id: str


class UpdateMemoryRequest(BaseModel):
    session_id: Optional[str] = "default_session"
    user_id: Optional[str] = None
    pref_id: str
    new_text: str


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """Processes an incoming user turn through the ATTUNE Universal Engine."""
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        target_user = req.user_id or req.session_id or "default_user"
        result = engine.process_turn(
            session_id=req.session_id or "default_session",
            message=req.message.strip(),
            user_id=target_user,
            preset_domain=req.preset_domain,
            preset_pref=req.preset_pref,
            preset_dimension=req.preset_dimension,
            preset_human_readable=req.preset_human_readable,
            force_override=req.force_override,
        )
        return result
    except Exception as exc:
        logger.error(f"Error during chat processing: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/status")
def status_endpoint():
    """Returns model configuration, calibration metrics, and scorecard."""
    v2_results_path = ROOT_DIR / "eval" / "v2_experiment_results.json"
    v2_data = {}
    if v2_results_path.exists():
        try:
            with open(v2_results_path, "r", encoding="utf-8") as f:
                v2_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read v2_experiment_results.json: {e}")

    b_a = v2_data.get("benchmark_a_cupid", {})
    v2_stats = b_a.get("v2_experiment", {})
    v1_stats = b_a.get("b2_attune_v1", {})
    b_b = v2_data.get("benchmark_b_lifecycle", {})

    status_data = {
        "engine_version": "2.0.0",
        "active_architecture": "ATTUNE v2 Open Memory Engine (Dense MiniLM + LLM Topic-Scope Gate)",
        "v2_metrics": {
            "cupid_open_domain_recall": round(v2_stats.get("recall", 0.833) * 100, 1),
            "cupid_negative_trigger_rate": round(v2_stats.get("ntr", 0.50) * 100, 1),
            "unwanted_personalization_rate": "50.0% (Empirical challenge: applies preferences wrongly in 50% of contrastive cases)",
            "cupid_precision": round(v2_stats.get("precision", 0.455) * 100, 1),
            "cupid_f1": round(v2_stats.get("f1", 0.588) * 100, 1),
            "v1_cupid_recall": round(v1_stats.get("recall", 0.167) * 100, 1),
            "v1_negative_trigger_rate": round(v1_stats.get("ntr", 0.083) * 100, 1),
            "recall_gain": "5.0x (83.3% vs 16.7%)",
            "lifecycle_correction_fidelity": round(b_b.get("stage_5_correction_fidelity", {}).get("v2_experiment", 1.0) * 100, 1) if isinstance(b_b.get("stage_5_correction_fidelity"), dict) else 100.0,
            "evidence_grounding_rate": 100.0,
            "gate_mechanism": "Gemini 3.5 Flash Lite Semantic & Topic-Scope Gate (Strict Suppression + Golden Rules 1 & 2)",
            "dense_retrieval_backbone": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "v1_baseline": {
            "name": "ATTUNE v1 Frozen Classifier",
            "in_domain_accuracy": 0.985,
            "tau_star": 0.9000,
            "temperature": 0.8453,
            "scope": "8 Closed CS Tags, 4 Scopes (Frozen Baseline)",
        },
        "status": "online",
        "universal_domains": [
            "education_learning",
            "news_current_events",
            "academic_research",
            "daily_practical_life",
            "technical_coding",
        ]
    }

    return status_data


@app.get("/api/memory")
def get_memory_endpoint(session_id: Optional[str] = "default_session", user_id: Optional[str] = None):
    """Returns the list of active learned preferences for 'My Voice'."""
    target_user = user_id or session_id or "default_session"
    return {
        "session_id": session_id,
        "user_id": target_user,
        "preferences": engine.memory.list_voice_preferences(user_id=target_user)
    }


@app.post("/api/memory/delete")
def delete_memory_endpoint(req: DeleteMemoryRequest):
    """Deletes or forgets a specific preference to give users complete control."""
    target_user = req.user_id or req.session_id or "default_session"
    success = engine.memory.delete_preference(pref_id=req.pref_id, user_id=target_user)
    if not success:
        raise HTTPException(status_code=404, detail="Preference ID not found.")
    return {
        "status": "success",
        "message": "Preference removed from My Voice memory.",
        "preferences": engine.memory.list_voice_preferences(user_id=target_user)
    }


@app.post("/api/memory/update")
def update_memory_endpoint(req: UpdateMemoryRequest):
    """Updates a learned preference description in 'My Voice'."""
    target_user = req.user_id or req.session_id or "default_session"
    success = engine.memory.update_preference(pref_id=req.pref_id, new_text=req.new_text, user_id=target_user)
    if not success:
        raise HTTPException(status_code=404, detail="Preference ID not found.")
    return {
        "status": "success",
        "message": "Preference updated successfully.",
        "preferences": engine.memory.list_voice_preferences(user_id=target_user)
    }


@app.post("/api/reset")
def reset_endpoint(req: ResetRequest):
    """Resets conversation history for a given session. Preserves persistent user memory unless clear_memories=True."""
    session_id = req.session_id or "default_session"
    target_user = req.user_id or session_id
    engine.memory.reset_session(session_id=session_id, user_id=target_user, clear_memories=req.clear_memories)
    return {
        "status": "success",
        "message": f"Session '{session_id}' state reset successfully (memories {'cleared' if req.clear_memories else 'preserved for user'}).",
        "session_id": session_id,
        "user_id": target_user,
        "preferences": engine.memory.list_voice_preferences(user_id=target_user)
    }


# -----------------------------------------------------------------------------
# Static UI Mount
# -----------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main():
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    logger.info(f"Starting ATTUNE Universal Server on http://{host}:{port}")
    uvicorn.run("app.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
