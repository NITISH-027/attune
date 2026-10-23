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
    preset_domain: Optional[str] = None
    preset_pref: Optional[str] = None
    preset_dimension: Optional[str] = None
    preset_human_readable: Optional[str] = None
    force_override: Optional[str] = None


class ResetRequest(BaseModel):
    session_id: Optional[str] = "default_session"


class DeleteMemoryRequest(BaseModel):
    session_id: str
    pref_id: str


class UpdateMemoryRequest(BaseModel):
    session_id: str
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
        result = engine.process_turn(
            session_id=req.session_id or "default_session",
            message=req.message.strip(),
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
    status_data = {
        "model_architecture": "AttuneUniversalEngine (all-MiniLM-L6-v2 + 2-Tier Gate)",
        "backbone_model": "sentence-transformers/all-MiniLM-L6-v2",
        "temperature": 0.8453,
        "tau_star": 0.9000,
        "test_accuracy": 0.985,
        "negative_transfer_rate": 0.000,
        "false_suppression_rate": 0.0375,
        "repeated_correction_rate": 0.000,
        "mean_latency_ms": 24.73,
        "status": "online",
        "universal_domains": [
            "education_learning",
            "news_current_events",
            "academic_research",
            "daily_practical_life",
            "technical_coding",
        ]
    }

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                status_data["temperature"] = cfg.get("temperature", status_data["temperature"])
                status_data["tau_star"] = cfg.get("tau_star", status_data["tau_star"])
        except Exception as e:
            logger.warning(f"Could not read config.json: {e}")

    if SCORECARD_PATH.exists():
        try:
            with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
                sc = json.load(f)
                for sys_info in sc.get("comparative_systems", []):
                    if "ATTUNE" in sys_info.get("system_name", ""):
                        status_data["test_accuracy"] = sys_info.get("accuracy", 0.985)
                        status_data["negative_transfer_rate"] = sys_info.get("ntr", 0.0)
                        status_data["false_suppression_rate"] = sys_info.get("fsr", 0.0375)
                        status_data["mean_latency_ms"] = sys_info.get("latency_ms_per_query", 24.73)
        except Exception as e:
            logger.warning(f"Could not read final_scorecard.json: {e}")

    return status_data


@app.get("/api/memory")
def get_memory_endpoint(session_id: str = "default_session"):
    """Returns the list of active learned preferences for 'My Voice'."""
    return {
        "session_id": session_id,
        "preferences": engine.memory.list_voice_preferences(session_id)
    }


@app.post("/api/memory/delete")
def delete_memory_endpoint(req: DeleteMemoryRequest):
    """Deletes or forgets a specific preference to give users complete control."""
    success = engine.memory.delete_preference(req.session_id, req.pref_id)
    if not success:
        raise HTTPException(status_code=404, detail="Preference ID not found.")
    return {
        "status": "success",
        "message": "Preference removed from My Voice memory.",
        "preferences": engine.memory.list_voice_preferences(req.session_id)
    }


@app.post("/api/memory/update")
def update_memory_endpoint(req: UpdateMemoryRequest):
    """Updates a learned preference description in 'My Voice'."""
    success = engine.memory.update_preference(req.session_id, req.pref_id, req.new_text)
    if not success:
        raise HTTPException(status_code=404, detail="Preference ID not found.")
    return {
        "status": "success",
        "message": "Preference updated successfully.",
        "preferences": engine.memory.list_voice_preferences(req.session_id)
    }


@app.post("/api/reset")
def reset_endpoint(req: ResetRequest):
    """Resets the memory state for a given session."""
    session_id = req.session_id or "default_session"
    engine.memory.reset_session(session_id)
    return {
        "status": "success",
        "message": f"Session '{session_id}' state reset successfully.",
        "session_id": session_id,
        "preferences": engine.memory.list_voice_preferences(session_id)
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
