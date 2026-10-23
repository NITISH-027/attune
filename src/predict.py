"""
ATTUNE Production Inference Engine
==================================
Implements the 2-Tier Decision Architecture specified in docs/spec.md Section 4:
- Tier 1: The Golden Priority Rule (Query-Level / Explicit Override Pre-Filter).
  If `recent_override` is present, immediately suppresses preference (decision = 0).
- Tier 2: Calibrated Statistical Preference Gate.
  Evaluates serialized context tuple, applies temperature-calibrated probability scaling,
  and gates against the empirically validated decision threshold tau*.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
from transformers import AutoTokenizer

from src.dataset import serialize_record, DEFAULT_MODEL_NAME, DEFAULT_MAX_LENGTH
from src.model import AttuneGatingModel

logger = logging.getLogger("attune_predict")

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "attune_gate_best.pt"
DEFAULT_CONFIG_PATH = ROOT_DIR / "models" / "config.json"


class AttuneGateInference:
    """Production runtime gate for ATTUNE."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
        device: str = "cpu",
    ):
        self.device = device
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        # Load config
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.tau_star = float(self.config.get("tau_star", 0.50))
        self.temperature = float(self.config.get("temperature", 1.0))
        self.max_length = int(self.config.get("max_length", DEFAULT_MAX_LENGTH))
        self.backbone_name = str(self.config.get("backbone_model", DEFAULT_MODEL_NAME))

        logger.info(f"Loading tokenizer '{self.backbone_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)

        logger.info(f"Loading AttuneGatingModel from {self.model_path}...")
        self.model = AttuneGatingModel(
            pretrained_model_name=self.backbone_name,
            initial_temperature=self.temperature,
        ).to(self.device)

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        logger.info(
            f"AttuneGateInference initialized successfully. "
            f"[tau* = {self.tau_star:.4f}, T = {self.temperature:.4f}]"
        )

    def predict(
        self,
        current_query: str,
        candidate_preference: str,
        preference_scope: str,
        supporting_evidence: Optional[Union[List[str], str]] = None,
        recent_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes gating evaluation on an incoming interaction tuple.

        Returns:
            Dict containing:
                decision: 1 (Apply) or 0 (Suppress)
                probability: float in [0.0, 1.0]
                threshold: float (tau*)
                applied: bool (True if decision == 1)
                tier: str ("Tier 1 Pre-Filter" or "Tier 2 Gating Model")
                reason: Optional explanation string
        """
        # Normalize evidence input
        if supporting_evidence is None:
            supporting_evidence = []
        elif isinstance(supporting_evidence, str):
            supporting_evidence = [supporting_evidence]

        # Model Execution
        record = {
            "current_query": current_query,
            "candidate_preference": candidate_preference,
            "preference_scope": preference_scope,
            "supporting_evidence": supporting_evidence,
            "recent_override": recent_override,
        }

        serialized_text = serialize_record(record)
        encoding = self.tokenizer(
            serialized_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            proba = float(self.model.predict_proba(input_ids, attention_mask).item())

        # Tier 1 Golden Rule: Explicit override suppresses the application decision
        has_override = recent_override is not None and str(recent_override).strip()
        if has_override:
            decision = 0
            applied = False
            tier = "Tier 1 Golden Rule"
            reason = f"Suppressed by Golden Rule: Explicit User Override detected ('{recent_override}')."
        else:
            decision = 1 if proba >= self.tau_star else 0
            applied = (decision == 1)
            tier = "Tier 2 Calibrated Model"
            reason = "Contextually Applicable" if decision == 1 else "Context Shift / Insufficient Evidence"

        return {
            "decision": decision,
            "probability": proba,
            "threshold": self.tau_star,
            "applied": applied,
            "tier": tier,
            "reason": reason,
        }
