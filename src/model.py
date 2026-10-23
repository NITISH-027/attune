"""
ATTUNE Gating Model Architecture
================================
Lightweight transformer-based preference gating model:
- Backbone: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional embeddings)
- Mean Pooling with attention mask
- Classification Head: Linear(384, 128) -> ReLU() -> Dropout(0.2) -> Linear(128, 1)
- Calibratable scalar temperature parameter for post-hoc probability scaling.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger("attune_model")

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class AttuneGatingModel(nn.Module):
    """Context-aware preference gating classifier."""

    def __init__(
        self,
        pretrained_model_name: str = DEFAULT_MODEL_NAME,
        dropout_rate: float = 0.2,
        initial_temperature: float = 1.0,
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size  # 384 for MiniLM

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

        # Calibratable temperature parameter (scalar)
        self.temperature = nn.Parameter(
            torch.tensor(initial_temperature, dtype=torch.float32),
            requires_grad=False,
        )

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Computes attention-weighted mean pooling across token representations."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing raw uncalibrated logits.
        Returns:
            torch.Tensor of shape (batch_size,)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Inference method returning temperature-calibrated applicability probabilities.
        Returns:
            torch.Tensor of shape (batch_size,) with values in [0, 1]
        """
        logits = self.forward(input_ids, attention_mask)
        temp = torch.clamp(self.temperature, min=1e-3)
        return torch.sigmoid(logits / temp)
