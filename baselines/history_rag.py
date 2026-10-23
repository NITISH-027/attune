"""
System C: History Retrieval (RAG) Baseline
==========================================
Operational Policy:
Simulates dense vector retrieval over past interaction turns.
Computes cosine similarity between `current_query` and `supporting_evidence`
using a lightweight sentence embedding model (`sentence-transformers/all-MiniLM-L6-v2`).
If max cosine similarity >= threshold (default 0.45), predicts applicability = 1; else 0.

This baseline captures semantic topical proximity, but illustrates why pure vector retrieval
fails under context shifts (keyword overlap across different technical tasks)
and explicit reversals (retrieves the historical preference even after it has been superseded).
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger("history_rag")


class HistoryRAGBaseline:
    """Baseline C: Dense vector retrieval baseline using SentenceTransformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.45,
        name: str = "System C (History Retrieval - RAG)",
    ):
        self.name = name
        self.threshold = threshold
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy loader for sentence transformer model."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def compute_similarity(self, query: str, evidence_list: List[str]) -> float:
        """Computes maximum cosine similarity between query and any evidence turn."""
        if not evidence_list:
            return 0.0

        query_emb = self.model.encode(query, normalize_embeddings=True)
        evidence_embs = self.model.encode(evidence_list, normalize_embeddings=True)

        # Cosine similarity for normalized vectors is dot product
        similarities = np.dot(evidence_embs, query_emb)
        return float(np.max(similarities))

    def predict(self, record: Dict[str, Any]) -> int:
        """Predicts 1 if max similarity >= threshold, else 0."""
        query = record.get("current_query", "")
        evidence = record.get("supporting_evidence", [])
        if not query or not evidence:
            return 0

        max_sim = self.compute_similarity(query, evidence)
        return 1 if max_sim >= self.threshold else 0

    def predict_batch(self, records: List[Dict[str, Any]], batch_size: int = 64) -> List[int]:
        """Vectorized batch prediction for optimal evaluation performance."""
        if not records:
            return []

        queries = [r.get("current_query", "") for r in records]
        # Flatten evidence with index mapping
        evidence_texts = []
        evidence_map = []  # maps evidence index to record index

        for rec_idx, r in enumerate(records):
            ev_list = r.get("supporting_evidence", [])
            for ev in ev_list:
                evidence_texts.append(ev)
                evidence_map.append(rec_idx)

        # Batch encode queries
        query_embs = self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        preds = [0] * len(records)
        if not evidence_texts:
            return preds

        # Batch encode all evidence
        evidence_embs = self.model.encode(
            evidence_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Group evidence embeddings per record and compute max similarity
        record_evidence_embs = [[] for _ in range(len(records))]
        for ev_idx, rec_idx in enumerate(evidence_map):
            record_evidence_embs[rec_idx].append(evidence_embs[ev_idx])

        for rec_idx in range(len(records)):
            ev_embs = record_evidence_embs[rec_idx]
            if not ev_embs:
                preds[rec_idx] = 0
            else:
                ev_embs_arr = np.array(ev_embs)
                sims = np.dot(ev_embs_arr, query_embs[rec_idx])
                max_sim = float(np.max(sims))
                preds[rec_idx] = 1 if max_sim >= self.threshold else 0

        return preds


HistoryRAGBase = HistoryRAGBaseline
