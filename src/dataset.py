"""
ATTUNE Gating Dataset Implementation
====================================
PyTorch Dataset that loads JSONL decision tuples and formats each record
into the canonical structured template defined in docs/spec.md:

    CURRENT REQUEST: {current_query}
    CANDIDATE PREFERENCE: {candidate_preference}
    PREFERENCE SCOPE: {preference_scope}
    SUPPORTING EVIDENCE: {supporting_evidence}
    RECENT OVERRIDE: {recent_override}

Tokenizes with HuggingFace AutoTokenizer (sentence-transformers/all-MiniLM-L6-v2)
with max_length=192 and returns tensors ready for training and evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger("attune_dataset")

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_LENGTH = 192


def serialize_record(record: Dict[str, Any]) -> str:
    """Serializes a decision tuple into the canonical prompt format."""
    current_query = (record.get("current_query") or "").strip()
    candidate_pref = (record.get("candidate_preference") or "").strip()
    pref_scope = (record.get("preference_scope") or "").strip()

    evidence = record.get("supporting_evidence", [])
    if isinstance(evidence, list):
        evidence_str = " | ".join(evidence) if evidence else "NONE"
    else:
        evidence_str = str(evidence).strip() if evidence else "NONE"

    override = record.get("recent_override")
    override_str = str(override).strip() if override else "NONE"

    return (
        f"CURRENT REQUEST: {current_query}\n"
        f"CANDIDATE PREFERENCE: {candidate_pref}\n"
        f"PREFERENCE SCOPE: {pref_scope}\n"
        f"SUPPORTING EVIDENCE: {evidence_str}\n"
        f"RECENT OVERRIDE: {override_str}"
    )


class AttuneDataset(Dataset):
    """PyTorch Dataset for ATTUNE gating tuples."""

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        else:
            self.tokenizer = tokenizer

        self.records = self._load_data(self.data_path)
        logger.info(f"Loaded {len(self.records)} records from {self.data_path}")

    def _load_data(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        text = serialize_record(record)
        label = float(record.get("gold_applicability", 0))

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }
