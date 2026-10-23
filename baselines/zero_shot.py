"""
System A: Zero-Shot Base Model Baseline
=======================================
Operational Policy:
Assumes zero personalization memory. Always predicts applicability = 0.
Under this baseline, the assistant never applies historical preferences,
ensuring zero Negative Transfer (NTR -> 0.0) at the cost of total personalization loss (FSR -> 1.0).
"""

from typing import Any, Dict, List


class ZeroShotBaseline:
    """Baseline A: Always predicts 0 (never applies preference)."""

    def __init__(self, name: str = "System A (Zero-Shot Base)"):
        self.name = name

    def predict(self, record: Dict[str, Any]) -> int:
        """Returns 0 regardless of evidence or query."""
        return 0

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[int]:
        """Returns list of 0s for all records."""
        return [0] * len(records)


ZeroShotBase = ZeroShotBaseline
