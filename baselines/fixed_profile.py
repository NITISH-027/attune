"""
System B: Fixed Profile Baseline
================================
Operational Policy:
Simulates static custom instructions or persistent user profiles.
Always assumes the candidate preference applies universally (predicts applicability = 1),
ignoring task boundaries, domain shifts, and corrective overrides.
Under this baseline, personalization is maximally preserved (FSR -> 0.0),
but Negative Transfer and Repeated-Correction errors are pathological (NTR -> high, RCR -> 1.0).
"""

from typing import Any, Dict, List


class FixedProfileBaseline:
    """Baseline B: Always predicts 1 (always applies preference)."""

    def __init__(self, name: str = "System B (Fixed Profile)"):
        self.name = name

    def predict(self, record: Dict[str, Any]) -> int:
        """Returns 1 whenever a candidate preference is present."""
        return 1

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[int]:
        """Returns list of 1s for all records."""
        return [1] * len(records)


FixedProfileBase = FixedProfileBaseline
