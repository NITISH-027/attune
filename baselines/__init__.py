"""ATTUNE Baselines Package"""
from .zero_shot import ZeroShotBaseline
from .fixed_profile import FixedProfileBaseline
from .history_rag import HistoryRAGBaseline
from .rule_based import RuleBasedBaseline

ZeroShotBase = ZeroShotBaseline
FixedProfileBase = FixedProfileBaseline
HistoryRAGBase = HistoryRAGBaseline
RuleBasedGate = RuleBasedBaseline

__all__ = [
    "ZeroShotBaseline",
    "ZeroShotBase",
    "FixedProfileBaseline",
    "FixedProfileBase",
    "HistoryRAGBaseline",
    "HistoryRAGBase",
    "RuleBasedBaseline",
    "RuleBasedGate",
]
