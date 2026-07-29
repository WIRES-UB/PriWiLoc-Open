"""Deliberate public API for secure aggregation."""

from __future__ import annotations

from models.secure_agg.aggregator import SecureAggregator
from models.secure_agg.trust_tracking import (
    APTrustEvent,
    APTrustTracker,
    ServerWeightChangeEvent,
    ServerWeightChangeTracker,
)
from models.secure_agg.types import (
    AggregationReport,
    AttackRecord,
    Report,
    ServerResponse,
)

__all__ = [
    "AggregationReport",
    "APTrustEvent",
    "APTrustTracker",
    "AttackRecord",
    "Report",
    "SecureAggregator",
    "ServerResponse",
    "ServerWeightChangeEvent",
    "ServerWeightChangeTracker",
]
