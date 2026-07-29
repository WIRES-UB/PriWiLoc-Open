"""Immutable records exchanged by aggregation phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from models.aggregation import AggregationResult


@dataclass(frozen=True)
class AttackContext:
    """Provide deterministic context for one rule applied to one AP response.

    Args:
        round: Secure aggregation round index.
        epoch: Training epoch index.
        step: Global training step.
        ap_id: Target access-point identifier.
        rule_order: Attack rule position in configuration order.
        master_seed: Root deterministic seed.
    """

    round: int
    epoch: int
    step: int
    ap_id: int
    rule_order: int
    master_seed: int


@dataclass(frozen=True)
class AttackRecord:
    """Describe the auditable effects of one attack rule.

    Args:
        rule_name: Configured attack rule name.
        attack_type: Concrete attack implementation name.
        order: Rule position in configuration order.
        ap_ids: Targeted access-point identifiers.
        tensor_keys: State tensors selected by the attack.
        affected_element_count: Number of selected tensor elements.
    """

    rule_name: str
    attack_type: str
    order: int
    ap_ids: tuple[int, ...]
    tensor_keys: tuple[str, ...]
    affected_element_count: int


@dataclass(frozen=True)
class ServerResponse:
    """Represent one AP-specific response and its integrity decision.

    Args:
        ap_id: Recipient access-point identifier.
        state: Model state carried by the response.
        attacked: Whether the response changed effectively.
        verified: Whether integrity verification succeeded.
        attack_records: Attack rules applied to the response.
    """

    ap_id: int
    state: Mapping[str, torch.Tensor]
    attacked: bool
    verified: bool
    attack_records: tuple[AttackRecord, ...] = ()


@dataclass(frozen=True)
class AggregationReport:
    """Store complete metadata and per-AP decisions for one aggregation round.

    Args:
        round: Aggregation round index.
        epoch: Training epoch index.
        step: Global training step.
        accepted: Whether every AP response verified.
        server_honest: Whether no attack rule executed.
        ap_trust: Per-AP verification decisions.
        changed_ap_ids: AP responses changed effectively.
        attack_records: Applied attack descriptions.
        aggregation: Canonical aggregation result.
        recover_max_abs_err: Maximum fixed-point recovery error.
        quant_rel_err: Relative quantization error.
        rejections_cumulative: Rejected-round count through this round.
        round_seconds: Wall-clock duration of the round.
    """

    round: int
    epoch: int
    step: int
    accepted: bool
    server_honest: bool
    ap_trust: tuple[bool, ...]
    changed_ap_ids: tuple[int, ...]
    attack_records: tuple[AttackRecord, ...]
    aggregation: AggregationResult
    recover_max_abs_err: float
    quant_rel_err: float
    rejections_cumulative: int
    round_seconds: float


Report = AggregationReport
