"""Typed inputs and outputs shared by aggregation implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

StateDict = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class ClientUpdate:
    """Represent one client's model state and public aggregation weight. A Client refers
    to a single Access Point.

    Args:
        client_id: Unique client identifier. Access Point index.
        state: Model state submitted by the client.
        weight: Coefficient used by weighted aggregation.
    """

    client_id: int
    state: StateDict
    weight: float | int = 1


@dataclass(frozen=True)
class LinearAggregationPlan:
    """Describes arithmetic for single aggregation round.

    Args:
        coefficients: Coefficient corresponding to each client update.
        normalizer: Positive value applied to the aggregate numerator.
    """

    coefficients: tuple[float, ...]
    normalizer: float


@dataclass(frozen=True)
class AggregationResult:
    """Store aggregated result and its metadata.

    Args:
        state: Aggregated model state.
        client_ids: Client identifiers in aggregation order.
        coefficients: Linear coefficient for each client.
        normalizer: Aggregate normalization value.
        strategy_name: Name of the strategy that produced the result.
    """

    state: dict[str, torch.Tensor]
    client_ids: tuple[int, ...]
    coefficients: tuple[float, ...]
    normalizer: float
    strategy_name: str
