"""Aggregation strategies."""

from __future__ import annotations

import math
import numbers
from abc import ABC, abstractmethod
from collections.abc import Sequence

from models.aggregation.state_dict_utils import (
    REFERENCE_CLIENT_POLICY,
    aggregate_linear_state,
    validate_client_updates,
)
from models.aggregation.types import (
    AggregationResult,
    ClientUpdate,
    LinearAggregationPlan,
)


class AggregationStrategy(ABC):
    """Define the interface for model-state aggregation strategies. Aggregation
    strategies must be linear.

    Args:
        buffer_policy: Policy used for excluded model buffers.
        reference_client_id: Client supplying excluded buffer values.
    """

    secure_compatible = True

    def __init__(
        self,
        *,
        buffer_policy: str = REFERENCE_CLIENT_POLICY,
        reference_client_id: int = 0,
    ) -> None:
        """Initialize shared excluded-buffer behavior.

        Args:
            buffer_policy: Policy used for excluded model buffers.
            reference_client_id: Client supplying excluded buffer values.

        Returns:
            None.
        """
        if buffer_policy != REFERENCE_CLIENT_POLICY:
            raise ValueError(
                "Only buffer_policy='reference_client' is supported."
            )
        self.buffer_policy = buffer_policy
        self.reference_client_id = int(reference_client_id)

    @property
    def name(self) -> str:
        """Return a stable strategy name for reports and checkpoints.

        Returns:
            The concrete strategy class name.
        """

        return type(self).__name__

    def linear_plan(
        self,
        updates: Sequence[ClientUpdate],
    ) -> LinearAggregationPlan:
        """Create a validated plan for plain aggregation.

        Args:
            updates: Client updates participating in the round.

        Returns:
            Linear coefficients and their normalizer.
        """

        validate_client_updates(updates)
        return self._linear_plan(updates)

    @abstractmethod
    def _linear_plan(
        self,
        updates: Sequence[ClientUpdate],
    ) -> LinearAggregationPlan:
        """Build a plain linear plan for already validated updates.

        Args:
            updates: Validated client updates.

        Returns:
            Strategy-specific coefficients and normalizer.
        """

    def secure_plan(
        self,
        updates: Sequence[ClientUpdate],
    ) -> LinearAggregationPlan:
        """Create an integer plan supported by secure aggregation.

        Args:
            updates: Client updates participating in the round.

        Returns:
            Secure-compatible coefficients and normalizer.
        """

        validate_client_updates(updates)
        return self._secure_plan(updates)

    def _secure_plan(
        self,
        updates: Sequence[ClientUpdate],
    ) -> LinearAggregationPlan:
        """Validate that the plain plan has a secure integer representation.

        Args:
            updates: Validated client updates.

        Returns:
            Secure-compatible coefficients and normalizer.
        """

        plan = self._linear_plan(updates)
        if any(
            coefficient < 0 or not float(coefficient).is_integer()
            for coefficient in plan.coefficients
        ) or not float(plan.normalizer).is_integer():
            raise ValueError(
                f"{self.name} cannot be represented securely with public "
                "non-negative integer coefficients."
            )
        return plan

    def prepare(
        self,
        updates: Sequence[ClientUpdate],
        *,
        secure: bool = False,
    ) -> AggregationResult:
        """Validate, plan, and aggregate client states in one pass.

        Args:
            updates: Client updates participating in the round.
            secure: Whether secure-compatible planning is required.

        Returns:
            The aggregate state and reproducibility metadata.
        """

        keys = validate_client_updates(updates)
        plan = (
            self._secure_plan(updates)
            if secure
            else self._linear_plan(updates)
        )
        state = aggregate_linear_state(
            updates,
            coefficients=plan.coefficients,
            normalizer=plan.normalizer,
            reference_client_id=self.reference_client_id,
            validated_keys=keys,
        )
        return AggregationResult(
            state=state,
            client_ids=tuple(update.client_id for update in updates),
            coefficients=plan.coefficients,
            normalizer=plan.normalizer,
            strategy_name=self.name,
        )

    def aggregate(
        self,
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        """Aggregate client states without mutating input tensors.

        Args:
            updates: Client updates participating in the round.

        Returns:
            The aggregate state and reproducibility metadata.
        """

        return self.prepare(updates)


class SumAggregation(AggregationStrategy):
    """Compute the unnormalized elementwise client sum."""

    def _linear_plan(
        self,
        updates: Sequence[ClientUpdate],
    ) -> LinearAggregationPlan:
        """Create a unit-coefficient sum plan.

        Args:
            updates: Validated client updates.

        Returns:
            A plan with unit coefficients and unit normalizer.
        """
        return LinearAggregationPlan(
            coefficients=(1.0,) * len(updates),
            normalizer=1.0,
        )


class UniformMeanAggregation(AggregationStrategy):
    """Compute an equal-weight elementwise client mean."""

    def _linear_plan(
        self,
        updates: Sequence[ClientUpdate],
    ) -> LinearAggregationPlan:
        """Create an equal-weight mean plan.

        Args:
            updates: Validated client updates.

        Returns:
            A plan normalized by the number of clients.
        """
        return LinearAggregationPlan(
            coefficients=(1.0,) * len(updates),
            normalizer=float(len(updates)),
        )


class WeightedMeanAggregation(AggregationStrategy):
    """Compute a public weighted mean, typically from sample counts.

    Args:
        weight_source: Semantic source of client weights.
        client_weights: Optional fixed public weights.
        buffer_policy: Policy used for excluded model buffers.
        reference_client_id: Client supplying excluded buffer values.
    """

    def __init__(
        self,
        *,
        weight_source: str = "sample_count",
        client_weights: Sequence[float | int] | None = None,
        buffer_policy: str = REFERENCE_CLIENT_POLICY,
        reference_client_id: int = 0,
    ) -> None:
        """Initialize weighted aggregation settings.

        Args:
            weight_source: Semantic source of client weights.
            client_weights: Optional fixed public weights.
            buffer_policy: Policy used for excluded model buffers.
            reference_client_id: Client supplying excluded buffer values.

        Returns:
            None.
        """
        super().__init__(
            buffer_policy=buffer_policy,
            reference_client_id=reference_client_id,
        )
        self.weight_source = str(weight_source)
        if client_weights is None and self.weight_source != "sample_count":
            raise ValueError(
                "Non-sample-count weight sources require client_weights."
            )
        self.client_weights = (
            tuple(client_weights) if client_weights is not None else None
        )

    def _linear_plan(
        self,
        updates: Sequence[ClientUpdate],
    ) -> LinearAggregationPlan:
        """Create a plan from validated finite non-negative weights.

        Args:
            updates: Validated client updates carrying public weights.

        Returns:
            A weighted-mean plan.
        """
        coefficients: list[float] = []
        for update in updates:
            weight = update.weight
            if isinstance(weight, bool) or not isinstance(weight, numbers.Real):
                raise ValueError("Aggregation weights must be real numbers.")
            value = float(weight)
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "Aggregation weights must be finite and non-negative."
                )
            coefficients.append(value)
        normalizer = sum(coefficients)
        if normalizer <= 0:
            raise ValueError("At least one aggregation weight must be positive.")
        return LinearAggregationPlan(
            coefficients=tuple(coefficients),
            normalizer=normalizer,
        )

    def _secure_plan(
        self,
        updates: Sequence[ClientUpdate],
    ) -> LinearAggregationPlan:
        """Require public non-negative integer weights in secure mode.

        Args:
            updates: Validated client updates carrying public weights.

        Returns:
            A secure-compatible weighted-mean plan.
        """

        for update in updates:
            if isinstance(update.weight, bool) or not isinstance(
                update.weight,
                numbers.Integral,
            ):
                raise ValueError(
                    "Secure weighted aggregation requires public "
                    "non-negative integer weights."
                )
        return super()._secure_plan(updates)
