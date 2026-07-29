"""Server attack rules."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import torch

from models.secure_agg.selectors import (
    APSelector,
    ParameterSelection,
    ParameterSelector,
    RoundSelector,
    deterministic_seed,
)
from models.secure_agg.types import AttackContext, AttackRecord


class ServerAttack(ABC):
    """Define a copy-on-write mutation for selected response elements."""

    @abstractmethod
    def apply(
        self,
        state: Mapping[str, torch.Tensor],
        selection: ParameterSelection,
        *,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Apply an attack to selected state elements.

        Args:
            state: Canonical or previously attacked model state.
            selection: Parameters and elements selected for modification.
            generator: Local deterministic random generator.

        Returns:
            A copy-on-write attacked state dictionary.
        """


def _clone_selected(
    state: Mapping[str, torch.Tensor],
    selection: ParameterSelection,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Clone only tensors selected for modification.

    Args:
        state: Source model state.
        selection: Parameters selected for modification.

    Returns:
        The shallow-copied state and flattened selected tensor clones.
    """

    result = dict(state)
    flattened: dict[str, torch.Tensor] = {}
    for element in selection.elements:
        cloned = state[element.key].detach().clone()
        result[element.key] = cloned
        flattened[element.key] = cloned.reshape(-1)
    return result, flattened


def _device_selector(
    selector: torch.Tensor | None,
    target: torch.Tensor,
):
    """Move a compact selector to a target tensor's device.

    Args:
        selector: Optional index tensor or boolean mask.
        target: Tensor to which the selector will be applied.

    Returns:
        A device-aligned selector or a full slice.
    """

    return (
        slice(None)
        if selector is None
        else selector.to(device=target.device)
    )


class AdditiveNoiseAttack(ServerAttack):
    """Add deterministic normal or uniform noise to selected elements.

    Args:
        magnitude: Positive noise scale.
        distribution: `normal` or `uniform` noise distribution.
    """

    def __init__(
        self,
        *,
        magnitude: float,
        distribution: str = "normal",
    ) -> None:
        """Initialize additive-noise parameters.

        Args:
            magnitude: Positive noise scale.
            distribution: `normal` or `uniform` noise distribution.

        Returns:
            None.
        """
        if not math.isfinite(magnitude) or magnitude <= 0:
            raise ValueError("Noise magnitude must be finite and positive.")
        if distribution not in {"normal", "uniform"}:
            raise ValueError("Noise distribution must be normal or uniform.")
        self.magnitude = float(magnitude)
        self.distribution = distribution

    def apply(
        self,
        state: Mapping[str, torch.Tensor],
        selection: ParameterSelection,
        *,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Add generated noise to selected state elements.

        Args:
            state: Source model state.
            selection: Parameters and elements selected for modification.
            generator: Local deterministic random generator.

        Returns:
            A copy-on-write attacked state dictionary.
        """

        result, flattened = _clone_selected(state, selection)
        for element in selection.elements:
            if self.distribution == "normal":
                noise = torch.randn(element.count, generator=generator)
            else:
                noise = torch.rand(
                    element.count,
                    generator=generator,
                ) * 2 - 1
            target = flattened[element.key]
            selector = _device_selector(element.selector, target)
            target[selector] += noise.to(
                device=target.device,
                dtype=target.dtype,
            ) * self.magnitude
        return result


class ScalingAttack(ServerAttack):
    """Multiply selected response elements by a configured factor.

    Args:
        factor: Finite multiplier different from one.
    """

    def __init__(self, *, factor: float) -> None:
        """Initialize the response scaling factor.

        Args:
            factor: Finite multiplier different from one.

        Returns:
            None.
        """
        if not torch.isfinite(torch.tensor(factor)) or factor == 1:
            raise ValueError("Scaling factor must be finite and differ from 1.")
        self.factor = float(factor)

    def apply(
        self,
        state: Mapping[str, torch.Tensor],
        selection: ParameterSelection,
        *,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Scale selected state elements.

        Args:
            state: Source model state.
            selection: Parameters and elements selected for modification.
            generator: Unused generator accepted by the attack interface.

        Returns:
            A copy-on-write attacked state dictionary.
        """

        del generator
        result, flattened = _clone_selected(state, selection)
        for element in selection.elements:
            target = flattened[element.key]
            selector = _device_selector(element.selector, target)
            target[selector] *= self.factor
        return result


class ReplacementAttack(ServerAttack):
    """Replace selected elements with constant or generated values.

    Args:
        value: Constant value or generated-value offset.
        distribution: `constant`, `normal`, or `uniform`.
        magnitude: Non-negative generated-value scale.
    """

    def __init__(
        self,
        *,
        value: float = 0,
        distribution: str = "constant",
        magnitude: float = 1,
    ) -> None:
        """Initialize replacement-value settings.

        Args:
            value: Constant value or generated-value offset.
            distribution: `constant`, `normal`, or `uniform`.
            magnitude: Non-negative generated-value scale.

        Returns:
            None.
        """
        if distribution not in {"constant", "normal", "uniform"}:
            raise ValueError(
                "Replacement distribution must be constant, normal, or uniform."
            )
        if (
            not math.isfinite(value)
            or not math.isfinite(magnitude)
            or magnitude < 0
        ):
            raise ValueError(
                "Replacement value and magnitude must be finite, with a "
                "non-negative magnitude."
            )
        self.value = float(value)
        self.distribution = distribution
        self.magnitude = float(magnitude)

    def apply(
        self,
        state: Mapping[str, torch.Tensor],
        selection: ParameterSelection,
        *,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Replace selected state elements.

        Args:
            state: Source model state.
            selection: Parameters and elements selected for modification.
            generator: Local deterministic random generator.

        Returns:
            A copy-on-write attacked state dictionary.
        """

        result, flattened = _clone_selected(state, selection)
        for element in selection.elements:
            if self.distribution == "normal":
                values = torch.randn(element.count, generator=generator)
            elif self.distribution == "uniform":
                values = torch.rand(
                    element.count,
                    generator=generator,
                ) * 2 - 1
            else:
                values = torch.zeros(element.count)
            values = values * self.magnitude + self.value
            target = flattened[element.key]
            selector = _device_selector(element.selector, target)
            target[selector] = values.to(
                device=target.device,
                dtype=target.dtype,
            )
        return result


@dataclass(frozen=True)
class AttackRule:
    """Combine an ordered attack with AP, round, and parameter selectors.

    Args:
        name: Stable rule name.
        attack: Attack implementation.
        ap_selector: Target AP selector.
        rounds: Target round selector.
        parameter_selector: Target parameter selector.
        seed_offset: Rule-specific deterministic seed offset.
    """

    name: str
    attack: ServerAttack
    ap_selector: APSelector
    rounds: RoundSelector
    parameter_selector: ParameterSelector
    seed_offset: int = 0

    def applies_to_round(self, round_idx: int) -> bool:
        """Determine whether the attack is scheduled for a round.

        Args:
            round_idx: Aggregation round index.

        Returns:
            `True` when the rule applies.
        """

        return self.rounds.matches(round_idx)

    def selected_aps(self, n_aps: int) -> tuple[int, ...]:
        """Resolve APs targeted by this rule.

        Args:
            n_aps: Number of available access points.

        Returns:
            Selected AP identifiers.
        """

        return self.ap_selector.select(n_aps)

    def apply(
        self,
        state: Mapping[str, torch.Tensor],
        context: AttackContext,
    ) -> tuple[dict[str, torch.Tensor], AttackRecord]:
        """Apply the rule with a local deterministic generator.

        Args:
            state: Source server response state.
            context: Round, AP, ordering, and seed context.

        Returns:
            The attacked state and its immutable audit record.
        """

        seed = deterministic_seed(
            context.master_seed,
            "attack",
            context.round,
            context.ap_id,
            context.rule_order,
            self.name,
            self.seed_offset,
        )
        selection = self.parameter_selector.select(state, seed=seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        attacked = self.attack.apply(
            state,
            selection,
            generator=generator,
        )
        return attacked, AttackRecord(
            rule_name=self.name,
            attack_type=type(self.attack).__name__,
            order=context.rule_order,
            ap_ids=(context.ap_id,),
            tensor_keys=selection.keys,
            affected_element_count=selection.count,
        )
