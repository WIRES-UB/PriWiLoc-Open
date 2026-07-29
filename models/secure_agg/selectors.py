"""Validated AP, round, and parameter selection."""

from __future__ import annotations

import hashlib
import math
import numbers
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from models.aggregation.state_dict_utils import is_aggregatable_tensor


def deterministic_seed(master_seed: int, *parts: object) -> int:
    """Derive a stable local RNG seed from explicit context values.

    Args:
        master_seed: Root deterministic seed.
        *parts: Context values that scope the derived seed.

    Returns:
        A non-negative 63-bit seed.
    """

    payload = "|".join(
        [str(master_seed), *(str(part) for part in parts)]
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class APSelector:
    """Select no APs, every AP, or explicit AP identifiers.

    Args:
        value: Normalized selector value.
    """

    value: str | tuple[int, ...]

    @classmethod
    def from_value(cls, value: Any) -> APSelector:
        """Build and validate an AP selector from configuration data.

        Args:
            value: `none`, `all`, or an iterable of AP IDs.

        Returns:
            A normalized AP selector.
        """

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized not in {"none", "all"}:
                raise ValueError(
                    "AP selector must be 'none', 'all', or a list of AP IDs."
                )
            return cls(normalized)
        try:
            raw_ids = tuple(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "AP selector must be 'none', 'all', or a list of AP IDs."
            ) from error
        if any(
            isinstance(ap_id, bool)
            or not isinstance(ap_id, numbers.Integral)
            for ap_id in raw_ids
        ):
            raise ValueError("Explicit AP IDs must be integers.")
        ids = tuple(sorted({int(ap_id) for ap_id in raw_ids}))
        return cls(ids)

    def select(self, n_aps: int) -> tuple[int, ...]:
        """Resolve selected AP IDs against the current AP count.

        Args:
            n_aps: Number of available access points.

        Returns:
            Sorted selected AP identifiers.
        """

        if n_aps <= 0:
            raise ValueError("AP count must be positive.")
        if self.value == "none":
            return ()
        if self.value == "all":
            return tuple(range(n_aps))
        selected = tuple(int(ap_id) for ap_id in self.value)
        invalid = [
            ap_id for ap_id in selected if ap_id < 0 or ap_id >= n_aps
        ]
        if invalid:
            raise ValueError(
                f"AP selector contains IDs outside [0, {n_aps}): {invalid}."
            )
        return selected


@dataclass(frozen=True)
class RoundSelector:
    """Select every round or explicit non-negative round identifiers.

    Args:
        rounds: `all` or a normalized set of round IDs.
    """

    rounds: str | frozenset[int] = "all"

    @classmethod
    def from_value(cls, value: Any) -> RoundSelector:
        """Build and validate a round selector from configuration data.

        Args:
            value: `all` or an iterable of round IDs.

        Returns:
            A normalized round selector.
        """

        if isinstance(value, str):
            if value.strip().lower() != "all":
                raise ValueError("Round selector must be 'all' or round IDs.")
            return cls("all")
        try:
            raw_rounds = tuple(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Round selector must be 'all' or a list of round IDs."
            ) from error
        if any(
            isinstance(round_idx, bool)
            or not isinstance(round_idx, numbers.Integral)
            for round_idx in raw_rounds
        ):
            raise ValueError("Round IDs must be integers.")
        rounds = frozenset(int(round_idx) for round_idx in raw_rounds)
        if not rounds or min(rounds) < 0:
            raise ValueError("Round IDs must be a non-empty non-negative list.")
        return cls(rounds)

    def matches(self, round_idx: int) -> bool:
        """Determine whether the selector includes a round.

        Args:
            round_idx: Round index to test.

        Returns:
            `True` when the rule is active for that round.
        """

        return self.rounds == "all" or int(round_idx) in self.rounds


@dataclass(frozen=True)
class SelectedParameter:
    """Describe selected elements from one state tensor.

    Args:
        key: State-dictionary key.
        selector: Index tensor, boolean mask, or `None` for all elements.
        count: Number of selected scalar elements.
    """

    key: str
    selector: torch.Tensor | None
    count: int


@dataclass(frozen=True)
class ParameterSelection:
    """Store selected tensor elements without per-scalar Python objects.

    Args:
        elements: Per-tensor selections in state-dictionary order.
    """

    elements: tuple[SelectedParameter, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        """Return selected tensor keys in state-dictionary order.

        Returns:
            Selected state-dictionary keys.
        """

        return tuple(element.key for element in self.elements)

    @property
    def count(self) -> int:
        """Return the total number of selected tensor elements.

        Returns:
            Total selected scalar count.
        """

        return sum(element.count for element in self.elements)


@dataclass(frozen=True)
class ParameterSelector:
    """Select aggregatable state tensors or deterministic scalar elements.

    Args:
        mode: Selection mode.
        value: Mode-specific key, pattern, fraction, or count.
    """

    mode: str = "all"
    value: Any = None

    def validate(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> None:
        """Validate a selector against a reference state without sampling.

        Args:
            state: Reference model state dictionary.

        Returns:
            None.
        """

        eligible = [
            (key, tensor)
            for key, tensor in state.items()
            if is_aggregatable_tensor(key, tensor)
        ]
        if not eligible:
            raise ValueError("No aggregatable tensors are available to attack.")
        mode = self.mode.strip().lower()
        if mode not in {"fraction", "count"}:
            self.select(state, seed=0)
            return

        total = sum(tensor.numel() for _, tensor in eligible)
        if mode == "fraction":
            try:
                fraction = float(self.value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Parameter fraction must be a finite number."
                ) from error
            if not math.isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError(
                    "Parameter fraction must be in the interval (0, 1]."
                )
        else:
            if isinstance(self.value, bool) or not isinstance(
                self.value,
                numbers.Integral,
            ):
                raise ValueError("Parameter count must be an integer.")
            count = int(self.value)
            if count <= 0 or count > total:
                raise ValueError(
                    f"Parameter count must be in [1, {total}]."
                )

    def select(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        seed: int,
    ) -> ParameterSelection:
        """Resolve selected elements for a concrete state dictionary.

        Args:
            state: Model state dictionary to select from.
            seed: Deterministic seed for partial selection.

        Returns:
            A normalized parameter selection.
        """

        eligible = [
            (key, tensor)
            for key, tensor in state.items()
            if is_aggregatable_tensor(key, tensor)
        ]
        if not eligible:
            raise ValueError("No aggregatable tensors are available to attack.")

        mode = self.mode.strip().lower()
        if mode == "all":
            selected = eligible
        elif mode == "keys":
            if isinstance(self.value, str):
                requested = [self.value]
            else:
                try:
                    requested = list(self.value or [])
                except TypeError as error:
                    raise ValueError(
                        "Parameter keys must be a key or list of keys."
                    ) from error
            missing = [
                key
                for key in requested
                if key not in state
                or not is_aggregatable_tensor(key, state[key])
            ]
            if missing:
                raise ValueError(
                    f"Parameter selector keys are missing or excluded: {missing}."
                )
            requested_set = set(requested)
            selected = [
                (key, tensor)
                for key, tensor in eligible
                if key in requested_set
            ]
        elif mode == "prefix":
            if self.value is None or not str(self.value):
                raise ValueError("Parameter prefix must not be empty.")
            prefix = str(self.value)
            selected = [
                (key, tensor)
                for key, tensor in eligible
                if key.startswith(prefix)
            ]
        elif mode == "regex":
            if self.value is None or str(self.value) == "":
                raise ValueError("Parameter regex must not be empty.")
            try:
                pattern = re.compile(str(self.value))
            except re.error as error:
                raise ValueError(
                    f"Invalid parameter-selector regex: {error}."
                ) from error
            selected = [
                (key, tensor)
                for key, tensor in eligible
                if pattern.search(key)
            ]
        elif mode in {"fraction", "count"}:
            return self._select_partial(eligible, seed=seed)
        else:
            raise ValueError(
                "Parameter selector mode must be all, keys, prefix, regex, "
                "fraction, or count."
            )

        if not selected:
            raise ValueError("Parameter selector matched no tensors.")
        return ParameterSelection(
            tuple(
                SelectedParameter(
                    key=key,
                    selector=None,
                    count=tensor.numel(),
                )
                for key, tensor in selected
            )
        )

    def _select_partial(
        self,
        eligible: Sequence[tuple[str, torch.Tensor]],
        *,
        seed: int,
    ) -> ParameterSelection:
        """Select a deterministic fraction or count across eligible tensors.

        Args:
            eligible: Aggregatable state tensors.
            seed: Deterministic sampling seed.

        Returns:
            A normalized partial parameter selection.
        """

        total = sum(tensor.numel() for _, tensor in eligible)
        mode = self.mode.strip().lower()
        if mode == "fraction":
            try:
                fraction = float(self.value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Parameter fraction must be a finite number."
                ) from error
            if not math.isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError(
                    "Parameter fraction must be in the interval (0, 1]."
                )
            count = max(1, int(total * fraction))
        else:
            if isinstance(self.value, bool) or not isinstance(
                self.value,
                numbers.Integral,
            ):
                raise ValueError("Parameter count must be an integer.")
            count = int(self.value)
            if count <= 0 or count > total:
                raise ValueError(
                    f"Parameter count must be in [1, {total}]."
                )

        if count == total:
            return ParameterSelection(
                tuple(
                    SelectedParameter(key, None, tensor.numel())
                    for key, tensor in eligible
                )
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        use_mask = count > total // 2
        sample_count = total - count if use_mask else count
        sampled = self._sample_unique_indexes(
            total,
            sample_count,
            generator,
        )
        offset = 0
        selections: list[SelectedParameter] = []
        for key, tensor in eligible:
            length = tensor.numel()
            local = sampled[
                (sampled >= offset) & (sampled < offset + length)
            ] - offset
            if use_mask:
                selector = torch.ones(length, dtype=torch.bool)
                selector[local] = False
                selected_count = length - int(local.numel())
            else:
                selector = local
                selected_count = int(local.numel())
            if selected_count:
                selections.append(
                    SelectedParameter(
                        key=key,
                        selector=selector,
                        count=selected_count,
                    )
                )
            offset += length
        return ParameterSelection(tuple(selections))

    @staticmethod
    def _sample_unique_indexes(
        total: int,
        count: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Sample unique indexes with memory proportional to the selection.

        Args:
            total: Total population size.
            count: Number of unique indexes to sample.
            generator: Local deterministic random generator.

        Returns:
            Sorted unique int64 indexes.
        """

        if count == 0:
            return torch.empty(0, dtype=torch.int64)
        selected = torch.empty(0, dtype=torch.int64)
        while selected.numel() < count:
            remaining = count - selected.numel()
            draw_count = max(32, int(remaining * 1.5))
            drawn = torch.randint(
                total,
                (draw_count,),
                generator=generator,
                dtype=torch.int64,
            )
            selected = torch.unique(torch.cat((selected, drawn)))
        if selected.numel() > count:
            order = torch.randperm(
                selected.numel(),
                generator=generator,
            )
            selected = selected[order[:count]]
        return torch.sort(selected).values
