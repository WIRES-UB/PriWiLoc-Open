"""Validation and tensor policies for model-state aggregation."""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import torch

from models.aggregation.types import ClientUpdate

BATCH_NORM_BUFFERS = frozenset(
    {"running_mean", "running_var", "num_batches_tracked"}
)
REFERENCE_CLIENT_POLICY = "reference_client"


def is_batch_norm_buffer(key: str) -> bool:
    """Determine whether a state key names a standard BatchNorm buffer.

    Args:
        key: Model state-dictionary key.

    Returns:
        `True` when the key identifies an excluded BatchNorm buffer.
    """

    return key.rsplit(".", 1)[-1] in BATCH_NORM_BUFFERS


def is_aggregatable_tensor(key: str, tensor: torch.Tensor) -> bool:
    """Determine whether a tensor participates in numeric aggregation.

    Args:
        key: Model state-dictionary key.
        tensor: Tensor stored under the key.

    Returns:
        `True` for floating non-BatchNorm tensors.
    """

    return tensor.is_floating_point() and not is_batch_norm_buffer(key)


def validate_client_updates(updates: Sequence[ClientUpdate]) -> tuple[str, ...]:
    """Validate client identities and compatible state-dictionary layouts.

    Args:
        updates: Client updates to validate.

    Returns:
        State keys in canonical iteration order.
    """

    if not updates:
        raise ValueError("Aggregation requires at least one client update.")

    if any(
        isinstance(update.client_id, bool)
        or not isinstance(update.client_id, numbers.Integral)
        for update in updates
    ):
        raise ValueError("Client update IDs must be integers.")
    client_ids = [int(update.client_id) for update in updates]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("Client update IDs must be unique.")

    reference = updates[0].state
    reference_keys = tuple(reference)
    reference_key_set = set(reference_keys)
    for key, tensor in reference.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State entry {key!r} is not a tensor.")

    for update in updates[1:]:
        keys = set(update.state)
        missing = sorted(reference_key_set - keys)
        extra = sorted(keys - reference_key_set)
        if missing or extra:
            raise ValueError(
                f"Client {update.client_id} state keys do not match; "
                f"missing={missing}, extra={extra}."
            )
        for key in reference_keys:
            tensor = update.state[key]
            expected = reference[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"State entry {key!r} is not a tensor.")
            if tensor.shape != expected.shape:
                raise ValueError(
                    f"State entry {key!r} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(expected.shape)}."
                )
            if tensor.dtype != expected.dtype:
                raise ValueError(
                    f"State entry {key!r} has dtype {tensor.dtype}, "
                    f"expected {expected.dtype}."
                )
            if tensor.device != expected.device:
                raise ValueError(
                    f"State entry {key!r} is on {tensor.device}, "
                    f"expected {expected.device}."
                )

    if not any(
        is_aggregatable_tensor(key, tensor)
        for key, tensor in reference.items()
    ):
        raise ValueError("No floating non-BatchNorm model tensors were found.")
    return reference_keys


def resolve_reference_update(
    updates: Sequence[ClientUpdate],
    reference_client_id: int,
) -> ClientUpdate:
    """Resolve the client update used for excluded state entries.

    Args:
        updates: Available client updates.
        reference_client_id: Identifier of the configured reference client.

    Returns:
        The matching reference client update.
    """

    for update in updates:
        if update.client_id == reference_client_id:
            return update
    raise ValueError(
        f"Reference client {reference_client_id} is not present in the updates."
    )


def aggregate_linear_state(
    updates: Sequence[ClientUpdate],
    *,
    coefficients: Sequence[float],
    normalizer: float,
    reference_client_id: int,
    validated_keys: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate included tensors and clone excluded reference-client values.

    Args:
        updates: Compatible client updates.
        coefficients: Linear coefficient for each update.
        normalizer: Positive aggregate divisor.
        reference_client_id: Client supplying excluded values.
        validated_keys: Optional prevalidated state-key order.

    Returns:
        A newly allocated aggregated state dictionary.
    """

    keys = (
        tuple(validated_keys)
        if validated_keys is not None
        else validate_client_updates(updates)
    )
    if len(coefficients) != len(updates):
        raise ValueError("Each client update requires one coefficient.")
    if normalizer <= 0:
        raise ValueError("Aggregation normalizer must be positive.")

    reference = resolve_reference_update(updates, reference_client_id).state
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        reference_tensor = reference[key]
        if not is_aggregatable_tensor(key, reference_tensor):
            result[key] = reference_tensor.detach().clone()
            continue

        accumulator = torch.zeros_like(reference_tensor)
        for update, coefficient in zip(updates, coefficients):
            accumulator.add_(update.state[key], alpha=coefficient)
        result[key] = accumulator.div(normalizer)
    return result
