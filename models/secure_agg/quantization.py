"""State-dict flattening and exact fixed-point conversion."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch

from models.aggregation.state_dict_utils import is_batch_norm_buffer


"""
Client model state dictionaries
        │
        │ flatten_weights()
        ▼
Floating-point vectors
        │
        │ quantize()
        │ x_int = round(x_float x 2^frac_bits) mod q
        ▼
Quantized integer vectors
        │
        ├── apply client coefficients
        ├── add client masks
        ├── sum modulo q
        └── verify server response
        │
        │ recover()
        ▼
Recovered quantized weighted sum
        │
        │ dequantize()
        │ x_float = centered(x_int mod q) / 2^frac_bits
        ▼
Recovered floating-point weighted sum
        │
        │ divide by aggregation normalizer
        ▼
Recovered floating-point aggregate
        │
        │ unflatten_weights(layout)
        ▼
Aggregated model state dictionary
"""


@dataclass
class TensorLayout:
    """Describe one state tensor. It is used for reconstuction of the model
    ater flattening.

    Args:
        key: State-dictionary key.
        shape: Original tensor shape.
        dtype: Original tensor dtype.
        device: Original tensor device.
        start: Inclusive flattened-vector offset.
        end: Exclusive flattened-vector offset.
        is_bn: Whether the tensor is a BatchNorm buffer.
        included: Whether the tensor participates in secure aggregation.
        preserved_value: Cloned value retained for excluded tensors.
    """

    key: str
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    start: int
    end: int
    is_bn: bool
    included: bool
    preserved_value: torch.Tensor | None = None


@dataclass
class StateDictLayout:
    """Describe the flattened layout of an aggregatable state dictionary of model.

    Args:
        entries: Per-tensor layout descriptions.
        vector_length: Total number of included scalar elements.
    """

    entries: list[TensorLayout]
    vector_length: int


def center_mod(value: torch.Tensor, q: int) -> torch.Tensor:
    """Map integer residues into the centered interval for an odd modulus.

    Args:
        value: Integer values or residues to center.
        q: Odd field modulus.

    Returns:
        Centered signed representatives modulo `q`.
    """
    if q <= 2 or q % 2 == 0:
        raise ValueError("q must be an odd integer greater than 2.")
    residue = torch.remainder(value.to(dtype=torch.int64), q)
    return torch.where(residue > q // 2, residue - q, residue)


def flatten_weights(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, StateDictLayout]:
    """Flatten model state dictionary into 1-d flat vector. Flatten
    floating non-BN tensors while preserving excluded values.

    Args:
        state_dict: Model state to flatten.

    Returns:
        The CPU float64 vector and its reconstruction layout.
    """
    entries: list[TensorLayout] = []
    flattened: list[torch.Tensor] = []
    offset = 0

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State entry {key!r} is not a tensor.")
        is_bn = is_batch_norm_buffer(key)
        included = tensor.is_floating_point() and not is_bn
        length = tensor.numel() if included else 0
        entries.append(
            TensorLayout(
                key=key,
                shape=tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
                start=offset,
                end=offset + length,
                is_bn=is_bn,
                included=included,
                preserved_value=(
                    None if included else tensor.detach().clone()
                ),
            )
        )
        if included:
            # Fixed-point work is deliberately performed on CPU float64/int64.
            flattened.append(tensor.detach().reshape(-1).to(device="cpu", dtype=torch.float64))
            offset += length

    vector = (
        torch.cat(flattened)
        if flattened
        else torch.empty(0, dtype=torch.float64)
    )
    return vector, StateDictLayout(entries=entries, vector_length=offset)


def flatten_with_layout(
    state_dict: Mapping[str, torch.Tensor],
    layout: StateDictLayout,
) -> torch.Tensor:
    """Flatten included tensors using a validated reference layout.

    Args:
        state_dict: Model state matching the reference layout.
        layout: Reference state-dictionary layout.

    Returns:
        A CPU float64 vector in layout order.
    """

    flattened = [
        state_dict[entry.key]
        .detach()
        .reshape(-1)
        .to(device="cpu", dtype=torch.float64)
        for entry in layout.entries
        if entry.included
    ]
    return (
        torch.cat(flattened)
        if flattened
        else torch.empty(0, dtype=torch.float64)
    )


def unflatten_weights(
    vector: torch.Tensor,
    layout: StateDictLayout,
) -> OrderedDict[str, torch.Tensor]:
    """Reassemble a state dictionary and restore excluded values. Used for
    reconstruction of model state dictionary.

    Args:
        vector: Flattened included tensor values.
        layout: Layout describing reconstruction and preserved values.

    Returns:
        An ordered model state dictionary on original devices and dtypes.
    """
    flat = vector.reshape(-1)
    if flat.numel() != layout.vector_length:
        raise ValueError(
            f"Vector has {flat.numel()} elements, expected {layout.vector_length}."
        )

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for entry in layout.entries:
        if entry.included:
            value = flat[entry.start:entry.end].reshape(entry.shape)
            result[entry.key] = value.to(device=entry.device, dtype=entry.dtype)
        else:
            if entry.preserved_value is None:
                raise RuntimeError(f"Missing preserved value for {entry.key!r}.")
            result[entry.key] = entry.preserved_value.detach().clone()
    return result


def quantize(x_float: torch.Tensor, frac_bits: int, q: int) -> torch.Tensor:
    """Quantize a floating vector to centered fixed-point residues.

    Args:
        x_float: Floating values to quantize.
        frac_bits: Number of fixed-point fractional bits.
        q: Odd field modulus.

    Returns:
        A CPU int64 tensor of centered residues.
    """
    if frac_bits < 0:
        raise ValueError("frac_bits must be non-negative.")
    values = x_float.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("Cannot quantize NaN or infinite model weights.")
    scaled = torch.round(values * float(1 << frac_bits))
    int64_limit = torch.iinfo(torch.int64).max
    if scaled.numel() and float(scaled.abs().max()) > int64_limit:
        raise OverflowError("Scaled weights exceed int64; lower frac_bits.")
    return center_mod(scaled.to(dtype=torch.int64), q)


def dequantize(x_int: torch.Tensor, frac_bits: int, q: int) -> torch.Tensor:
    """Convert centered modular integers back to floating values.

    Args:
        x_int: Quantized integer values or residues.
        frac_bits: Number of fixed-point fractional bits.
        q: Odd field modulus.

    Returns:
        A CPU float64 tensor.
    """
    if frac_bits < 0:
        raise ValueError("frac_bits must be non-negative.")
    centered = center_mod(x_int, q)
    return centered.to(device="cpu", dtype=torch.float64) / float(1 << frac_bits)
