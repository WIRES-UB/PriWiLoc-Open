"""Characterization tests for secure-aggregation protocol guarantees."""

from __future__ import annotations

import pytest
import torch

from models.secure_agg.protocol import (
    TrustedUser,
    freivalds_verify,
    modular_matvec,
    recover,
    server_sum,
)
from models.secure_agg.quantization import dequantize, quantize
from models.aggregation import ClientUpdate
from models.secure_agg.aggregator import SecureAggregator


def test_masks_and_projections_are_deterministic_and_round_scoped():
    user = TrustedUser(master_seed=9, N=2, q=(1 << 31) - 1, k=2, d=4)

    assert torch.equal(user.expand_mask(0, 3), user.expand_mask(0, 3))
    assert not torch.equal(user.expand_mask(0, 3), user.expand_mask(1, 3))
    assert not torch.equal(user.projection(3), user.projection(4))


def test_honest_recovery_is_exact_in_the_quantized_domain():
    q = (1 << 31) - 1
    frac_bits = 8
    user = TrustedUser(master_seed=3, N=2, q=q, k=2, d=3)
    vectors = [
        torch.tensor([-1.25, 0.5, 2.0]),
        torch.tensor([0.25, 1.5, -1.0]),
    ]
    quantized = [quantize(vector, frac_bits, q) for vector in vectors]
    masked = [
        torch.remainder(value + user.expand_mask(index, 0), q)
        for index, value in enumerate(quantized)
    ]
    aggregate = server_sum(masked, q)
    projection = user.projection(0)

    assert freivalds_verify(
        aggregate,
        projection,
        modular_matvec(projection, aggregate, q),
        q,
    )
    recovered = recover(aggregate, user.total_mask(0), q, len(vectors))
    expected = quantized[0] + quantized[1]
    assert torch.equal(recovered, expected)
    assert torch.allclose(
        dequantize(recovered, frac_bits, q),
        (vectors[0] + vectors[1]).to(dtype=torch.float64),
    )


def test_known_nonzero_server_corruption_is_rejected():
    q = (1 << 31) - 1
    user = TrustedUser(master_seed=7, N=1, q=q, k=2, d=3)
    masked = torch.remainder(
        torch.tensor([1, 2, 3]) + user.expand_mask(0, 0),
        q,
    )
    projection = user.projection(0)
    checksum = modular_matvec(projection, masked, q)
    corrupted = server_sum(
        [masked],
        q,
        error=torch.tensor([1, 0, 0]),
    )

    assert not freivalds_verify(corrupted, projection, checksum, q)


def test_secure_range_check_rejects_wraparound_boundary():
    aggregator = SecureAggregator(
        frac_bits=0,
        modulus_bits=7,
        k=1,
        master_seed=1,
    )
    updates = [
        ClientUpdate(0, {"weight": torch.tensor([40.0])}),
        ClientUpdate(1, {"weight": torch.tensor([40.0])}),
    ]

    with pytest.raises(OverflowError, match="range condition"):
        aggregator.aggregate(updates, round_idx=0)
