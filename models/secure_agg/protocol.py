"""Plaintext aggregation simulation and Freivalds verification."""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import torch

from models.secure_agg.quantization import center_mod


def _seed_from_parts(master_seed: int, *parts: object) -> int:
    """Derive a deterministic signed-safe seed from contextual values.

    Args:
        master_seed: Root protocol seed.
        *parts: Context values that scope the derived seed.

    Returns:
        A deterministic non-negative 63-bit seed.
    """
    payload = "|".join([str(master_seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _uniform_mod_q(shape: tuple[int, ...], q: int, seed: int) -> torch.Tensor:
    """Generate deterministic uniform residues modulo `q`.

    Args:
        shape: Requested output shape.
        q: Exclusive upper bound and field modulus.
        seed: Local deterministic generator seed.

    Returns:
        A CPU int64 tensor of residues.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randint(0, q, shape, generator=generator, dtype=torch.int64)


def modular_sum(values: Sequence[torch.Tensor], q: int) -> torch.Tensor:
    """Sum same-shaped integer tensors modulo `q` without overflow.

    Args:
        values: Integer tensors to sum.
        q: Odd field modulus.

    Returns:
        The elementwise modular sum.
    """
    if not values:
        raise ValueError("At least one tensor is required.")
    result = torch.zeros_like(values[0], dtype=torch.int64, device="cpu")
    for value in values:
        operand = value.detach().to(device="cpu", dtype=torch.int64)
        if operand.shape != result.shape:
            raise ValueError("All modular-sum operands must have the same shape.")
        # Both operands are reduced first, so their sum is below 2q < 2^62.
        result = torch.remainder(result + torch.remainder(operand, q), q)
    return result


def modular_dot(left: torch.Tensor, right: torch.Tensor, q: int) -> int:
    """Compute an exact modular dot product using safe int64 limbs.

    Args:
        left: First integer vector.
        right: Second integer vector.
        q: Odd field modulus.

    Returns:
        The scalar dot product modulo `q`.
    """
    left_flat = torch.remainder(
        left.detach().reshape(-1).to(device="cpu", dtype=torch.int64),
        q,
    )
    right_flat = torch.remainder(
        right.detach().reshape(-1).to(device="cpu", dtype=torch.int64),
        q,
    )
    if left_flat.shape != right_flat.shape:
        raise ValueError("Dot-product operands must have the same number of elements.")
    if left_flat.numel() == 0:
        return 0

    # Choose limbs so d * (2^b - 1)^2 remains safely below signed int64.
    dimension_bits = max(0, math.ceil(math.log2(left_flat.numel())))
    limb_bits = max(4, min(20, (61 - dimension_bits) // 2))
    base = 1 << limb_bits
    mask = base - 1
    limb_count = math.ceil(max(1, q.bit_length()) / limb_bits)

    result = 0
    for left_index in range(limb_count):
        left_limb = torch.bitwise_and(
            torch.bitwise_right_shift(left_flat, left_index * limb_bits),
            mask,
        )
        for right_index in range(limb_count):
            right_limb = torch.bitwise_and(
                torch.bitwise_right_shift(right_flat, right_index * limb_bits),
                mask,
            )
            component = int(torch.sum(left_limb * right_limb, dtype=torch.int64).item())
            factor = pow(base, left_index + right_index, q)
            result = (result + component * factor) % q
    return result


def modular_matvec(matrix: torch.Tensor, vector: torch.Tensor, q: int) -> torch.Tensor:
    """Compute `matrix @ vector`.

    Args:
        matrix: Rank-two integer projection matrix.
        vector: Rank-one integer vector.
        q: Odd field modulus.

    Returns:
        The modular matrix-vector product.
    """
    if matrix.dim() != 2 or vector.dim() != 1:
        raise ValueError("Expected a rank-2 matrix and rank-1 vector.")
    if matrix.shape[1] != vector.numel():
        raise ValueError("Projection width does not match vector length.")
    return torch.tensor(
        [modular_dot(row, vector, q) for row in matrix],
        dtype=torch.int64,
    )


class TrustedUser:
    """Own deterministic mask and secret projection seeds for each round.

    Args:
        master_seed: Root protocol seed.
        N: Number of participating clients.
        q: Odd field modulus.
        k: Number of verification projections.
        d: Flattened update dimension.
    """

    def __init__(self, master_seed: int, N: int, q: int, k: int, d: int):
        """Initialize trusted-user protocol dimensions and seeds.

        Args:
            master_seed: Root protocol seed.
            N: Number of participating clients.
            q: Odd field modulus.
            k: Number of verification projections.
            d: Flattened update dimension.

        Returns:
            None.
        """
        if N <= 0 or k <= 0 or d <= 0:
            raise ValueError("N, k, and d must all be positive.")
        if q <= 2 or q > (1 << 61) - 1 or q % 2 == 0:
            raise ValueError("q must be an odd modulus in the signed-safe 61-bit range.")
        self.master_seed = int(master_seed)
        self.N = int(N)
        self.q = int(q)
        self.k = int(k)
        self.d = int(d)

    def mask_seed(self, client_index: int, round_idx: int) -> int:
        """Derive a client- and round-specific mask seed.

        Args:
            client_index: Zero-based client position.
            round_idx: Secure aggregation round index.

        Returns:
            A deterministic mask seed.
        """
        if client_index < 0 or client_index >= self.N:
            raise IndexError(f"Client index {client_index} is outside [0, {self.N}).")
        return _seed_from_parts(self.master_seed, "mask", client_index, round_idx)

    def proj_seed(self, round_idx: int) -> int:
        """Derive a round-specific projection seed.

        Args:
            round_idx: Secure aggregation round index.

        Returns:
            A deterministic projection seed.
        """
        return _seed_from_parts(self.master_seed, "projection", round_idx)

    def expand_mask(self, client_index: int, round_idx: int) -> torch.Tensor:
        """Expand one client's deterministic mask vector.

        Args:
            client_index: Zero-based client position.
            round_idx: Secure aggregation round index.

        Returns:
            A length-`d` modular mask.
        """
        return _uniform_mod_q(
            (self.d,),
            self.q,
            self.mask_seed(client_index, round_idx),
        )

    def total_mask(self, round_idx: int) -> torch.Tensor:
        """Sum every client mask for one round.

        Args:
            round_idx: Secure aggregation round index.

        Returns:
            The combined modular mask.
        """
        return modular_sum(
            [self.expand_mask(index, round_idx) for index in range(self.N)],
            self.q,
        )

    def projection(self, round_idx: int) -> torch.Tensor:
        """Generate the secret projection matrix for one round.

        Args:
            round_idx: Secure aggregation round index.

        Returns:
            A `[k, d]` modular projection matrix.
        """
        return _uniform_mod_q(
            (self.k, self.d),
            self.q,
            self.proj_seed(round_idx),
        )

    def checksum(self, masked_list: Sequence[torch.Tensor], round_idx: int) -> torch.Tensor:
        """Compute the trusted checksum for masked client vectors.

        Args:
            masked_list: One masked update vector per client.
            round_idx: Secure aggregation round index.

        Returns:
            A length `k` modular checksum.
        """
        if len(masked_list) != self.N:
            raise ValueError(f"Expected {self.N} masked vectors, got {len(masked_list)}.")
        projection = self.projection(round_idx)
        checksums = [
            modular_matvec(projection, masked, self.q)
            for masked in masked_list
        ]
        return modular_sum(checksums, self.q)


def server_sum(
    masked_list: Sequence[torch.Tensor],
    q: int,
    error: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the server aggregate with an optional injected error.

    Args:
        masked_list: Masked client vectors.
        q: Odd field modulus.
        error: Optional server-side corruption vector.

    Returns:
        The modular aggregate vector.
    """
    aggregate = modular_sum(masked_list, q)
    if error is not None:
        error_cpu = error.detach().to(device="cpu", dtype=torch.int64)
        if error_cpu.shape != aggregate.shape:
            raise ValueError("Injected server error has the wrong shape.")
        aggregate = torch.remainder(aggregate + torch.remainder(error_cpu, q), q)
    return aggregate


def freivalds_verify(
    aggregate: torch.Tensor,
    projection: torch.Tensor,
    checksum: torch.Tensor,
    q: int,
) -> bool:
    """Verify that an aggregate matches the trusted projection checksum.

    Args:
        aggregate: Server-provided modular aggregate.
        projection: Trusted secret projection matrix.
        checksum: Expected trusted checksum.
        q: Odd field modulus.

    Returns:
        `True` when the integrity equation matches.
    """
    observed = modular_matvec(projection, aggregate, q)
    expected = torch.remainder(checksum.to(device="cpu", dtype=torch.int64), q)
    return bool(torch.equal(observed, expected))


def recover(
    aggregate: torch.Tensor,
    total_mask: torch.Tensor,
    q: int,
    N: int,
) -> torch.Tensor:
    """Remove the total mask and recover a centered integer model sum.

    Args:
        aggregate: Verified modular aggregate.
        total_mask: Combined client mask.
        q: Odd field modulus.
        N: Number of participating clients.

    Returns:
        The centered quantized model sum.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if aggregate.shape != total_mask.shape:
        raise ValueError("Aggregate and total mask must have the same shape.")
    unmasked = torch.remainder(
        torch.remainder(aggregate.to(device="cpu", dtype=torch.int64), q)
        - torch.remainder(total_mask.to(device="cpu", dtype=torch.int64), q),
        q,
    )
    return center_mod(unmasked, q)
