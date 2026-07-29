"""Typed secure-aggregation round orchestration."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from models.aggregation import (
    AggregationResult,
    AggregationStrategy,
    ClientUpdate,
    UniformMeanAggregation,
)
from models.secure_agg.attacks import AttackRule
from models.secure_agg.protocol import (
    TrustedUser,
    freivalds_verify,
    modular_matvec,
    recover,
)
from models.secure_agg.quantization import (
    StateDictLayout,
    dequantize,
    flatten_weights,
    flatten_with_layout,
    quantize,
    unflatten_weights,
)
from models.secure_agg.trust_tracking import ServerWeightChangeTracker
from models.secure_agg.types import (
    AggregationReport,
    AttackContext,
    AttackRecord,
    ServerResponse,
)


@dataclass(frozen=True)
class _PreparedRound:
    """Store validated client inputs and their canonical linear aggregate.

    Args:
        updates: Validated client updates.
        aggregation: Canonical aggregation result.
        coefficients: Secure integer client coefficients.
        normalizer: Secure integer aggregate divisor.
        vectors: Flattened client states.
        canonical_vector: Flattened canonical aggregate.
        layout: Shared state-dictionary layout.
    """

    updates: tuple[ClientUpdate, ...]
    aggregation: AggregationResult
    coefficients: tuple[int, ...]
    normalizer: int
    vectors: tuple[torch.Tensor, ...]
    canonical_vector: torch.Tensor
    layout: StateDictLayout


@dataclass(frozen=True)
class _MaskedRound:
    """Store masked and plaintext intermediates for one secure round.

    Args:
        masked_sum: Server-visible modular aggregate.
        total_mask: Combined trusted client mask.
        plain_quantized_numerator: Quantized weighted plaintext sum.
        plain_float_numerator: Floating weighted plaintext sum.
        projection: Secret verification projection.
        checksum: Trusted projection checksum.
    """

    masked_sum: torch.Tensor
    total_mask: torch.Tensor
    plain_quantized_numerator: torch.Tensor
    plain_float_numerator: torch.Tensor
    projection: torch.Tensor
    checksum: torch.Tensor


@dataclass(frozen=True)
class _PendingResponse:
    """Represent one AP response before integrity verification.

    Args:
        ap_id: Recipient access-point identifier.
        state: Response model state.
        attack_records: Attacks applied to the response.
    """

    ap_id: int
    state: Mapping[str, torch.Tensor]
    attack_records: tuple[AttackRecord, ...] = ()


def _is_prime_64(value: int) -> bool:
    """Test an unsigned 64-bit integer for primality.

    Args:
        value: Integer to test.

    Returns:
        `True` when the value is prime.
    """

    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value % prime == 0:
            return value == prime

    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        shifts += 1
        exponent //= 2
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        candidate = pow(base, exponent, value)
        if candidate in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            candidate = pow(candidate, 2, value)
            if candidate == value - 1:
                break
        else:
            return False
    return True


class SecureAggregator:
    """Run masked linear aggregation and verify each AP response.

    Args:
        frac_bits: Fixed-point fractional precision.
        modulus_bits: Mersenne-prime field size in bits.
        k: Number of verification projections.
        master_seed: Root deterministic protocol seed.
        strategy: Shared linear aggregation strategy.
        attack_rules: Ordered server attack rules.
    """

    def __init__(
        self,
        *,
        frac_bits: int,
        modulus_bits: int,
        k: int,
        master_seed: int,
        strategy: AggregationStrategy | None = None,
        attack_rules: Sequence[AttackRule] = (),
    ) -> None:
        """Initialize secure protocol parameters and audit state.

        Args:
            frac_bits: Fixed-point fractional precision.
            modulus_bits: Mersenne-prime field size in bits.
            k: Number of verification projections.
            master_seed: Root deterministic protocol seed.
            strategy: Shared linear aggregation strategy.
            attack_rules: Ordered server attack rules.

        Returns:
            None.
        """
        if frac_bits < 0:
            raise ValueError("secure_agg.frac_bits must be non-negative.")
        if modulus_bits < 3 or modulus_bits > 61:
            raise ValueError("secure_agg.modulus_bits must be in [3, 61].")
        if k <= 0:
            raise ValueError("secure_agg.k must be positive.")

        self.frac_bits = int(frac_bits)
        self.modulus_bits = int(modulus_bits)
        self.q = (1 << self.modulus_bits) - 1
        if not _is_prime_64(self.q):
            raise ValueError(
                f"2^{self.modulus_bits}-1 is not prime; choose a Mersenne-prime "
                "modulus_bits value."
            )
        self.k = int(k)
        self.master_seed = int(master_seed)
        self.strategy = strategy or UniformMeanAggregation()
        if not self.strategy.secure_compatible:
            raise ValueError(
                f"{self.strategy.name} is not supported by secure aggregation."
            )
        self.attack_rules = tuple(attack_rules)
        self.user: TrustedUser | None = None
        self.rejections_cumulative = 0
        self.weight_change_tracker = ServerWeightChangeTracker()

    @property
    def soundness_failure_probability(self) -> float:
        """Calculate the Freivalds false-accept upper bound.

        Returns:
            The verification soundness failure probability.
        """

        return math.pow(self.q, -self.k)

    @property
    def weight_change_history(self):
        """Return an immutable server audit-history snapshot.

        Returns:
            Pending server response-change events.
        """

        return tuple(self.weight_change_tracker.history)

    def validate_configuration(
        self,
        reference_state: Mapping[str, torch.Tensor],
        *,
        n_aps: int,
    ) -> None:
        """Validate weights and attack selectors against the model layout.

        Args:
            reference_state: Representative client model state.
            n_aps: Number of configured access points.

        Returns:
            None.
        """

        configured_weights = getattr(
            self.strategy,
            "client_weights",
            None,
        )
        if configured_weights is not None:
            if len(configured_weights) != n_aps:
                raise ValueError(
                    "Configured client_weights must match the AP count."
                )
            self.strategy.secure_plan(
                [
                    ClientUpdate(ap_id, reference_state, weight)
                    for ap_id, weight in enumerate(configured_weights)
                ]
            )
        for rule in self.attack_rules:
            rule.selected_aps(n_aps)
            rule.parameter_selector.validate(reference_state)

    def _trusted_user(self, n_clients: int, dimension: int) -> TrustedUser:
        """Resolve a cached trusted-user primitive for the current layout.

        Args:
            n_clients: Number of participating clients.
            dimension: Flattened model dimension.

        Returns:
            A compatible trusted-user primitive.
        """

        if (
            self.user is None
            or self.user.N != n_clients
            or self.user.d != dimension
        ):
            self.user = TrustedUser(
                master_seed=self.master_seed,
                N=n_clients,
                q=self.q,
                k=self.k,
                d=dimension,
            )
        return self.user

    def _prepare_updates(
        self,
        updates: Sequence[ClientUpdate],
    ) -> _PreparedRound:
        """Validate inputs and build the canonical secure round plan.

        Args:
            updates: Client updates participating in the round.

        Returns:
            Prepared secure-round inputs and metadata.
        """

        resolved = tuple(updates)
        aggregation = self.strategy.prepare(resolved, secure=True)
        coefficients = tuple(
            int(coefficient) for coefficient in aggregation.coefficients
        )
        normalizer = int(aggregation.normalizer)
        _, layout = flatten_weights(aggregation.state)
        vectors = tuple(
            flatten_with_layout(update.state, layout)
            for update in resolved
        )
        canonical_vector = flatten_with_layout(aggregation.state, layout)
        if layout.vector_length <= 0:
            raise ValueError("No floating non-BatchNorm model tensors were found.")
        if any(not torch.isfinite(vector).all() for vector in vectors):
            raise ValueError("Client model tensors must contain only finite values.")
        return _PreparedRound(
            updates=resolved,
            aggregation=aggregation,
            coefficients=coefficients,
            normalizer=normalizer,
            vectors=vectors,
            canonical_vector=canonical_vector,
            layout=layout,
        )

    def _certify_range(self, prepared: _PreparedRound) -> None:
        """Reject a weighted quantized sum that could wrap modulo `q`.

        Args:
            prepared: Prepared secure-round inputs.

        Returns:
            None.
        """

        scale = 1 << self.frac_bits
        worst_case = sum(
            coefficient
            * (
                int(torch.round(vector.abs().max() * scale).item())
                if vector.numel()
                else 0
            )
            for coefficient, vector in zip(
                prepared.coefficients,
                prepared.vectors,
            )
        )
        if worst_case >= self.q // 2:
            raise OverflowError(
                "Secure aggregation range condition violated: "
                f"weighted max sum {worst_case} is not below q/2={self.q // 2}. "
                "Raise secure_agg.modulus_bits or lower secure_agg.frac_bits."
            )

    def _quantize_and_mask(
        self,
        prepared: _PreparedRound,
        *,
        round_idx: int,
    ) -> _MaskedRound:
        """Quantize weighted updates and construct the masked server sum.

        Args:
            prepared: Prepared secure-round inputs.
            round_idx: Secure aggregation round index.

        Returns:
            Masked and plaintext round intermediates.
        """

        dimension = prepared.layout.vector_length
        user = self._trusted_user(len(prepared.updates), dimension)
        plain_quantized = torch.zeros(dimension, dtype=torch.int64)
        plain_float = torch.zeros(dimension, dtype=torch.float64)
        total_mask = torch.zeros(dimension, dtype=torch.int64)
        masked_sum = torch.zeros(dimension, dtype=torch.int64)

        for client_index, (vector, coefficient) in enumerate(
            zip(prepared.vectors, prepared.coefficients)
        ):
            quantized = quantize(vector, self.frac_bits, self.q)
            weighted = quantized * coefficient
            mask = user.expand_mask(client_index, round_idx)
            masked = torch.remainder(weighted + mask, self.q)
            plain_quantized += weighted
            plain_float += vector * coefficient
            total_mask = torch.remainder(total_mask + mask, self.q)
            masked_sum = torch.remainder(masked_sum + masked, self.q)

        projection = user.projection(round_idx)
        checksum = modular_matvec(projection, masked_sum, self.q)
        return _MaskedRound(
            masked_sum=masked_sum,
            total_mask=total_mask,
            plain_quantized_numerator=plain_quantized,
            plain_float_numerator=plain_float,
            projection=projection,
            checksum=checksum,
        )

    @staticmethod
    def _build_server_responses(
        prepared: _PreparedRound,
    ) -> list[_PendingResponse]:
        """Create one AP-ordered response sharing the canonical state.

        Args:
            prepared: Prepared secure-round inputs.

        Returns:
            Pending responses ordered by AP identifier.
        """

        return [
            _PendingResponse(
                ap_id=update.client_id,
                state=prepared.aggregation.state,
            )
            for update in sorted(
                prepared.updates,
                key=lambda item: item.client_id,
            )
        ]

    def _apply_attack_rules(
        self,
        responses: list[_PendingResponse],
        *,
        round_idx: int,
        epoch_idx: int,
        step_idx: int,
    ) -> list[_PendingResponse]:
        """Apply scheduled attack rules in configuration order.

        Args:
            responses: Pending canonical AP responses.
            round_idx: Secure aggregation round index.
            epoch_idx: Training epoch index.
            step_idx: Global training step.

        Returns:
            AP responses after configured attacks.
        """

        n_aps = len(responses)
        by_ap_id = {response.ap_id: response for response in responses}
        for order, rule in enumerate(self.attack_rules):
            if not rule.applies_to_round(round_idx):
                continue
            for ap_id in rule.selected_aps(n_aps):
                response = by_ap_id[ap_id]
                state, record = rule.apply(
                    response.state,
                    AttackContext(
                        round=round_idx,
                        epoch=epoch_idx,
                        step=step_idx,
                        ap_id=ap_id,
                        rule_order=order,
                        master_seed=self.master_seed,
                    ),
                )
                by_ap_id[ap_id] = _PendingResponse(
                    ap_id=ap_id,
                    state=state,
                    attack_records=(*response.attack_records, record),
                )
        return [by_ap_id[ap_id] for ap_id in range(n_aps)]

    def _masked_response(
        self,
        response: _PendingResponse,
        prepared: _PreparedRound,
        masked: _MaskedRound,
    ) -> tuple[torch.Tensor, bool]:
        """Translate an attacked state into a masked-field response.

        Args:
            response: Pending AP response.
            prepared: Prepared secure-round inputs.
            masked: Masked round intermediates.

        Returns:
            The modular response vector and effective-change flag.
        """

        if not response.attack_records:
            return masked.masked_sum, False
        vector = flatten_with_layout(response.state, prepared.layout)
        scale = 1 << self.frac_bits
        scaled_target = torch.round(
            vector * scale * prepared.normalizer
        )
        if (
            not torch.isfinite(scaled_target).all()
            or (
                scaled_target.numel()
                and float(scaled_target.abs().max()) >= self.q // 2
            )
        ):
            raise OverflowError(
                "Attacked server response exceeds the secure field range."
            )
        target = scaled_target.to(dtype=torch.int64)
        canonical_target = torch.round(
            prepared.canonical_vector * scale * prepared.normalizer
        ).to(dtype=torch.int64)
        delta = target - canonical_target
        if not torch.any(delta):
            return masked.masked_sum, False
        attacked_numerator = masked.plain_quantized_numerator + delta
        if (
            attacked_numerator.numel()
            and int(attacked_numerator.abs().max().item()) >= self.q // 2
        ):
            raise OverflowError(
                "Attacked server response exceeds the secure field range."
            )
        return torch.remainder(masked.masked_sum + delta, self.q), True

    def _verify_responses(
        self,
        responses: Sequence[_PendingResponse],
        prepared: _PreparedRound,
        masked: _MaskedRound,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[bool, ...],
        tuple[bool, ...],
    ]:
        """Verify every AP response independently before recovery.

        Args:
            responses: Pending AP responses.
            prepared: Prepared secure-round inputs.
            masked: Masked round intermediates.

        Returns:
            Modular responses, verification decisions, and change flags.
        """

        masked_responses: list[torch.Tensor] = []
        decisions: list[bool] = []
        effective_changes: list[bool] = []
        for response in responses:
            candidate, effective = self._masked_response(
                response,
                prepared,
                masked,
            )
            verified = freivalds_verify(
                candidate,
                masked.projection,
                masked.checksum,
                self.q,
            )
            if effective and verified:
                raise RuntimeError(
                    "A changed server response passed verification; this "
                    "indicates a protocol implementation bug."
                )
            if not effective and not verified:
                raise RuntimeError(
                    "An unchanged server response failed verification."
                )
            masked_responses.append(candidate)
            decisions.append(verified)
            effective_changes.append(effective)
        return (
            tuple(masked_responses),
            tuple(decisions),
            tuple(effective_changes),
        )

    def _recover_responses(
        self,
        responses: Sequence[_PendingResponse],
        decisions: Sequence[bool],
        effective_changes: Sequence[bool],
        prepared: _PreparedRound,
        masked: _MaskedRound,
    ) -> tuple[list[ServerResponse], float, float]:
        """Recover the canonical state only for responses that verified.

        Args:
            responses: Pending AP responses.
            decisions: Per-response verification decisions.
            effective_changes: Per-response effective-change flags.
            prepared: Prepared secure-round inputs.
            masked: Masked round intermediates.

        Returns:
            Public responses, maximum recovery error, and relative
            quantization error.
        """

        if not any(decisions):
            return (
                [
                    ServerResponse(
                        ap_id=response.ap_id,
                        state=response.state,
                        attacked=bool(effective),
                        verified=False,
                        attack_records=response.attack_records,
                    )
                    for response, effective in zip(
                        responses,
                        effective_changes,
                    )
                ],
                float("nan"),
                float("nan"),
            )

        recovered = recover(
            masked.masked_sum,
            masked.total_mask,
            self.q,
            len(prepared.updates),
        )
        if not torch.equal(recovered, masked.plain_quantized_numerator):
            raise RuntimeError(
                "Honest recovery did not reproduce the quantized weighted sum."
            )
        recovered_float = dequantize(
            recovered,
            self.frac_bits,
            self.q,
        )
        recover_error = float(
            torch.max(
                torch.abs(recovered_float - masked.plain_float_numerator)
            ).item()
        )
        error_bound = sum(prepared.coefficients) * (
            2.0 ** (-self.frac_bits)
        )
        if recover_error >= error_bound:
            raise RuntimeError(
                "Recovered model exceeded the fixed-point error bound: "
                f"{recover_error} >= {error_bound}."
            )

        recovered_normalized = recovered_float / prepared.normalizer
        denominator = float(
            torch.linalg.vector_norm(prepared.canonical_vector).item()
        )
        numerator = float(
            torch.linalg.vector_norm(
                recovered_normalized - prepared.canonical_vector
            ).item()
        )
        quant_rel_err = numerator / max(
            denominator,
            torch.finfo(torch.float64).eps,
        )
        recovered_state = unflatten_weights(
            recovered_normalized,
            prepared.layout,
        )
        public_responses = [
            ServerResponse(
                ap_id=response.ap_id,
                state=recovered_state if verified else response.state,
                attacked=bool(effective),
                verified=bool(verified),
                attack_records=response.attack_records,
            )
            for response, verified, effective in zip(
                responses,
                decisions,
                effective_changes,
            )
        ]
        return public_responses, recover_error, quant_rel_err

    def _build_report(
        self,
        *,
        prepared: _PreparedRound,
        responses: Sequence[ServerResponse],
        round_idx: int,
        epoch_idx: int,
        step_idx: int,
        recover_error: float,
        quant_rel_err: float,
        started: float,
    ) -> AggregationReport:
        """Build a report and record completed-round audit data.

        Args:
            prepared: Prepared secure-round inputs.
            responses: Verified public server responses.
            round_idx: Secure aggregation round index.
            epoch_idx: Training epoch index.
            step_idx: Global training step.
            recover_error: Maximum fixed-point recovery error.
            quant_rel_err: Relative quantization error.
            started: Round start time from `perf_counter`.

        Returns:
            Immutable diagnostics for the completed round.
        """

        decisions = tuple(response.verified for response in responses)
        changed_ap_ids = tuple(
            response.ap_id for response in responses if response.attacked
        )
        attack_records = tuple(
            record
            for response in responses
            for record in response.attack_records
        )
        accepted = all(decisions)
        if not accepted:
            self.rejections_cumulative += 1
        self.weight_change_tracker.record(
            epoch=epoch_idx,
            round_idx=round_idx,
            changed_ap_ids=changed_ap_ids,
            step_idx=step_idx,
            attack_records=attack_records,
        )
        return AggregationReport(
            round=int(round_idx),
            epoch=int(epoch_idx),
            step=int(step_idx),
            accepted=accepted,
            server_honest=not attack_records,
            ap_trust=decisions,
            changed_ap_ids=changed_ap_ids,
            attack_records=attack_records,
            aggregation=prepared.aggregation,
            recover_max_abs_err=recover_error,
            quant_rel_err=quant_rel_err,
            rejections_cumulative=self.rejections_cumulative,
            round_seconds=time.perf_counter() - started,
        )

    def aggregate(
        self,
        updates: Sequence[ClientUpdate],
        round_idx: int,
        epoch_idx: int = 0,
        step_idx: int | None = None,
    ) -> tuple[list[ServerResponse], AggregationReport]:
        """Run one complete secure aggregation round.

        Args:
            updates: Client updates participating in the round.
            round_idx: Secure aggregation round index.
            epoch_idx: Training epoch index.
            step_idx: Optional global training step.

        Returns:
            AP-specific server responses and the completed-round report.
        """

        started = time.perf_counter()
        resolved_step = int(round_idx if step_idx is None else step_idx)
        prepared = self._prepare_updates(updates)
        client_ids = tuple(
            sorted(update.client_id for update in prepared.updates)
        )
        if client_ids != tuple(range(len(prepared.updates))):
            raise ValueError(
                "Secure aggregation requires contiguous AP IDs starting at 0."
            )
        self._certify_range(prepared)
        masked = self._quantize_and_mask(prepared, round_idx=round_idx)
        pending = self._build_server_responses(prepared)
        attacked = self._apply_attack_rules(
            pending,
            round_idx=round_idx,
            epoch_idx=epoch_idx,
            step_idx=resolved_step,
        )
        _, decisions, effective_changes = self._verify_responses(
            attacked,
            prepared,
            masked,
        )
        responses, recover_error, quant_rel_err = self._recover_responses(
            attacked,
            decisions,
            effective_changes,
            prepared,
            masked,
        )
        report = self._build_report(
            prepared=prepared,
            responses=responses,
            round_idx=round_idx,
            epoch_idx=epoch_idx,
            step_idx=resolved_step,
            recover_error=recover_error,
            quant_rel_err=quant_rel_err,
            started=started,
        )
        return responses, report
