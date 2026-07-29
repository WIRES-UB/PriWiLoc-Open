"""Unit tests for typed secure-aggregation rounds."""

from __future__ import annotations

import pytest
import torch

from models.aggregation import (
    ClientUpdate,
    SumAggregation,
    UniformMeanAggregation,
    WeightedMeanAggregation,
)
from models.secure_agg.aggregator import SecureAggregator
from models.secure_agg.attacks import AttackRule, ScalingAttack
from models.secure_agg.selectors import (
    APSelector,
    ParameterSelector,
    RoundSelector,
)


def _updates(weights=(1, 1, 1)) -> list[ClientUpdate]:
    return [
        ClientUpdate(
            client_id=client_id,
            state={
                "weight": torch.tensor([float(client_id + 1)]),
                "bn.running_mean": torch.tensor([10.0 + client_id]),
                "bn.num_batches_tracked": torch.tensor(client_id),
            },
            weight=weights[client_id],
        )
        for client_id in range(3)
    ]


@pytest.mark.parametrize(
    ("strategy", "weights", "expected"),
    [
        (SumAggregation(), (1, 1, 1), 6.0),
        (UniformMeanAggregation(), (1, 1, 1), 2.0),
        (WeightedMeanAggregation(), (1, 2, 3), 14 / 6),
    ],
)
def test_secure_linear_strategies_recover_each_ap_response(
    strategy,
    weights,
    expected,
):
    aggregator = SecureAggregator(
        frac_bits=12,
        modulus_bits=31,
        k=2,
        master_seed=4,
        strategy=strategy,
    )

    responses, report = aggregator.aggregate(
        _updates(weights),
        round_idx=0,
    )

    assert len(responses) == 3
    assert [response.ap_id for response in responses] == [0, 1, 2]
    assert all(response.verified for response in responses)
    assert report.ap_trust == (True, True, True)
    assert responses[0].state["weight"].item() == pytest.approx(
        expected,
        abs=1e-3,
    )
    assert torch.equal(
        responses[0].state["bn.running_mean"],
        torch.tensor([10.0]),
    )


def test_secure_weighted_mean_requires_public_integer_weights():
    aggregator = SecureAggregator(
        frac_bits=8,
        modulus_bits=31,
        k=1,
        master_seed=1,
        strategy=WeightedMeanAggregation(),
    )

    with pytest.raises(ValueError, match="integer weights"):
        aggregator.aggregate(
            _updates(weights=(1, 2.5, 3)),
            round_idx=0,
        )


def test_attack_selectors_are_validated_against_model_layout_at_startup():
    rule = AttackRule(
        name="bad_prefix",
        attack=ScalingAttack(factor=2),
        ap_selector=APSelector.from_value([2]),
        rounds=RoundSelector.from_value([50]),
        parameter_selector=ParameterSelector("prefix", "missing."),
    )
    aggregator = SecureAggregator(
        frac_bits=8,
        modulus_bits=31,
        k=1,
        master_seed=1,
        attack_rules=(rule,),
    )

    with pytest.raises(ValueError, match="matched no tensors"):
        aggregator.validate_configuration(
            {"weight": torch.ones(2)},
            n_aps=3,
        )


def test_each_ap_verifies_and_receives_only_its_own_response():
    canonical_inputs = _updates()
    originals = [
        {
            key: value.clone()
            for key, value in update.state.items()
        }
        for update in canonical_inputs
    ]
    rule = AttackRule(
        name="scale_ap_one",
        attack=ScalingAttack(factor=2),
        ap_selector=APSelector.from_value([1]),
        rounds=RoundSelector.from_value("all"),
        parameter_selector=ParameterSelector(),
    )
    aggregator = SecureAggregator(
        frac_bits=8,
        modulus_bits=31,
        k=2,
        master_seed=8,
        attack_rules=(rule,),
    )

    responses, report = aggregator.aggregate(canonical_inputs, round_idx=0)

    assert report.ap_trust == (True, False, True)
    assert report.changed_ap_ids == (1,)
    assert responses[0].state is responses[2].state
    assert responses[1].state is not responses[0].state
    assert torch.equal(
        report.aggregation.state["weight"],
        torch.tensor([2.0]),
    )
    for update, original in zip(canonical_inputs, originals):
        for key in update.state:
            assert torch.equal(update.state[key], original[key])


def test_all_attacked_responses_are_returned_but_never_verified():
    rule = AttackRule(
        name="scale_all",
        attack=ScalingAttack(factor=3),
        ap_selector=APSelector.from_value("all"),
        rounds=RoundSelector.from_value("all"),
        parameter_selector=ParameterSelector(),
    )
    aggregator = SecureAggregator(
        frac_bits=8,
        modulus_bits=31,
        k=2,
        master_seed=8,
        attack_rules=(rule,),
    )

    responses, report = aggregator.aggregate(_updates(), round_idx=0)

    assert len(responses) == 3
    assert not any(response.verified for response in responses)
    assert report.rejections_cumulative == 1
    assert not report.accepted


def test_unrepresentable_no_effect_attack_is_recorded_without_rejection():
    rule = AttackRule(
        name="scale_zeros",
        attack=ScalingAttack(factor=2),
        ap_selector=APSelector.from_value([0]),
        rounds=RoundSelector.from_value("all"),
        parameter_selector=ParameterSelector(),
    )
    aggregator = SecureAggregator(
        frac_bits=8,
        modulus_bits=31,
        k=2,
        master_seed=8,
        attack_rules=(rule,),
    )
    updates = [
        ClientUpdate(
            client_id=client_id,
            state={"weight": torch.zeros(2)},
        )
        for client_id in range(2)
    ]

    responses, report = aggregator.aggregate(updates, round_idx=0)

    assert all(response.verified for response in responses)
    assert not responses[0].attacked
    assert report.changed_ap_ids == ()
    assert len(report.attack_records) == 1
    assert not report.server_honest
