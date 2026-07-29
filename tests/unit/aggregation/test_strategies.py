"""Unit tests for shared aggregation strategies."""

from __future__ import annotations

import pytest
import torch

from models.aggregation import (
    ClientUpdate,
    SumAggregation,
    UniformMeanAggregation,
    WeightedMeanAggregation,
)


def _updates(weights=(1, 1)) -> list[ClientUpdate]:
    states = [
        {
            "layer.weight": torch.tensor([1.0, 3.0]),
            "bn.running_mean": torch.tensor([10.0]),
            "bn.num_batches_tracked": torch.tensor(2),
        },
        {
            "layer.weight": torch.tensor([3.0, 7.0]),
            "bn.running_mean": torch.tensor([20.0]),
            "bn.num_batches_tracked": torch.tensor(4),
        },
    ]
    return [
        ClientUpdate(client_id=index, state=state, weight=weights[index])
        for index, state in enumerate(states)
    ]


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (SumAggregation(), torch.tensor([4.0, 10.0])),
        (UniformMeanAggregation(), torch.tensor([2.0, 5.0])),
        (
            WeightedMeanAggregation(),
            torch.tensor([2.5, 6.0]),
        ),
    ],
)
def test_linear_strategies_compute_expected_state(strategy, expected):
    updates = _updates(weights=(1, 3))

    result = strategy.aggregate(updates)

    assert torch.equal(result.state["layer.weight"], expected)
    assert torch.equal(
        result.state["bn.running_mean"],
        updates[0].state["bn.running_mean"],
    )
    assert torch.equal(
        result.state["bn.num_batches_tracked"],
        updates[0].state["bn.num_batches_tracked"],
    )


def test_reference_client_policy_is_explicit():
    result = UniformMeanAggregation(reference_client_id=1).aggregate(_updates())

    assert torch.equal(
        result.state["bn.running_mean"],
        torch.tensor([20.0]),
    )


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_weighted_mean_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match="finite and non-negative"):
        WeightedMeanAggregation().aggregate(_updates(weights=(1, weight)))


def test_weighted_mean_rejects_zero_total_weight():
    with pytest.raises(ValueError, match="positive"):
        WeightedMeanAggregation().aggregate(_updates(weights=(0, 0)))


def test_secure_weighted_mean_rejects_floating_weights():
    with pytest.raises(ValueError, match="integer weights"):
        WeightedMeanAggregation().secure_plan(_updates(weights=(1, 2.5)))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda state: state.pop("layer.weight"),
        lambda state: state.__setitem__(
            "layer.weight",
            torch.ones(3),
        ),
        lambda state: state.__setitem__(
            "layer.weight",
            torch.ones(2, dtype=torch.float64),
        ),
    ],
)
def test_layout_mismatches_are_rejected(mutator):
    updates = _updates()
    mutator(updates[1].state)

    with pytest.raises(ValueError):
        UniformMeanAggregation().aggregate(updates)


def test_aggregation_preserves_inputs_and_returns_independent_tensors():
    updates = _updates()
    originals = [
        {key: value.clone() for key, value in update.state.items()}
        for update in updates
    ]

    result = UniformMeanAggregation().aggregate(updates)
    result.state["layer.weight"].add_(100)
    result.state["bn.running_mean"].add_(100)

    for update, original in zip(updates, originals):
        for key in update.state:
            assert torch.equal(update.state[key], original[key])
