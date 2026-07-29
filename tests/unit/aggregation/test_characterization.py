"""Characterization tests for federated state aggregation."""

from __future__ import annotations

import pytest
import torch

from models.federated_learning import FederatedLearningModel


def _states() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "weight": torch.tensor([1.0, 3.0]),
            "running_mean": torch.tensor([10.0]),
            "num_batches_tracked": torch.tensor(2),
        },
        {
            "weight": torch.tensor([3.0, 5.0]),
            "running_mean": torch.tensor([20.0]),
            "num_batches_tracked": torch.tensor(4),
        },
    ]


def test_legacy_average_uses_uniform_mean_and_last_client_buffers():
    states = _states()

    result = FederatedLearningModel.average_resnet_encoder_parameters(states)

    assert torch.equal(result["weight"], torch.tensor([2.0, 4.0]))
    assert torch.equal(result["running_mean"], states[-1]["running_mean"])
    assert torch.equal(
        result["num_batches_tracked"],
        states[-1]["num_batches_tracked"],
    )


def test_legacy_average_does_not_mutate_inputs():
    states = _states()
    originals = [
        {key: value.clone() for key, value in state.items()}
        for state in states
    ]

    FederatedLearningModel.average_resnet_encoder_parameters(states)

    for state, original in zip(states, originals):
        for key in state:
            assert torch.equal(state[key], original[key])


def test_legacy_average_rejects_empty_input():
    with pytest.raises((IndexError, ValueError)):
        FederatedLearningModel.average_resnet_encoder_parameters([])
