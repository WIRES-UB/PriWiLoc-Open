"""Small CPU integration tests for AP-specific secure model updates."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.aggregation import (
    UniformMeanAggregation,
    WeightedMeanAggregation,
)
from models.federated_learning import FederatedLearningModel
from models.federated_learning_secagg import SecureFederatedLearningModel
from models.secure_agg.aggregator import SecureAggregator
from models.secure_agg.attacks import AttackRule, ScalingAttack
from models.secure_agg.selectors import (
    APSelector,
    ParameterSelector,
    RoundSelector,
)


class _RoundHolder:
    get_resnet_encoder_parameters = (
        FederatedLearningModel.get_resnet_encoder_parameters
    )
    set_resnet_encoder_parameters = (
        FederatedLearningModel.set_resnet_encoder_parameters
    )
    get_aggregation_weights = FederatedLearningModel.get_aggregation_weights

    def __init__(self, aggregator):
        self.resnet_encoder_list = [
            torch.nn.Linear(1, 1, bias=False)
            for _ in range(3)
        ]
        for encoder, value in zip(
            self.resnet_encoder_list,
            (1.0, 2.0, 6.0),
        ):
            encoder.weight.data.fill_(value)
        self.secure_aggregator = aggregator
        self.aggregation_strategy = aggregator.strategy
        self._secure_round_idx = torch.tensor(0)
        self._secure_rejections_cumulative = torch.tensor(0)
        self.config = SimpleNamespace(
            model=SimpleNamespace(average_weight_every_n_batches=1)
        )
        self.trainer = SimpleNamespace(current_epoch=0, global_step=1)
        self.report = None

    def print(self, *args, **kwargs):
        return None

    def _log_secure_report(self, report):
        self.report = report


def test_secure_hook_loads_only_each_aps_verified_response():
    rule = AttackRule(
        name="reject_ap_one",
        attack=ScalingAttack(factor=2),
        ap_selector=APSelector.from_value([1]),
        rounds=RoundSelector.from_value("all"),
        parameter_selector=ParameterSelector(),
    )
    aggregator = SecureAggregator(
        frac_bits=12,
        modulus_bits=31,
        k=2,
        master_seed=17,
        strategy=UniformMeanAggregation(),
        attack_rules=(rule,),
    )
    holder = _RoundHolder(aggregator)

    SecureFederatedLearningModel.on_train_batch_end(
        holder,
        outputs=None,
        batch=None,
        batch_idx=0,
    )

    expected_mean = (1 + 2 + 6) / 3
    assert holder.resnet_encoder_list[0].weight.item() == expected_mean
    assert holder.resnet_encoder_list[1].weight.item() == 2.0
    assert holder.resnet_encoder_list[2].weight.item() == expected_mean
    assert holder.report.ap_trust == (True, False, True)
    assert holder._secure_round_idx.item() == 1
    assert holder._secure_rejections_cumulative.item() == 1


def test_secure_hook_passes_configured_non_uniform_client_weights():
    strategy = WeightedMeanAggregation(
        weight_source="ap_reliability",
        client_weights=(1, 2, 3),
    )
    aggregator = SecureAggregator(
        frac_bits=12,
        modulus_bits=31,
        k=2,
        master_seed=17,
        strategy=strategy,
    )
    holder = _RoundHolder(aggregator)

    SecureFederatedLearningModel.on_train_batch_end(
        holder,
        outputs=None,
        batch=None,
        batch_idx=0,
    )

    expected = (1 * 1 + 2 * 2 + 3 * 6) / 6
    assert all(
        encoder.weight.item() == pytest.approx(expected, abs=1e-3)
        for encoder in holder.resnet_encoder_list
    )
    assert holder.report.aggregation.coefficients == (1.0, 2.0, 3.0)
