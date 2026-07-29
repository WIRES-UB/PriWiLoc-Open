"""Tests for clean secure-pipeline configuration construction."""

from __future__ import annotations

import pytest

from models.aggregation import WeightedMeanAggregation
from models.secure_agg.attacks import ScalingAttack
from models.secure_agg.factory import (
    build_aggregation_strategy,
    build_secure_aggregator,
    validate_secure_config,
)


def _config() -> dict:
    return {
        "aggregation": {
            "_target_": (
                "models.aggregation.strategies.WeightedMeanAggregation"
            ),
            "buffer_policy": "reference_client",
            "reference_client_id": 0,
            "weight_source": "sample_count",
        },
        "secure_agg": {
            "enabled": True,
            "frac_bits": 8,
            "modulus_bits": 31,
            "k": 2,
            "master_seed": 5,
            "server": {
                "attacks": [
                    {
                        "name": "scale_selected",
                        "_target_": (
                            "models.secure_agg.attacks.ScalingAttack"
                        ),
                        "ap_selector": [0, 2],
                        "rounds": [3],
                        "parameter_selector": {
                            "mode": "prefix",
                            "value": "encoder.",
                        },
                        "factor": 4,
                        "seed_offset": 11,
                    }
                ]
            },
        },
    }


def test_factory_builds_strategy_and_ordered_attack_rules():
    config = _config()

    strategy = build_aggregation_strategy(config)
    aggregator = build_secure_aggregator(config, strategy=strategy)

    assert isinstance(strategy, WeightedMeanAggregation)
    assert aggregator is not None
    assert aggregator.strategy is strategy
    assert len(aggregator.attack_rules) == 1
    rule = aggregator.attack_rules[0]
    assert isinstance(rule.attack, ScalingAttack)
    assert rule.selected_aps(3) == (0, 2)
    assert rule.applies_to_round(3)
    assert rule.seed_offset == 11


@pytest.mark.parametrize(
    "legacy_key",
    [
        "honest",
        "error_scale",
        "error_sparsity",
        "changed_ap_ids",
        "weight_change_schedule",
    ],
)
def test_removed_legacy_server_keys_fail_with_replacement_path(legacy_key):
    config = _config()
    config["secure_agg"]["server"][legacy_key] = True

    with pytest.raises(ValueError, match="secure_agg.server.attacks"):
        validate_secure_config(config)
