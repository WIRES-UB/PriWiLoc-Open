"""Unit tests for deterministic copy-on-write server attacks."""

from __future__ import annotations

import pytest
import torch

from models.secure_agg.attacks import (
    AdditiveNoiseAttack,
    AttackRule,
    ReplacementAttack,
    ScalingAttack,
)
from models.secure_agg.selectors import (
    APSelector,
    ParameterSelector,
    RoundSelector,
)
from models.secure_agg.types import AttackContext


def _state() -> dict[str, torch.Tensor]:
    return {
        "encoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "encoder.bias": torch.ones(2),
        "bn.running_mean": torch.tensor([5.0]),
    }


def _context(*, ap_id: int = 0, rule_order: int = 0) -> AttackContext:
    return AttackContext(
        round=3,
        epoch=1,
        step=12,
        ap_id=ap_id,
        rule_order=rule_order,
        master_seed=99,
    )


def test_ap_selectors_cover_none_some_and_all():
    assert APSelector.from_value("none").select(3) == ()
    assert APSelector.from_value([2, 0, 2]).select(3) == (0, 2)
    assert APSelector.from_value("all").select(3) == (0, 1, 2)
    with pytest.raises(ValueError, match="outside"):
        APSelector.from_value([3]).select(3)


def test_large_parameter_selection_uses_compact_tensor_indexes():
    state = {"weight": torch.zeros(1_000_000)}

    partial = ParameterSelector("count", 1_000).select(state, seed=4)
    full = ParameterSelector("all").select(state, seed=4)

    assert partial.count == 1_000
    assert partial.elements[0].selector.dtype == torch.int64
    assert partial.elements[0].selector.numel() == 1_000
    assert full.elements[0].selector is None


def test_noise_is_deterministic_and_copy_on_write():
    canonical = _state()
    rule = AttackRule(
        name="noise",
        attack=AdditiveNoiseAttack(magnitude=0.5),
        ap_selector=APSelector.from_value([0]),
        rounds=RoundSelector.from_value("all"),
        parameter_selector=ParameterSelector("fraction", 0.25),
    )

    first, first_record = rule.apply(canonical, _context())
    second, second_record = rule.apply(canonical, _context())

    assert torch.equal(first["encoder.weight"], second["encoder.weight"])
    assert torch.equal(first["encoder.bias"], second["encoder.bias"])
    assert first_record == second_record
    assert first_record.affected_element_count == 2
    assert torch.equal(canonical["encoder.weight"], _state()["encoder.weight"])
    assert first["bn.running_mean"] is canonical["bn.running_mean"]

    different, _ = rule.apply(canonical, _context(ap_id=1))
    assert any(
        not torch.equal(first[key], different[key])
        for key in ("encoder.weight", "encoder.bias")
    )


def test_ordered_rules_chain_without_mutating_canonical_state():
    canonical = _state()
    scale = AttackRule(
        name="scale",
        attack=ScalingAttack(factor=2),
        ap_selector=APSelector.from_value("all"),
        rounds=RoundSelector.from_value([3]),
        parameter_selector=ParameterSelector("keys", ["encoder.bias"]),
    )
    replace = AttackRule(
        name="replace",
        attack=ReplacementAttack(value=-4),
        ap_selector=APSelector.from_value("all"),
        rounds=RoundSelector.from_value("all"),
        parameter_selector=ParameterSelector("prefix", "encoder.weight"),
    )

    scaled, _ = scale.apply(canonical, _context(rule_order=0))
    attacked, record = replace.apply(scaled, _context(rule_order=1))

    assert torch.equal(attacked["encoder.bias"], torch.full((2,), 2.0))
    assert torch.equal(
        attacked["encoder.weight"],
        torch.full((2, 3), -4.0),
    )
    assert record.order == 1
    assert torch.equal(canonical["encoder.bias"], torch.ones(2))
    assert scaled["encoder.weight"] is canonical["encoder.weight"]
