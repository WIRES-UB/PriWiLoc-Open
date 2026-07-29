"""Build secure domain objects from application configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from models.aggregation import (
    AggregationStrategy,
    SumAggregation,
    UniformMeanAggregation,
    WeightedMeanAggregation,
)
from models.secure_agg.aggregator import SecureAggregator
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

LEGACY_SERVER_KEYS = frozenset(
    {
        "honest",
        "error_scale",
        "error_sparsity",
        "changed_ap_ids",
        "weight_change_schedule",
    }
)

_STRATEGIES = {
    "models.aggregation.strategies.SumAggregation": SumAggregation,
    "models.aggregation.strategies.UniformMeanAggregation": (
        UniformMeanAggregation
    ),
    "models.aggregation.strategies.WeightedMeanAggregation": (
        WeightedMeanAggregation
    ),
}
_ATTACKS = {
    "models.secure_agg.attacks.AdditiveNoiseAttack": AdditiveNoiseAttack,
    "models.secure_agg.attacks.ScalingAttack": ScalingAttack,
    "models.secure_agg.attacks.ReplacementAttack": ReplacementAttack,
}


def _value(config: Any, key: str, default: Any = None) -> Any:
    """Read one key from a mapping or typed configuration object.

    Args:
        config: Configuration object or mapping.
        key: Value name to read.
        default: Fallback when the value is unavailable.

    Returns:
        The configured value or `default`.
    """

    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _keys(config: Any) -> set[str]:
    """Collect available keys without depending on OmegaConf types.

    Args:
        config: Configuration object or mapping.

    Returns:
        Available key names.
    """

    if config is None:
        return set()
    if isinstance(config, Mapping) or hasattr(config, "keys"):
        return {str(key) for key in config.keys()}
    return set(vars(config))


def _optional(config: Any, key: str, default: Any) -> Any:
    """Read an optional value while treating explicit null as absent.

    Args:
        config: Configuration object or mapping.
        key: Value name to read.
        default: Fallback for missing or null values.

    Returns:
        The configured non-null value or `default`.
    """

    value = _value(config, key, default)
    return default if value is None else value


def _required(config: Any, key: str, owner: str) -> Any:
    """Read a required value or raise an actionable error.

    Args:
        config: Configuration object or mapping.
        key: Required value name.
        owner: Human-readable configuration owner.

    Returns:
        The configured non-null value.
    """

    value = _value(config, key)
    if value is None:
        raise ValueError(f"{owner}.{key} is required.")
    return value


def validate_secure_config(config: Any) -> None:
    """Validate the secure server schema and reject removed keys.

    Args:
        config: Root experiment configuration.

    Returns:
        None.
    """

    secure = _value(config, "secure_agg")
    server = _value(secure, "server", {})
    removed = sorted(LEGACY_SERVER_KEYS & _keys(server))
    if removed:
        raise ValueError(
            "Removed secure server keys "
            f"{removed} are not supported. Configure ordered rules under "
            "secure_agg.server.attacks instead."
        )
    attacks = _value(server, "attacks", [])
    if (
        attacks is None
        or isinstance(attacks, (str, bytes))
        or not isinstance(attacks, Sequence)
    ):
        raise ValueError("secure_agg.server.attacks must be a list.")


def build_aggregation_strategy(config: Any) -> AggregationStrategy:
    """Build the configured shared aggregation strategy.

    Args:
        config: Root experiment configuration.

    Returns:
        A validated aggregation strategy instance.
    """

    aggregation = _value(config, "aggregation")
    if aggregation is None:
        raise ValueError("An aggregation configuration is required.")
    target = str(_value(aggregation, "_target_", ""))
    strategy_cls = _STRATEGIES.get(target)
    if strategy_cls is None:
        raise ValueError(f"Unsupported aggregation strategy: {target!r}.")
    kwargs = {
        "buffer_policy": str(
            _value(aggregation, "buffer_policy", "reference_client")
        ),
        "reference_client_id": int(
            _value(aggregation, "reference_client_id", 0)
        ),
    }
    configured_weights = _value(aggregation, "client_weights")
    if strategy_cls is WeightedMeanAggregation:
        kwargs["weight_source"] = str(
            _value(aggregation, "weight_source", "sample_count")
        )
        kwargs["client_weights"] = configured_weights
    elif configured_weights is not None:
        raise ValueError(
            "aggregation.client_weights is supported only by weighted_mean."
        )
    return strategy_cls(**kwargs)


def _build_attack_rule(rule_config: Any) -> AttackRule:
    """Build and validate one configured attack rule.

    Args:
        rule_config: Attack rule configuration.

    Returns:
        A normalized deterministic attack rule.
    """

    target = str(_value(rule_config, "_target_", ""))
    attack_cls = _ATTACKS.get(target)
    if attack_cls is None:
        raise ValueError(f"Unsupported server attack: {target!r}.")

    if attack_cls is AdditiveNoiseAttack:
        attack = attack_cls(
            magnitude=float(
                _required(rule_config, "magnitude", "server attack")
            ),
            distribution=str(
                _optional(rule_config, "distribution", "normal")
            ),
        )
    elif attack_cls is ScalingAttack:
        attack = attack_cls(
            factor=float(
                _required(rule_config, "factor", "server attack")
            )
        )
    else:
        attack = attack_cls(
            value=float(_optional(rule_config, "value", 0)),
            distribution=str(
                _optional(rule_config, "distribution", "constant")
            ),
            magnitude=float(_optional(rule_config, "magnitude", 1)),
        )

    selector_config = _value(rule_config, "parameter_selector", {})
    return AttackRule(
        name=str(_value(rule_config, "name", attack_cls.__name__)),
        attack=attack,
        ap_selector=APSelector.from_value(
            _value(rule_config, "ap_selector", "none")
        ),
        rounds=RoundSelector.from_value(
            _value(rule_config, "rounds", "all")
        ),
        parameter_selector=ParameterSelector(
            mode=str(_value(selector_config, "mode", "all")),
            value=_value(selector_config, "value"),
        ),
        seed_offset=int(_value(rule_config, "seed_offset", 0)),
    )


def build_secure_aggregator(
    config: Any,
    *,
    strategy: AggregationStrategy,
) -> SecureAggregator | None:
    """Build the enabled secure aggregator from validated configuration.

    Args:
        config: Root experiment configuration.
        strategy: Shared aggregation strategy.

    Returns:
        A secure aggregator when enabled, otherwise `None`.
    """

    validate_secure_config(config)
    secure = _value(config, "secure_agg")
    if secure is None or not bool(_value(secure, "enabled", False)):
        return None
    server = _value(secure, "server", {})
    rules = tuple(
        _build_attack_rule(rule)
        for rule in _value(server, "attacks", [])
    )
    return SecureAggregator(
        frac_bits=int(_value(secure, "frac_bits", 20)),
        modulus_bits=int(_value(secure, "modulus_bits", 61)),
        k=int(_value(secure, "k", 2)),
        master_seed=int(_value(secure, "master_seed", 1234)),
        strategy=strategy,
        attack_rules=rules,
    )
