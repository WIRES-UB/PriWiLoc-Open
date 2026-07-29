"""Hydra-compatible configuration classes for PriWiLoc experiments.

This module provides configuration classes that work with Hydra for configuration management.
The original config.py is kept for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from omegaconf import MISSING

@dataclass
class DatasetStandardizationConfig:
    """Configure per-AP input standardization.

    Args:
        enabled: Set to True to standardize data.
        eps: Minimum standard deviation.
    """

    enabled: bool = False
    eps: float = 1e-6


@dataclass
class DatasetConfig:
    """Configuration for the dataset.

    Args:
        batch_size: Samples per batch.
        num_workers: Worker processes per loader.
        prefetch_factor: Batches prefetched by each worker.
        sequence_length: Requested sequence length.
        shuffle_train: Set True to shuffle training samples
        train_val_split: Fraction used to split training set into train and validations sets. Only applicable if `val_data_path` is `None`.
        split_seed: Seed for splitting.
        train_data_path: Training dataset path.
        val_data_path: Validation dataset path.
        test_data_path: Test dataset path.
        train_n_aps: Training access-point count.
        val_n_aps: Validation access-point count.
        test_n_aps: Test access-point count.
        eval_datasets: Additional datasets for evaluation at the end.
        standardization: Per-AP standardization settings.
    """
    
    batch_size: int = 16
    num_workers: int = 8
    prefetch_factor: int = 2
    sequence_length: int = 20
    shuffle_train: bool = False
    train_val_split: Optional[list[float]] = None
    split_seed: int = 42
    
    train_data_path: str = MISSING  # Required field
    val_data_path: Optional[str] = None
    test_data_path: str = MISSING  # Required field
    
    train_n_aps: int = 4
    val_n_aps: int = 4
    test_n_aps: int = 4
    eval_datasets: list[dict[str, Any]] = field(default_factory=list)
    standardization: DatasetStandardizationConfig = field(
        default_factory=DatasetStandardizationConfig
    )


@dataclass
class ModelConfig:
    """Configuration for the model.

    Args:
        name: Model display name.
        dropout: Encoder dropout probability.
        in_channels: Number of input feature channels.
        average_weight_every_n_batches: Batches between aggregation rounds.
    """
    
    name: str = "FederatedLearningModel"
    dropout: float = 0.3
    in_channels: int = 1
    average_weight_every_n_batches: int = 10


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing.

    Args:
        monitor: Metric used.
        mode: Minimize or maximize the metric.
        save_top_k: Number of best checkpoints to save.
        filename: Lightning checkpoint filename template.
    """
    
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 1
    filename: str = "{epoch:02d}"


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping.

    Args:
        monitor: Metric monitored.
        patience: Epochs without improvement before stopping.
        mode: Whether lower or higher values are preferred.
        verbose: Whether stopping decisions are logged.
    """
    
    monitor: str = "val_loss"
    patience: int = 5
    mode: str = "min"
    verbose: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for visualization callback.

    Args:
        max_visualization_samples: Maximum samples retained for plots.
        display_viz_data_epoch_interval: Epochs between visualization runs.
    """
    
    max_visualization_samples: int = 10
    display_viz_data_epoch_interval: int = 1


@dataclass
class TrainerConfig:
    """Configuration for PyTorch Lightning Trainer.

    Args:
        accelerator: Hardware accelerator type (`cpu` or `gpu`).
        devices: Devices assigned.
        strategy: Lightning distribution strategy.
        log_every_n_steps: Training-step logging interval.
        resume_checkpoint_path: Checkpoint used to resume fitting.
        checkpoint: Checkpoint callback settings.
        early_stopping: Early-stopping callback settings.
        visualization: Visualization callback settings.
    """

    accelerator: str = "gpu"
    devices: Any = field(default_factory=lambda: [1])
    strategy: str = "ddp_find_unused_parameters_true"
    log_every_n_steps: int = 50
    resume_checkpoint_path: str | None = None
    
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


@dataclass
class LoggerConfig:
    """Configure the selected experiment logging backend.

    Args:
        name: Logger backend name.
        project_name: Remote or local project name.
        save_dir: Local logger output directory.
        log_hyperparams: Whether resolved hyperparameters are logged.
        log_model: Whether model checkpoints are uploaded.
        log_code: Whether source code is uploaded.
        log_graph: Whether the computation graph is logged.
        log_git_metadata: Whether Git metadata is logged.
        log_git_patch: Whether the working-tree patch is logged.
        default_hp_metric: TensorBoard default metric behavior.
        entity: Optional Weights & Biases entity.
        offline: Whether Weights & Biases operates offline.
    """
    
    name: str = "comet"  # comet, tensorboard, wandb, csv, none
    project_name: str = "priwiloc"
    save_dir: str = "./logs"
    log_hyperparams: bool = True
    
    # Comet-specific
    log_model: bool = True
    log_code: bool = False
    log_graph: bool = False
    log_git_metadata: bool = True
    log_git_patch: bool = False
    
    # TensorBoard-specific
    default_hp_metric: bool = True
    
    # WandB-specific
    entity: Optional[str] = None
    offline: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for experiment.

    Args:
        name: Base experiment name.
        seed: Global reproducibility seed.
        max_epochs: Maximum training epochs.
        learning_rate: Optimizer learning rate.
    """
    
    name: str = "priwiloc_experiment"
    seed: int = 42
    max_epochs: int = 100
    learning_rate: float = 5e-5
    

@dataclass
class AggregationConfig:
    """Configuration for Aggregation Strategy.

    Args:
        _target_: Fully qualified aggregation strategy class.
        buffer_policy: Policy for excluded model-state buffers.
        reference_client_id: Client supplying excluded buffer values.
        weight_source: Source of weighted-mean coefficients.
        client_weights: Fixed public client weights.
    """

    _target_: str = (
        "models.aggregation.strategies.UniformMeanAggregation"
    )
    buffer_policy: str = "reference_client"
    reference_client_id: int = 0
    weight_source: str = "sample_count"
    client_weights: list[Any] | None = None


@dataclass
class ParameterSelectorConfig:
    """Configuration for model parameters attacked.

    Args:
        mode: Parameter selection mode.
        value: Mode-specific key, pattern, fraction, or count.
    """

    mode: str = "all"
    value: Any = None


@dataclass
class AttackRuleConfig:
    """Configuration for Attack rule.

    Args:
        name: Stable attack name.
        _target_: Fully qualified attack implementation class.
        ap_selector: Access points attacked.
        rounds: Aggregation rounds attacked.
        parameter_selector: Parameters targeted by the attack.
        distribution: Generated-value distribution.
        magnitude: Noise or replacement magnitude.
        factor: Scaling factor.
        value: Replacement offset or constant.
        seed_offset: Rule-specific deterministic seed.
    """

    name: str = MISSING
    _target_: str = MISSING
    ap_selector: Any = "none"
    rounds: Any = "all"
    parameter_selector: ParameterSelectorConfig = field(
        default_factory=ParameterSelectorConfig
    )
    distribution: str | None = None
    magnitude: float | None = None
    factor: float | None = None
    value: float | None = None
    seed_offset: int = 0


@dataclass
class SecureAggServerConfig:
    """Configuration for ordered server attacks. An empty list represents no attack.

    Args:
        attacks: Attack rules applied in list order.
    """

    attacks: list[AttackRuleConfig] = field(default_factory=list)


@dataclass
class SecureCallbackConfig:
    """Configuration for plotting audits.

    Args:
        output_dir: Directory receiving audit chunks.
        plot_interval_epochs: Epochs between trust figure updates.
        export_interval_epochs: Epochs between audit exports.
        max_pending_events: Event count that triggers an early export.
    """

    output_dir: str = "secure_audit"
    plot_interval_epochs: int = 1
    export_interval_epochs: int = 1
    max_pending_events: int = 10_000


@dataclass
class SecureAggConfig:
    """Configuration for Secure Aggregation.

    Args:
        enabled: Set to true for secure aggregation.
        frac_bits: Fixed-point fractional precision.
        modulus_bits: Mersenne-prime field size in bits.
        k: Number of verification projections.
        master_seed: seed.
        server: Server attack settings.
        callback: Audit and plotting settings.
    """

    enabled: bool = False
    frac_bits: int = 20
    modulus_bits: int = 61
    k: int = 2
    master_seed: int = 1234
    server: SecureAggServerConfig = field(default_factory=SecureAggServerConfig)
    callback: SecureCallbackConfig = field(
        default_factory=SecureCallbackConfig
    )


@dataclass
class EvalCDFConfig:
    """Configuration per-dataset and overlaid empirical CDF plots.

    Args:
        enabled: Set True to generate CDF figures.
        overlay_all_datasets: Set True to generate an overall figure.
        max_points: Maximum plotted points per dataset.
    """

    enabled: bool = True
    overlay_all_datasets: bool = True
    max_points: int = 200_000


@dataclass
class EvalConfig:
    """Configuration for evaluation statistics.

    Args:
        percentiles: Location-error percentiles to report.
        save_raw_errors: Set True to save raw evaluation lists.
        raw_error_format: Raw-error output format (`csv` or `npy`).
        cdf: CDF plotting settings.
        output_dir: Directory receiving evaluation artifacts.
    """

    percentiles: list[int] = field(default_factory=lambda: [50, 75, 90, 95, 99])
    save_raw_errors: bool = True
    raw_error_format: str = "npy"
    cdf: EvalCDFConfig = field(default_factory=EvalCDFConfig)
    output_dir: str = "eval_outputs"

@dataclass
class Config:
    """All Configurations.

    Args:
        experiment: Run-level settings.
        dataset: Dataset and loader settings.
        model: Model settings.
        trainer: Lightning trainer settings.
        logger: Experiment logger settings.
        aggregation: Federated aggregation settings.
        secure_agg: Secure aggregation settings.
        eval: Evaluation reporting settings.
    """
    
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    secure_agg: SecureAggConfig = field(default_factory=SecureAggConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


