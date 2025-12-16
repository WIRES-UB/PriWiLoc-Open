"""Hydra-compatible configuration classes for PriWiLoc experiments.

This module provides configuration classes that work with Hydra for configuration management.
The original config.py is kept for backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING

@dataclass
class DatasetConfig:
    """Configuration for the dataset."""
    
    batch_size: int = 16
    num_workers: int = 8
    prefetch_factor: int = 2
    sequence_length: int = 20
    
    train_data_path: str = MISSING  # Required field
    val_data_path: str = MISSING  # Required field
    test_data_path: str = MISSING  # Required field
    
    train_n_aps: int = 4
    val_n_aps: int = 4
    test_n_aps: int = 4


@dataclass
class ModelConfig:
    """Configuration for the model."""
    
    name: str = "FederatedLearningModel"
    dropout: float = 0.3
    in_channels: int = 1
    average_weight_every_n_batches: int = 10


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing."""
    
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 1
    filename: str = "{epoch:02d}"


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    
    monitor: str = "val_loss"
    patience: int = 5
    mode: str = "min"
    verbose: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for visualization callback."""
    
    max_visualization_samples: int = 10
    display_viz_data_epoch_interval: int = 1


@dataclass
class TrainerConfig:
    """Configuration for PyTorch Lightning Trainer."""
    
    accelerator: str = "gpu"
    devices: list = field(default_factory=lambda: [1])
    strategy: str = "ddp_find_unused_parameters_true"
    log_every_n_steps: int = 50
    
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


@dataclass
class LoggerConfig:
    """Configuration for logger."""
    
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
    """Main experiment configuration."""
    
    name: str = "priwiloc_experiment"
    seed: int = 42
    max_epochs: int = 100
    learning_rate: float = 5e-5
    

@dataclass
class Config:
    """Root configuration object."""
    
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)


