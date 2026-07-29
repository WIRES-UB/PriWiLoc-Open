"""Secure-aggregation training and multi-dataset evaluation entry point."""

from __future__ import annotations

import logging
from datetime import datetime

import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from callbacks.secure_aggregation_callback import SecureAggregationCallback
from callbacks.visualization_callback import AoAVisualizationCallback
from data_module import DLocDataModule
from dataset import DLocDatasetV2
from models.federated_learning_secagg import SecureFederatedLearningModel
from models.secure_agg.factory import (
    build_aggregation_strategy,
    validate_secure_config,
)
from utils.config_hydra import Config
from utils.feature_standardization import build_training_per_ap_standardizer
from utils.logger_factory import LoggerFactory


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config_secagg")
def main(cfg: DictConfig) -> None:
    """Run secure-aggregation training and isolated multi-dataset evaluation.

    Args:
        cfg: Hydra configuration for data, model, aggregation, logging, and
            training behavior.

    Returns:
        None.
    """

    validate_secure_config(cfg)
    cfg = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    aggregation_strategy = build_aggregation_strategy(cfg)
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.experiment.seed)

    device_name = f"gpu_{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    base_experiment_name = str(cfg.experiment.name)
    experiment_name = f"{timestamp}:{device_name}:{base_experiment_name}"
    # The model uses config.experiment.name when naming resolved configs,
    # summaries, predictions, targets, and raw-error artifacts. Store the
    # timestamped run name in the resolved config so all outputs share the
    # same unique identifier.
    cfg.experiment.name = experiment_name
    logger.info("Experiment name: %s", experiment_name)

    pl_logger = LoggerFactory.create_logger(cfg, experiment_name)
    if pl_logger is not None:
        logger.info("Using logger: %s", cfg.logger.name)
        LoggerFactory.log_hyperparameters(pl_logger, cfg)
        LoggerFactory.log_dataset_files(
            pl_logger,
            cfg.dataset.train_data_path,
            cfg.dataset.get("val_data_path"),
            cfg.dataset.get("test_data_path"),
        )
    else:
        logger.info("No logger configured")

    standardization_config = cfg.dataset.get("standardization", {})
    train_val_split_config = cfg.dataset.get("train_val_split")
    configured_train_val_split = (
        OmegaConf.to_container(
            train_val_split_config,
            resolve=True,
        )
        if train_val_split_config is not None
        else None
    )
    standardization_train_val_split = (
        None
        if cfg.dataset.get("val_data_path")
        else configured_train_val_split
    )
    feature_transform = None
    if bool(standardization_config.get("enabled", True)):
        logger.info("Fitting per-AP standardization on the effective training data...")
        feature_transform = build_training_per_ap_standardizer(
            data_paths=cfg.dataset.train_data_path,
            dataset_class=DLocDatasetV2,
            train_val_split=standardization_train_val_split,
            split_seed=cfg.dataset.split_seed,
            expected_n_aps=cfg.dataset.train_n_aps,
            eps=float(standardization_config.get("eps", 1e-6)),
        )
        logger.info(
            "Per-AP standardization fitted: mean=%s, std=%s",
            feature_transform.mean.tolist(),
            feature_transform.std.tolist(),
        )
        experiment = getattr(pl_logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "log_parameters"):
            experiment.log_parameters(
                {
                    **{
                        f"standardization_mean_ap_{ap_id}": float(value)
                        for ap_id, value in enumerate(feature_transform.mean)
                    },
                    **{
                        f"standardization_std_ap_{ap_id}": float(value)
                        for ap_id, value in enumerate(feature_transform.std)
                    },
                    "standardization_eps": float(feature_transform.eps),
                }
            )

    model_cls = SecureFederatedLearningModel
    model = model_cls(
        cfg,
        aggregation_strategy=aggregation_strategy,
    )
    logger.info("Model initialized: %s", cfg.model.name)

    data_module = DLocDataModule(
        train_data_paths=cfg.dataset.train_data_path,
        val_data_paths=cfg.dataset.get("val_data_path"),
        test_data_paths=cfg.dataset.get("test_data_path"),
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        transform=feature_transform,
        prefetch_factor=cfg.dataset.prefetch_factor,
        sequence_length=cfg.dataset.sequence_length,
        dataset_class=DLocDatasetV2,
        train_val_split=configured_train_val_split,
        split_seed=cfg.dataset.split_seed,
        shuffle_train=cfg.dataset.shuffle_train,
        eval_datasets=OmegaConf.to_container(
            cfg.dataset.eval_datasets,
            resolve=True,
        ),
    )
    logger.info("Data module created")

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.trainer.checkpoint.monitor,
        mode=cfg.trainer.checkpoint.mode,
        save_top_k=cfg.trainer.checkpoint.save_top_k,
        filename=cfg.trainer.checkpoint.filename,
    )
    early_stopping_callback = EarlyStopping(
        monitor=cfg.trainer.early_stopping.monitor,
        patience=cfg.trainer.early_stopping.patience,
        verbose=cfg.trainer.early_stopping.verbose,
        mode=cfg.trainer.early_stopping.mode,
    )
    visualization_callback = AoAVisualizationCallback(
        max_visualization_samples=cfg.trainer.visualization.max_visualization_samples,
        display_viz_data_epoch_interval=(
            cfg.trainer.visualization.display_viz_data_epoch_interval
        ),
    )
    secure_callback_config = cfg.secure_agg.callback
    secure_callback = SecureAggregationCallback(
        output_dir=secure_callback_config.output_dir,
        plot_interval_epochs=(
            secure_callback_config.plot_interval_epochs
        ),
        export_interval_epochs=(
            secure_callback_config.export_interval_epochs
        ),
        max_pending_events=secure_callback_config.max_pending_events,
    )
    callbacks = [
        checkpoint_callback,
        visualization_callback,
        secure_callback,
        early_stopping_callback,
    ]
    logger.info("Callbacks initialized")

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        logger=pl_logger,
        devices=cfg.trainer.devices,
        max_epochs=cfg.experiment.max_epochs,
        strategy=cfg.trainer.strategy,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )
    logger.info("Trainer initialized")

    logger.info("Starting training...")
    trainer.fit(
        model,
        data_module,
        ckpt_path=cfg.trainer.resume_checkpoint_path,
    )
    logger.info("Training completed")

    best_model_path = checkpoint_callback.best_model_path
    logger.info("Best model saved at: %s", best_model_path)

    if pl_logger is not None and cfg.logger.get("log_model", False):
        LoggerFactory.log_model_checkpoint(
            pl_logger,
            best_model_path,
            f"{cfg.experiment.name}_best_model",
        )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    best_model = model_cls.load_from_checkpoint(
        best_model_path,
        config=cfg,
        aggregation_strategy=build_aggregation_strategy(cfg),
    )
    logger.info("Best model loaded")

    logger.info("Starting testing...")
    trainer.test(best_model, datamodule=data_module)
    logger.info("Testing completed")
    logger.info("All done!")


if __name__ == "__main__":
    main()
