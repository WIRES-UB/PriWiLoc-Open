"""Factory for creating different types of loggers for PyTorch Lightning."""

import os
from typing import Optional, Union
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import (
    CometLogger,
    TensorBoardLogger,
    CSVLogger,
    WandbLogger,
)


class LoggerFactory:
    """Factory class for creating PyTorch Lightning loggers based on configuration."""

    @staticmethod
    def create_logger(cfg: DictConfig, experiment_name: str) -> Optional[pl.loggers.Logger]:
        """Create a logger based on the configuration.

        Args:
            cfg: Hydra configuration object containing logger settings.
            experiment_name: Name of the experiment.

        Returns:
            A PyTorch Lightning logger instance, or None if logger is disabled.

        Raises:
            ValueError: If an unsupported logger type is specified.
        """
        logger_name = cfg.logger.name.lower()

        if logger_name == "none":
            return None

        elif logger_name == "comet":
            return LoggerFactory._create_comet_logger(cfg, experiment_name)

        elif logger_name == "tensorboard":
            return LoggerFactory._create_tensorboard_logger(cfg, experiment_name)

        elif logger_name == "wandb":
            return LoggerFactory._create_wandb_logger(cfg, experiment_name)

        elif logger_name == "csv":
            return LoggerFactory._create_csv_logger(cfg, experiment_name)

        else:
            raise ValueError(
                f"Unsupported logger type: {logger_name}. "
                f"Supported types: comet, tensorboard, wandb, csv, none"
            )

    @staticmethod
    def _create_comet_logger(cfg: DictConfig, experiment_name: str) -> CometLogger:
        """Create a Comet logger.

        Args:
            cfg: Hydra configuration object.
            experiment_name: Name of the experiment.

        Returns:
            A CometLogger instance.

        Raises:
            ValueError: If required environment variables are not set.
        """
        api_key = os.getenv("COMET_API_KEY")
        workspace = os.getenv("COMET_WORKSPACE")

        if not api_key:
            raise ValueError(
                "COMET_API_KEY environment variable is not set. "
                "Please set it with: export COMET_API_KEY=<your_api_key>"
            )

        if not workspace:
            raise ValueError(
                "COMET_WORKSPACE environment variable is not set. "
                "Please set it with: export COMET_WORKSPACE=<your_workspace>"
            )

        return CometLogger(
            api_key=api_key,
            workspace=workspace,
            save_dir=cfg.logger.save_dir,
            project_name=cfg.logger.project_name,
            experiment_name=experiment_name,
        )

    @staticmethod
    def _create_tensorboard_logger(cfg: DictConfig, experiment_name: str) -> TensorBoardLogger:
        """Create a TensorBoard logger.

        Args:
            cfg: Hydra configuration object.
            experiment_name: Name of the experiment.

        Returns:
            A TensorBoardLogger instance.
        """
        return TensorBoardLogger(
            save_dir=cfg.logger.save_dir,
            name=experiment_name,
            default_hp_metric=cfg.logger.get("default_hp_metric", True),
        )

    @staticmethod
    def _create_wandb_logger(cfg: DictConfig, experiment_name: str) -> WandbLogger:
        """Create a Weights & Biases logger.

        Args:
            cfg: Hydra configuration object.
            experiment_name: Name of the experiment.

        Returns:
            A WandbLogger instance.
        """
        return WandbLogger(
            name=experiment_name,
            save_dir=cfg.logger.save_dir,
            project=cfg.logger.project_name,
            entity=cfg.logger.get("entity", None),
            offline=cfg.logger.get("offline", False),
            log_model=cfg.logger.get("log_model", False),
        )

    @staticmethod
    def _create_csv_logger(cfg: DictConfig, experiment_name: str) -> CSVLogger:
        """Create a CSV logger.

        Args:
            cfg: Hydra configuration object.
            experiment_name: Name of the experiment.

        Returns:
            A CSVLogger instance.
        """
        return CSVLogger(
            save_dir=cfg.logger.save_dir,
            name=experiment_name,
        )

    @staticmethod
    def log_hyperparameters(logger: Optional[pl.loggers.Logger], cfg: DictConfig) -> None:
        """Log hyperparameters to the logger.

        Args:
            logger: PyTorch Lightning logger instance.
            cfg: Hydra configuration object containing hyperparameters.
        """
        if logger is None:
            return

        if not cfg.logger.get("log_hyperparams", True):
            return

        # Convert OmegaConf to dict for logging
        from omegaconf import OmegaConf
        hparams = OmegaConf.to_container(cfg, resolve=True)

        # Remove logger config from hyperparameters to avoid logging credentials
        if isinstance(hparams, dict) and "logger" in hparams:
            hparams.pop("logger")

        logger.log_hyperparams(hparams)

    @staticmethod
    def log_dataset_files(
        logger: Optional[pl.loggers.Logger],
        train_path: str,
        val_path: str,
        test_path: str,
    ) -> None:
        """Log dataset CSV files to the logger (if supported).

        Args:
            logger: PyTorch Lightning logger instance.
            train_path: Path to training data CSV.
            val_path: Path to validation data CSV.
            test_path: Path to test data CSV.
        """
        if logger is None:
            return

        # Only Comet and WandB support file logging
        if isinstance(logger, CometLogger):
            logger.experiment.log_asset(train_path, metadata={"stage": "train"})
            logger.experiment.log_asset(val_path, metadata={"stage": "val"})
            logger.experiment.log_asset(test_path, metadata={"stage": "test"})

        elif isinstance(logger, WandbLogger):
            import wandb
            logger.experiment.save(train_path)
            logger.experiment.save(val_path)
            logger.experiment.save(test_path)

    @staticmethod
    def log_model_checkpoint(
        logger: Optional[pl.loggers.Logger],
        checkpoint_path: str,
        model_name: str,
    ) -> None:
        """Log model checkpoint to the logger (if supported).

        Args:
            logger: PyTorch Lightning logger instance.
            checkpoint_path: Path to the model checkpoint.
            model_name: Name for the model.
        """
        if logger is None:
            return

        # Only Comet and WandB support model logging
        if isinstance(logger, CometLogger):
            if hasattr(logger, "experiment"):
                logger.experiment.log_model(
                    name=model_name,
                    file_or_folder=checkpoint_path,
                )

        elif isinstance(logger, WandbLogger):
            import wandb
            artifact = wandb.Artifact(model_name, type="model")
            artifact.add_file(checkpoint_path)
            logger.experiment.log_artifact(artifact)

    @staticmethod
    def get_experiment_name(logger: Optional[pl.loggers.Logger]) -> Optional[str]:
        """Get the experiment name from the logger.

        Args:
            logger: PyTorch Lightning logger instance.

        Returns:
            The experiment name, or None if logger is None.
        """
        if logger is None:
            return None

        if isinstance(logger, CometLogger):
            return logger.experiment.name
        elif isinstance(logger, (TensorBoardLogger, CSVLogger)):
            return logger.name
        elif isinstance(logger, WandbLogger):
            return logger.experiment.name
        else:
            return None

