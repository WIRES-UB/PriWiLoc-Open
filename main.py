"""Script for training, validation, and testing of the model using Hydra configuration."""

import logging
from datetime import datetime

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from callbacks.visualization_callback import AoAVisualizationCallback
from data_module import DLocDataModule
from dataset import DLocDatasetV2
from models.federated_learning import FederatedLearningModel
from utils.logger_factory import LoggerFactory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration.
    
    Args:
        cfg: Hydra configuration object loaded from YAML files.
    """
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
    pl.seed_everything(cfg.experiment.seed)
    
    # Create experiment name with timestamp
    device_name = f"gpu_{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    experiment_name = f"{timestamp}:{device_name}:{cfg.experiment.name}"
    logger.info(f"Experiment name: {experiment_name}")
    
    # Create logger
    pl_logger = LoggerFactory.create_logger(cfg, experiment_name)
    
    if pl_logger is not None:
        logger.info(f"Using logger: {cfg.logger.name}")
        # Log hyperparameters
        LoggerFactory.log_hyperparameters(pl_logger, cfg)
        # Log dataset files
        LoggerFactory.log_dataset_files(
            pl_logger,
            cfg.dataset.train_data_path,
            cfg.dataset.val_data_path,
            cfg.dataset.test_data_path,
        )
    else:
        logger.info("No logger configured")
    
    # Initialize model
    model_cls = FederatedLearningModel
    model = model_cls(cfg)  # Pass the full root config
    logger.info(f"Model initialized: {cfg.model.name}")
    
    # Create Data Module
    data_module = DLocDataModule(
        train_data_paths=cfg.dataset.train_data_path,
        val_data_paths=cfg.dataset.val_data_path,
        test_data_paths=cfg.dataset.test_data_path,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        transform=None,
        prefetch_factor=cfg.dataset.prefetch_factor,
        dataset_class=DLocDatasetV2,
    )
    logger.info("Data module created")
    
    # Create callbacks
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
        display_viz_data_epoch_interval=cfg.trainer.visualization.display_viz_data_epoch_interval,
    )
    
    callbacks = [checkpoint_callback, visualization_callback, early_stopping_callback]
    logger.info("Callbacks initialized")
    
    # Create Trainer
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
    
    # Training
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    logger.info("Training completed")
    
    # Load Best Model for Testing
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"Best model saved at: {best_model_path}")
    
    # Log the checkpoint to logger (if supported)
    if pl_logger is not None and cfg.logger.get("log_model", False):
        LoggerFactory.log_model_checkpoint(
            pl_logger,
            best_model_path,
            f"{cfg.experiment.name}_best_model",
        )
    
    # Wait for all ranks in distributed training
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Load best model
    best_model = model_cls.load_from_checkpoint(
        best_model_path,
        config=cfg,  # pass full config here
    )
    logger.info("Best model loaded")
    
    # Testing
    logger.info("Starting testing...")
    trainer.test(best_model, datamodule=data_module)
    logger.info("Testing completed")
    
    logger.info("All done!")


if __name__ == "__main__":
    main()
