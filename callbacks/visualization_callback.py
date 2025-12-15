"""Callback for visualizing AoA plots and metrics."""

from collections import deque
from typing import Any, Mapping, Optional
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger, WandbLogger
import numpy as np

from metrics_calculator import MetricNames
from utils.plot_utils import (
    plot_aoa_error_histograms,
    plot_aoa_visualization,
    plot_gt_vs_pred_aoa,
    plot_location_pred_vs_gt,
    plot_location_error_cdf,
)
from utils.schema import AoAVisualizationSample


class AoAVisualizationCallback(Callback):
    def __init__(
        self,
        display_viz_data_epoch_interval: int = 5,
        max_visualization_samples: int = 5,
    ):
        """
        Callback for AoA visualization during validation.

        Args:
            ap_metadata: Metadata containing AP locations and orientations.
            display_viz_data_epoch_interval: Interval (in epochs) to display visualization data.
            max_visualization_samples: Maximum number of samples to visualize per epoch.
        """
        super().__init__()
        self.display_viz_data_epoch_interval = display_viz_data_epoch_interval
        self.val_visualization_data = deque(maxlen=max_visualization_samples)
        self.max_visualization_samples = max_visualization_samples

    @staticmethod
    def _log_figure(logger: Optional[pl.loggers.Logger], figure_name: str, figure) -> None:
        """Log a figure to the logger (works with multiple logger types).
        
        Args:
            logger: PyTorch Lightning logger instance.
            figure_name: Name for the figure.
            figure: Matplotlib figure object.
        """
        if logger is None:
            return
        
        if isinstance(logger, CometLogger):
            logger.experiment.log_figure(figure_name=figure_name, figure=figure)
        elif isinstance(logger, TensorBoardLogger):
            logger.experiment.add_figure(figure_name, figure)
        elif isinstance(logger, WandbLogger):
            import wandb
            logger.experiment.log({figure_name: wandb.Image(figure)})
        # CSV logger doesn't support figures, skip silently
    
    @staticmethod
    def _log_asset(logger: Optional[pl.loggers.Logger], file_path: str, overwrite: bool = True) -> None:
        """Log an asset file to the logger (if supported).
        
        Args:
            logger: PyTorch Lightning logger instance.
            file_path: Path to the file to log.
            overwrite: Whether to overwrite existing file.
        """
        if logger is None:
            return
        
        if isinstance(logger, CometLogger):
            logger.experiment.log_asset(file_path, overwrite=overwrite)
        elif isinstance(logger, WandbLogger):
            logger.experiment.save(file_path)
        # Other loggers don't support asset logging
    
    @staticmethod
    def _get_experiment_name(logger: Optional[pl.loggers.Logger]) -> str:
        """Get experiment name from logger.
        
        Args:
            logger: PyTorch Lightning logger instance.
            
        Returns:
            Experiment name or 'experiment' if logger is None.
        """
        if logger is None:
            return "experiment"
        
        if isinstance(logger, CometLogger):
            return logger.experiment.name
        elif isinstance(logger, (TensorBoardLogger, pl.loggers.CSVLogger)):
            return logger.name
        elif isinstance(logger, WandbLogger):
            return logger.experiment.name
        else:
            return "experiment"

    @staticmethod
    def _plot_metric_visualization(trainer: Trainer, pl_module: pl.LightningModule, stage: str, save_metrics: bool=False) -> None:
        """Plot metric visualizations.

        Args:
            trainer: PyTorch Lightning Trainer instance.
            pl_module: The LightningModule being trained.
            metric: The metric to visualize, typically from pl_module.val_metrics or pl_module.test_metrics.
            stage: "val" or "test" to indicate the phase of the metrics.
            save_metrics: Whether to save metrics to a .npy file.
        """
        if stage == "train":
            metric_result = pl_module.train_metrics.compute()
        elif stage == "val":
            metric_result = pl_module.val_metrics.compute()
        elif stage == "test":
            metric_result = pl_module.test_metrics.compute()
        else:
            raise ValueError("Invalid stage provided. Use 'train', 'val' or 'test'.")

        # Plot predicted vs ground truth location
        loc_plot = plot_location_pred_vs_gt(
            metric_result[MetricNames.LOCATION_PREDS].detach().cpu(),
            metric_result[MetricNames.LOCATION_TARGETS].detach().cpu(),
            ap_locations=pl_module.ap_metadata.ap_locations,
            ap_orientations=pl_module.ap_metadata.ap_orientations,
        )
        AoAVisualizationCallback._log_figure(trainer.logger, f"{stage}_location_pred_vs_gt", loc_plot)

        # Log AoA error distribution for each AP
        aoa_error_dist_plot = plot_aoa_error_histograms(
            metric_result[MetricNames.AOA_ERROR_ALL_RADIAN].detach().cpu().numpy(),
            metric_result[MetricNames.AOA_ERROR_MEAN_RADIAN].detach().cpu().numpy(),
            metric_result[MetricNames.AOA_ERROR_STD_RADIAN].detach().cpu().numpy(),
            plot_in_degrees=True,
        )
        AoAVisualizationCallback._log_figure(trainer.logger, f"{stage}_aoa_error_distribution", aoa_error_dist_plot)

        # Plot AoA prediction vs ground truth
        aoa_gt_vs_pred_plot = plot_gt_vs_pred_aoa(
            metric_result[MetricNames.AOA_PREDS_RADIAN].detach().cpu().numpy(),
            metric_result[MetricNames.AOA_TARGETS_RADIAN].detach().cpu().numpy(),
            delta=10,
            plot_in_degrees=True,
        )
        AoAVisualizationCallback._log_figure(trainer.logger, f"{stage}_aoa_pred_vs_gt", aoa_gt_vs_pred_plot)

        # log AoA error metrics in degree for each AP
        for ap_index in range(pl_module.ap_metadata.n_aps):
            pl_module.log_dict(
                {
                    f"{stage}_aoa_error_mean_ap{ap_index}_deg": torch.rad2deg(
                        metric_result[MetricNames.AOA_ERROR_MEAN_RADIAN][ap_index]
                    ).item(),
                    f"{stage}_aoa_error_std_ap{ap_index}_deg": torch.rad2deg(
                        metric_result[MetricNames.AOA_ERROR_STD_RADIAN][ap_index]
                    ).item(),
                    f"{stage}_aoa_error_rmse_ap{ap_index}_deg": torch.rad2deg(
                        metric_result[MetricNames.AOA_ERROR_RMSE_RADIAN][ap_index]
                    ).item(),
                }
            )

        # log location error metrics
        pl_module.log_dict(
            {
                f"{stage}_{name}": value
                for name, value in metric_result.items()
                if name.startswith("location_error") and "all" not in name
            }
        )

        # log location error cdf
        location_error_cdf_plot = plot_location_error_cdf(
            metric_result[MetricNames.LOCATION_ERROR_ALL].detach().cpu().numpy(),
        )
        AoAVisualizationCallback._log_figure(trainer.logger, f"{stage}_location_error_cdf", location_error_cdf_plot)

        # save metrics to npy file
        if save_metrics:
            metrics_dict = {
                key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                for key, value in metric_result.items()
            }
            experiment_name = AoAVisualizationCallback._get_experiment_name(trainer.logger)
            file_name = f"{experiment_name}_{stage}_metrics.npy"
            save_dir = trainer.logger.save_dir if trainer.logger else "./logs"
            save_path = os.path.join(save_dir, file_name)
            print(f"Saving {stage} metrics to {save_path}")
            np.save(save_path, metrics_dict)
            AoAVisualizationCallback._log_asset(trainer.logger, save_path, overwrite=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: Mapping[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Store visualization data for AoA-TOF plots at the end of each validation batch.

        Args:
            trainer: PyTorch Lightning Trainer instance.
            pl_module: The LightningModule being trained.
            outputs: Outputs from the validation step.
            batch: The current batch of data.
            batch_idx: The index of the current batch.
            dataloader_idx: Index of the dataloader (default is 0).
        """
        if len(self.val_visualization_data) < self.max_visualization_samples:
            model_pred = outputs["model_pred"]
            gt_label = outputs["gt_label"]
            if batch.features_2d.dim() == 5:
                self.val_visualization_data.append(
                    AoAVisualizationSample(
                        aoa_tof_plots=batch.features_2d[0, -1].detach().cpu(),
                        aoa_pred=torch.rad2deg(model_pred.aoa[0].detach().cpu()),
                        aoa_gt=torch.rad2deg(gt_label.aoa[0].detach().cpu()),
                        confidence=model_pred.confidence[0, -1].detach().cpu(),
                    )
                )
            else:
                self.val_visualization_data.append(
                    AoAVisualizationSample(
                        aoa_tof_plots=batch.features_2d[0].detach().cpu(),
                        aoa_pred=torch.rad2deg(model_pred.aoa[0].detach().cpu()),
                        aoa_gt=torch.rad2deg(gt_label.aoa[0].detach().cpu()),
                        confidence=model_pred.confidence[0].detach().cpu(),
                    )
                )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        """
        Log AoA visualizations at the end of the validation epoch.

        Args:
            trainer: PyTorch Lightning Trainer instance.
            pl_module: The LightningModule being trained.
        """
        # Log visualizations every `display_viz_data_epoch_interval` epochs
        if trainer.current_epoch % self.display_viz_data_epoch_interval == 0:
            # Plot metric visualizations
            self._plot_metric_visualization(trainer, pl_module, "val")

            # Log AoA-TOF visualizations
            sample_index = 0
            while self.val_visualization_data:
                sample_viz = self.val_visualization_data.popleft()
                aoa_plot = plot_aoa_visualization(
                    sample_viz,
                    title=f"AoA Visualization, sample index {sample_index}",
                    degree=True,
                )
                self._log_figure(trainer.logger, f"val_aoa_viz_sample{sample_index}", aoa_plot)
                sample_index += 1

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        # Log visualizations every `display_viz_data_epoch_interval` epochs
        if trainer.current_epoch % self.display_viz_data_epoch_interval == 0:
            self._plot_metric_visualization(trainer, pl_module, "train")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self._plot_metric_visualization(trainer, pl_module, "test", save_metrics=True)
