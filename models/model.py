"""Model definition and training and validation logic."""

from pathlib import Path
from typing import Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torchmetrics import MetricCollection
from torchvision.models import resnet34

from metrics_calculator import AoAAccuracy, LocationAccuracy, MetricNames, RSSIMetric
from models.model_utils import compute_geometric_loss, get_batch_gt_label
from utils.geometry_utils import cos_angle_diff, sin_angle_diff
from utils.ray_intersection_solver import solve_ray_intersection_batch
from utils.schema import (
    APMetadata,
    DLocBatchDataSample,
    GTlabel,
    LossTerms,
    ModelOutput,    
)
from utils.config_hydra import Config 
from utils.eval_reporting import (
    create_cdf_figure,
    safe_percentiles,
    sanitize_name,
    save_dataset_assets,
    save_summary_csv,
)
from utils.experiment_logging import log_asset, log_figure_and_close


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    """Read a value from mapping-style or attribute-style configuration.

    Args:
        config: Configuration object or mapping.
        key: Value name to read.
        default: Fallback when the value is unavailable.

    Returns:
        The configured value or `default`.
    """
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


# -----------------------------
# Helper modules
# -----------------------------

class QuadraticActivation(nn.Module):
    """Apply the bounded quadratic activation used by the localization model."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quadratic activation elementwise.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor with the same shape.
        """
        return torch.relu((-4 / torch.pi ** 2) * x ** 2 + 1)


class ResNetEncoder(nn.Module):
    """Encode one AP feature map with ResNet-34 and an MLP projection.

    Args:
        in_channels: Number of input feature channels.
        dropout: MLP dropout probability.
    """

    def __init__(self, in_channels: int, dropout: float):
        """Initialize the ResNet backbone and projection MLP.

        Args:
            in_channels: Number of input feature channels.
            dropout: MLP dropout probability.

        Returns:
            None.
        """
        super().__init__()

        self.resnet = resnet34(weights=None)
        self.output_dim = self.resnet.fc.in_features

        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Identity()

        self.mlp_output_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(self.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.mlp_output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of AP feature maps.

        Args:
            x: Feature maps shaped for the ResNet input.

        Returns:
            Projected feature embeddings.
        """
        return self.mlp(self.resnet(x))


class AoADecoder(nn.Module):
    """Decode encoder features into AoA and confidence predictions.

    Args:
        in_features: Encoder embedding dimension.
    """

    def __init__(self, in_features: int):
        """Initialize AoA and confidence prediction heads.

        Args:
            in_features: Encoder embedding dimension.

        Returns:
            None.
        """
        super().__init__()

        self.aoa_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Decode feature embeddings.

        Args:
            x: Encoder feature embeddings.

        Returns:
            AoA and confidence predictions.
        """
        return ModelOutput(
            aoa=self.aoa_head(x),
            confidence=self.confidence_head(x),
            cos_aoa=None,
            sin_aoa=None,
            location=None,
        )


# -----------------------------
# Lightning Model
# -----------------------------

class TrigAOAResNetModel(pl.LightningModule):
    """Train and evaluate the multi-AP trigonometric localization model.

    Args:
        config: Root experiment configuration.
    """

    def __init__(self, config: Config):
        """Initialize metrics, AP encoders, and the shared decoder.

        Args:
            config: Root experiment configuration.

        Returns:
            None.
        """
        super().__init__()
        self.config = config

        self.lr = config.experiment.learning_rate

        # Metrics
        self.train_metrics = MetricCollection([
            AoAAccuracy(n_aps=config.dataset.train_n_aps),
            LocationAccuracy(),
            RSSIMetric(),
        ])
        self.val_metrics = MetricCollection([
            AoAAccuracy(n_aps=config.dataset.val_n_aps),
            LocationAccuracy(),
            RSSIMetric(),
        ])
        eval_datasets = list(_config_value(config.dataset, "eval_datasets", []) or [])
        if eval_datasets:
            self.eval_dataset_specs = [
                (
                    str(_config_value(dataset, "name", f"dataset_{index}")),
                    int(_config_value(dataset, "n_aps", config.dataset.test_n_aps)),
                )
                for index, dataset in enumerate(eval_datasets)
            ]
            self.test_metrics_per_ds = nn.ModuleList([
                MetricCollection([
                    AoAAccuracy(n_aps=n_aps),
                    LocationAccuracy(),
                    RSSIMetric(),
                ])
                for _, n_aps in self.eval_dataset_specs
            ])
        else:
            # Preserve the original single-loader metric path for main.py.
            self.test_metrics = MetricCollection([
                AoAAccuracy(n_aps=config.dataset.test_n_aps),
                LocationAccuracy(),
                RSSIMetric(),
            ])

        # Encoders (one per AP)
        self.resnet_encoder_list= nn.ModuleList([
            ResNetEncoder(
                in_channels=config.model.in_channels,
                dropout=config.model.dropout,
            )
            for _ in range(config.dataset.train_n_aps)
        ])

        self.decoder = AoADecoder(self.resnet_encoder_list[0].mlp_output_dim)

    def forward(self, x: torch.Tensor, ap_metadata: torch.Tensor) -> ModelOutput:
        """Predict per-AP AoA values and the resulting device location.

        Args:
            x: Batched AP feature maps.
            ap_metadata: Batched AP locations and orientations.

        Returns:
            AoA, confidence, trigonometric, and location predictions.
        """
        ap_meta_list = APMetadata.from_tensor(ap_metadata)
        B, N = x.shape[:2]

        aoa_preds = torch.zeros(B, N, device=self.device)
        conf_preds = torch.zeros(B, N, device=self.device)

        for i in range(N):
            enc_idx = min(i, len(self.resnet_encoder_list) - 1)
            feat = self.resnet_encoder_list[enc_idx](x[:, i].unsqueeze(1))
            out = self.decoder(feat)

            aoa_preds[:, i] = out.aoa.squeeze(1) * torch.pi / 2
            conf_preds[:, i] = out.confidence.squeeze(1)

        cos_aoa = torch.cos(aoa_preds)
        sin_aoa = torch.sin(aoa_preds)

        cos_map = torch.zeros_like(cos_aoa)
        sin_map = torch.zeros_like(sin_aoa)

        for b in range(B):
            meta = ap_meta_list[b]
            cos_map[b] = cos_angle_diff(
                cos_aoa[b], sin_aoa[b],
                meta.cos_ap_orientations,
                meta.sin_ap_orientations,
            )
            sin_map[b] = sin_angle_diff(
                cos_aoa[b], sin_aoa[b],
                meta.cos_ap_orientations,
                meta.sin_ap_orientations,
            )

        location = torch.zeros(B, 2, device=self.device)
        conf = torch.ones_like(conf_preds)

        for b in range(B):
            meta = ap_meta_list[b]
            location[b] = solve_ray_intersection_batch(
                meta.ap_locations,
                cos_map[b].unsqueeze(0),
                sin_map[b].unsqueeze(0),
                conf[b].unsqueeze(0),
            ).squeeze(0)

        return ModelOutput(
            aoa=aoa_preds,
            confidence=conf_preds,
            cos_aoa=cos_aoa,
            sin_aoa=sin_aoa,
            location=location,
        )

    def _common_step(self, batch: DLocBatchDataSample, stage: str) -> Tuple[ModelOutput, GTlabel, LossTerms]:
        """Run shared forward, label, loss, and logging work for one batch.

        Args:
            batch: Collated localization batch.
            stage: `train`, `val`, or `test`.

        Returns:
            Model predictions, ground-truth labels, and loss terms.
        """
        assert stage in ["train", "val", "test"], "Stage must be one of 'train', 'val', or 'test'."

        # Forward pass
        model_pred: ModelOutput = self.forward(batch.features_2d, batch.ap_metadata)
    
        # Store AP metadata for visualization callback (use first sample's metadata)
        ap_metadata_list = APMetadata.from_tensor(batch.ap_metadata)
        self.ap_metadata = ap_metadata_list[0]

        # construct ground truth label
        gt_label: GTlabel = get_batch_gt_label(batch)

        # Compute geometric loss (PriWiLoc paper: cos + sin + location)
        loss_all: LossTerms = compute_geometric_loss(model_pred, gt_label)

        # log the loss
        self.log(f"{stage}_loss", loss_all.total_loss.item(), sync_dist=True)
        
        # Log individual loss components
        if loss_all.cos_loss is not None:
            self.log(f"{stage}_cos_loss", loss_all.cos_loss.item(), sync_dist=True)
        if loss_all.sin_loss is not None:
            self.log(f"{stage}_sin_loss", loss_all.sin_loss.item(), sync_dist=True)
        if loss_all.location_loss is not None:
            self.log(f"{stage}_location_loss", loss_all.location_loss.item(), sync_dist=True)

        return model_pred, gt_label, loss_all

    def training_step(self, batch, batch_idx):
        """Run one training batch and update training metrics.

        Args:
            batch: Collated training batch.
            batch_idx: Zero-based batch index.

        Returns:
            Total differentiable training loss.
        """
        pred, gt, loss = self._common_step(batch, "train")
        self.train_metrics.update(pred, gt)
        self.train_metrics["RSSIMetric"].set_rssi(batch.rssi)
        return loss.total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Run one validation batch and update validation metrics.

        Args:
            batch: Collated validation batch.
            batch_idx: Zero-based batch index.
            dataloader_idx: Validation loader index.

        Returns:
            Validation loss, predictions, and ground-truth labels.
        """
        pred, gt, loss = self._common_step(batch, "val")
        self.val_metrics.update(pred, gt)
        self.val_metrics["RSSIMetric"].set_rssi(batch.rssi)

        return {
            "val_loss": loss.total_loss,
            "model_pred": pred,
            "gt_label": gt,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Run one test batch with isolated per-dataset metrics.

        Args:
            batch: Collated test batch.
            batch_idx: Zero-based batch index.
            dataloader_idx: Test loader index.

        Returns:
            Test loss, predictions, and ground-truth labels.
        """
        if not hasattr(self, "test_metrics_per_ds"):
            pred, gt, loss = self._common_step(batch, "test")
            self.test_metrics.update(pred, gt)
            self.test_metrics["RSSIMetric"].set_rssi(batch.rssi)
            return {
                "test_loss": loss.total_loss,
                "model_pred": pred,
                "gt_label": gt,
            }

        if dataloader_idx >= len(self.test_metrics_per_ds):
            raise IndexError(
                f"Received test dataloader_idx={dataloader_idx}, but only "
                f"{len(self.test_metrics_per_ds)} evaluation metric collections exist."
            )
        pred, gt, loss = self._common_step(batch, "test")
        metrics = self.test_metrics_per_ds[dataloader_idx]
        metrics.update(pred, gt)
        metrics["RSSIMetric"].set_rssi(batch.rssi)
        dataset_name, _ = self.eval_dataset_specs[dataloader_idx]
        self.log(
            f"test/{dataset_name}/loss",
            loss.total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            add_dataloader_idx=False,
            batch_size=batch.features_2d.shape[0],
        )

        return {
            "test_loss": loss.total_loss,
            "model_pred": pred,
            "gt_label": gt,
        }

    def on_train_epoch_end(self):
        """Reset accumulated training metrics after each epoch.

        Returns:
            None.
        """
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        """Reset accumulated validation metrics after each epoch.

        Returns:
            None.
        """
        self.val_metrics.reset()

    def on_fit_start(self) -> None:
        """Record run discipline, split identity, and resolved configuration.

        Returns:
            None.
        """
        secure_config = _config_value(self.config, "secure_agg")
        eval_config = _config_value(self.config, "eval")
        if secure_config is None and eval_config is None:
            return

        secure_enabled = bool(_config_value(secure_config, "enabled", False))
        server_config = _config_value(secure_config, "server")
        attack_count = len(_config_value(server_config, "attacks", []))
        server_honest = attack_count == 0
        run_mode = "secure" if secure_enabled else "plain"
        shuffle_train = bool(_config_value(self.config.dataset, "shuffle_train", False))

        experiment = getattr(self.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "log_other"):
            experiment.log_other("run_mode", run_mode)
            experiment.log_other("server_honest", server_honest if secure_enabled else "n/a")
            experiment.log_other(
                "server_attack_rule_count",
                attack_count if secure_enabled else "n/a",
            )
            experiment.log_other("shuffle_train", shuffle_train)
            datamodule = getattr(self.trainer, "datamodule", None)
            split_hash = getattr(datamodule, "split_index_hash", None)
            if split_hash:
                experiment.log_other("split_index_hash", split_hash)

        output_dir = Path(str(_config_value(eval_config, "output_dir", "eval_outputs")))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = str(self.config.experiment.name)
        config_path = output_dir / f"resolved_config_{sanitize_name(run_name)}.yaml"
        config_path.write_text(OmegaConf.to_yaml(self.config, resolve=True), encoding="utf-8")
        log_asset(
            self.logger,
            config_path,
            metadata={"run_mode": run_mode, "kind": "resolved_config"},
        )

    def on_test_epoch_end(self):
        """Report isolated test metrics and save reproducible artifacts.

        Returns:
            None.
        """
        if not hasattr(self, "test_metrics_per_ds"):
            self.test_metrics.reset()
            return

        eval_config = _config_value(self.config, "eval")
        percentiles = list(_config_value(eval_config, "percentiles", [50, 75, 90, 95, 99]))
        output_dir = str(_config_value(eval_config, "output_dir", "eval_outputs"))
        save_raw_errors = bool(_config_value(eval_config, "save_raw_errors", True))
        raw_error_format = str(_config_value(eval_config, "raw_error_format", "npy"))
        cdf_config = _config_value(eval_config, "cdf")
        cdf_enabled = bool(_config_value(cdf_config, "enabled", True))
        overlay_enabled = bool(_config_value(cdf_config, "overlay_all_datasets", True))
        max_points = int(_config_value(cdf_config, "max_points", 20_000))
        run_name = str(self.config.experiment.name)
        secure_config = _config_value(self.config, "secure_agg")
        run_mode = (
            "secure"
            if bool(_config_value(secure_config, "enabled", False))
            else "plain"
        )

        errors_for_overlay: dict[str, Any] = {}
        summary_rows: list[dict[str, object]] = []
        observed_total = 0
        expected_total = 0

        test_loaders = getattr(self.trainer, "test_dataloaders", None)
        if test_loaders is not None and not isinstance(test_loaders, (list, tuple)):
            test_loaders = [test_loaders]
        if test_loaders is not None and len(test_loaders) != len(self.eval_dataset_specs):
            raise AssertionError(
                "Evaluation isolation requires one metric collection per loader: "
                f"got {len(test_loaders)} loaders and "
                f"{len(self.eval_dataset_specs)} configured datasets."
            )

        try:
            for index, ((name, _), metrics) in enumerate(
                zip(self.eval_dataset_specs, self.test_metrics_per_ds)
            ):
                result = metrics.compute()
                errors = result[MetricNames.LOCATION_ERROR_ALL]
                predictions = result[MetricNames.LOCATION_PREDS]
                targets = result[MetricNames.LOCATION_TARGETS]
                n_samples = int(errors.numel())
                observed_total += n_samples

                if test_loaders is not None:
                    expected = len(test_loaders[index].dataset)
                    expected_total += expected
                    if n_samples != expected:
                        raise AssertionError(
                            f"Evaluation isolation failed for {name!r}: collected "
                            f"{n_samples} samples, expected {expected}."
                        )

                existing_percentiles = {
                    50.0: float(result[MetricNames.LOCATION_ERROR_MEDIAN]),
                    90.0: float(result[MetricNames.LOCATION_ERROR_90_PERCENTILE]),
                    99.0: float(result[MetricNames.LOCATION_ERROR_99_PERCENTILE]),
                }
                missing_percentiles = [
                    percentile
                    for percentile in percentiles
                    if float(percentile) not in existing_percentiles
                ]
                derived_percentiles = (
                    safe_percentiles(errors, missing_percentiles)
                    if missing_percentiles
                    else {}
                )
                percentile_values = {
                    float(percentile): existing_percentiles.get(
                        float(percentile),
                        derived_percentiles.get(float(percentile)),
                    )
                    for percentile in percentiles
                }
                scalar_metrics = {
                    f"test/{name}/location_error_mean": result[
                        MetricNames.LOCATION_ERROR_MEAN
                    ],
                    f"test/{name}/location_error_median": result[
                        MetricNames.LOCATION_ERROR_MEDIAN
                    ],
                    f"test/{name}/location_error_std": result[
                        MetricNames.LOCATION_ERROR_STD
                    ],
                    f"test/{name}/n_samples": float(n_samples),
                }
                for percentile, value in percentile_values.items():
                    scalar_metrics[
                        f"test/{name}/location_error_p{int(percentile)}"
                    ] = value

                aoa_mean = result[MetricNames.AOA_ERROR_MEAN_RADIAN]
                aoa_rmse = result[MetricNames.AOA_ERROR_RMSE_RADIAN]
                for ap_index in range(aoa_mean.numel()):
                    scalar_metrics[
                        f"test/{name}/aoa_error_mean_radian/ap_{ap_index}"
                    ] = aoa_mean[ap_index]
                    scalar_metrics[
                        f"test/{name}/aoa_error_rmse_radian/ap_{ap_index}"
                    ] = aoa_rmse[ap_index]
                scalar_metrics[f"test/{name}/aoa_error_mean_radian/mean"] = aoa_mean.mean()
                scalar_metrics[f"test/{name}/aoa_error_rmse_radian/mean"] = aoa_rmse.mean()
                self.log_dict(
                    scalar_metrics,
                    sync_dist=False,
                    add_dataloader_idx=False,
                )

                display = ", ".join(
                    f"p{int(percentile)}={value:.6g}"
                    for percentile, value in percentile_values.items()
                )
                self.print(
                    f"Test metrics [{name}]: n={n_samples}, "
                    f"mean={float(result[MetricNames.LOCATION_ERROR_MEAN]):.6g}, "
                    f"{display}"
                )

                row: dict[str, object] = {
                    "dataset": name,
                    "n_samples": n_samples,
                    "mean": float(result[MetricNames.LOCATION_ERROR_MEAN]),
                    "median": float(result[MetricNames.LOCATION_ERROR_MEDIAN]),
                    "std": float(result[MetricNames.LOCATION_ERROR_STD]),
                }
                for percentile, value in percentile_values.items():
                    row[f"p{int(percentile)}"] = value
                summary_rows.append(row)

                errors_np = errors.detach().cpu().numpy()
                errors_for_overlay[name] = errors_np
                if cdf_enabled:
                    figure = create_cdf_figure(
                        {name: errors_np},
                        max_points=max_points,
                        title=f"Location error CDF — {name}",
                    )
                    log_figure_and_close(
                        self.logger,
                        f"CDF_{name}",
                        figure,
                    )

                if save_raw_errors:
                    paths = save_dataset_assets(
                        output_dir=output_dir,
                        run_name=run_name,
                        dataset_name=name,
                        errors=errors,
                        predictions=predictions,
                        targets=targets,
                        raw_error_format=raw_error_format,
                    )
                    for path in paths:
                        log_asset(
                            self.logger,
                            path,
                            metadata={"dataset": name, "run_mode": run_mode},
                        )

            if test_loaders is not None and observed_total != expected_total:
                raise AssertionError(
                    "Evaluation isolation failed across dataloaders: "
                    f"collected {observed_total} samples, expected {expected_total}."
                )

            if cdf_enabled and overlay_enabled:
                figure = create_cdf_figure(
                    errors_for_overlay,
                    max_points=max_points,
                    title="Location error CDF — all datasets",
                )
                log_figure_and_close(
                    self.logger,
                    "CDF_all_datasets",
                    figure,
                )

            summary_path = save_summary_csv(
                output_dir=output_dir,
                run_name=run_name,
                rows=summary_rows,
            )
            log_asset(
                self.logger,
                summary_path,
                metadata={"run_mode": run_mode, "kind": "summary_metrics"},
            )
        finally:
            for metrics in self.test_metrics_per_ds:
                metrics.reset()

    def configure_optimizers(self):
        """Create the Adam optimizer and step learning-rate scheduler.

        Returns:
            Lightning optimizer and scheduler configuration.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
