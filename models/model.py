"""Model definition and training and validation logic."""

from typing import Any, Mapping, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchvision.models import resnet34

from metrics_calculator import AoAAccuracy, LocationAccuracy, RSSIMetric
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
from utils.config_hydra import ExperimentConfig


# -----------------------------
# Helper modules
# -----------------------------

class QuadraticActivation(nn.Module):
    """y = -4/pi^2 * x^2 + 1"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((-4 / torch.pi ** 2) * x ** 2 + 1)


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
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
        x = self.resnet(x)
        return self.mlp(x)


class AoADecoder(nn.Module):
    def __init__(self, in_features: int, n_ap: int):
        super().__init__()

        self.aoa_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_ap),
            nn.Tanh(),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(in_features, n_ap),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        aoa = self.aoa_head(x)
        confidence = self.confidence_head(x)

        return ModelOutput(
            cos_aoa=None,
            sin_aoa=None,
            aoa=aoa,
            confidence=confidence,
            location=None,
        )


# -----------------------------
# Main Lightning model
# -----------------------------

class TrigAOAResNetModel(pl.LightningModule):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        # ✅ correct learning rate access
        self.lr = config.learning_rate

        # -------------------------
        # Metrics
        # -------------------------
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

        self.test_metrics = MetricCollection([
            AoAAccuracy(n_aps=config.dataset.test_n_aps),
            LocationAccuracy(),
            RSSIMetric(),
        ])

        # -------------------------
        # Model components
        # -------------------------
        self.encoders = nn.ModuleList([
            ResNetEncoder(
                in_channels=config.model.in_channels,
                dropout=config.model.dropout,
            )
            for _ in range(config.dataset.train_n_aps)
        ])

        encoder_dim = self.encoders[0].mlp_output_dim
        self.decoder = AoADecoder(encoder_dim, n_ap=1)

    # -------------------------
    # Forward
    # -------------------------

    def forward(self, x: torch.Tensor, ap_metadata: torch.Tensor) -> ModelOutput:
        ap_meta_list = APMetadata.from_tensor(ap_metadata)
        batch_size, n_aps = x.shape[0], x.shape[1]

        aoa_preds = torch.zeros(batch_size, n_aps, device=self.device)
        confidence_preds = torch.zeros(batch_size, n_aps, device=self.device)

        for ap_idx in range(n_aps):
            encoder_idx = min(ap_idx, len(self.encoders) - 1)
            features = self.encoders[encoder_idx](x[:, ap_idx].unsqueeze(1))
            out = self.decoder(features)

            aoa_preds[:, ap_idx] = out.aoa.squeeze(1) * torch.pi / 2
            confidence_preds[:, ap_idx] = out.confidence.squeeze(1)

        cos_aoa = torch.cos(aoa_preds)
        sin_aoa = torch.sin(aoa_preds)

        cos_map = torch.zeros_like(cos_aoa)
        sin_map = torch.zeros_like(sin_aoa)

        for i in range(batch_size):
            meta = ap_meta_list[i]
            cos_map[i] = cos_angle_diff(
                cos_aoa[i], sin_aoa[i],
                meta.cos_ap_orientations,
                meta.sin_ap_orientations,
            )
            sin_map[i] = sin_angle_diff(
                cos_aoa[i], sin_aoa[i],
                meta.cos_ap_orientations,
                meta.sin_ap_orientations,
            )

        location = torch.zeros(batch_size, 2, device=self.device)
        conf = torch.ones_like(confidence_preds)

        for i in range(batch_size):
            meta = ap_meta_list[i]
            loc = solve_ray_intersection_batch(
                meta.ap_locations,
                cos_map[i].unsqueeze(0),
                sin_map[i].unsqueeze(0),
                conf[i].unsqueeze(0),
            )
            location[i] = loc.squeeze(0)

        return ModelOutput(
            cos_aoa=cos_aoa,
            sin_aoa=sin_aoa,
            aoa=aoa_preds,
            confidence=confidence_preds,
            location=location,
        )

    # -------------------------
    # Train / Val / Test
    # -------------------------

    def _common_step(self, batch: DLocBatchDataSample, stage: str):
        pred = self(batch.features_2d, batch.ap_metadata)
        gt = get_batch_gt_label(batch)
        loss: LossTerms = compute_geometric_loss(pred, gt)

        self.log(f"{stage}_loss", loss.total_loss, prog_bar=True)
        return pred, gt, loss

    def training_step(self, batch, batch_idx):
        pred, gt, loss = self._common_step(batch, "train")
        self.train_metrics.update(pred, gt)
        self.train_metrics["RSSIMetric"].set_rssi(batch.rssi)
        self.log("learning_rate", self.lr)
        return loss.total_loss

    def validation_step(self, batch, batch_idx):
        pred, gt, loss = self._common_step(batch, "val")
        self.val_metrics.update(pred, gt)
        self.val_metrics["RSSIMetric"].set_rssi(batch.rssi)

    def test_step(self, batch, batch_idx):
        pred, gt, loss = self._common_step(batch, "test")
        self.test_metrics.update(pred, gt)
        self.test_metrics["RSSIMetric"].set_rssi(batch.rssi)

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.test_metrics.reset()

    # -------------------------
    # Optimizer
    # -------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=5e-5,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.9,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
