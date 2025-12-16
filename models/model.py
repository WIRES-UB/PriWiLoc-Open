"""Model definition and training and validation logic."""

from typing import Tuple

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
from utils.config_hydra import Config 


# -----------------------------
# Helper modules
# -----------------------------

class QuadraticActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((-4 / torch.pi ** 2) * x ** 2 + 1)


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels: int, dropout: float):
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
        return self.mlp(self.resnet(x))


class AoADecoder(nn.Module):
    def __init__(self, in_features: int):
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
    def __init__(self, config: Config):
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
        pred, gt, loss = self._common_step(batch, "train")
        self.train_metrics.update(pred, gt)
        self.train_metrics["RSSIMetric"].set_rssi(batch.rssi)
        return loss.total_loss

    def validation_step(self, batch, batch_idx):
        pred, gt, loss = self._common_step(batch, "val")
        self.val_metrics.update(pred, gt)
        self.val_metrics["RSSIMetric"].set_rssi(batch.rssi)

        return {
            "val_loss": loss.total_loss,
            "model_pred": pred,
            "gt_label": gt,
        }

    def test_step(self, batch, batch_idx):
        pred, gt, loss = self._common_step(batch, "test")
        self.test_metrics.update(pred, gt)
        self.test_metrics["RSSIMetric"].set_rssi(batch.rssi)

        return {
            "test_loss": loss.total_loss,
            "model_pred": pred,
            "gt_label": gt,
        }

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
