"""Model definition and training and validation logic."""

from typing import Any, Mapping, Optional, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchvision.models import resnet34
from metrics_calculator import AoAAccuracy, LocationAccuracy, RSSIMetric
from models.model_utils import compute_direct_aoa_loss, compute_geometric_loss, get_batch_gt_label
from utils.config import ExperimentConfig
from utils.geometry_utils import cos_angle_diff, sin_angle_diff
from utils.ray_intersection_solver import solve_2d_ray_intersection, solve_ray_intersection_batch
from utils.schema import (
    APMetadata,
    DLocBatchDataSample,
    GTlabel,
    LossTerms,
    ModelOutput,
)

class QuadraticActivation(nn.Module):
    """y = -4/pi^2 * x^2 + 1 activation function. Resembles cos(x) when x is in the range of [-pi/2, pi/2]"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((-4 / torch.pi**2) * x**2 + 1)


class ResNetEncoder(nn.Module):
    """ResNet Encoder for single AP"""

    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        """Constructor for ResNetEncoder.

        Args:
            in_channels: Number of Conv channel in first conv layer. Defaults to 1.
        """
        super().__init__()
        self.resnet_encoder = resnet34(weights=None)
        self.resnet_output_dim = self.resnet_encoder.fc.in_features

        # Customize the first convolutional layer to accept `in_channels` channels
        self.resnet_encoder.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

        # Remove the final classification layer
        self.resnet_encoder.fc = nn.Identity()

        # Add linear layer to reduce the output dimension
        self.mlp_output_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(self.resnet_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, self.mlp_output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet encoder.

        Args:
            x: AoA-TOF data tensor for single AP, shape (batch_size, 1, height, width)

        Returns:
            ResNet features tensor, shape (batch_size, resnet_output_dim)
        """
        resnet_output = self.resnet_encoder(x)  # shape (batch_size, resnet_output_dim)
        mlp_output = self.mlp(resnet_output)  # shape (batch_size, mlp_output_dim)
        return mlp_output


class Decoder(nn.Module):
    """Decoder for AoA prediction"""

    def __init__(self, in_features: int, n_ap: int = 1):
        super().__init__()

        # Cos Decoder
        self.cos_decoder = nn.Sequential(
            nn.Linear(in_features, n_ap),
            QuadraticActivation(),
        )

        # Sin Decoder
        self.sin_decoder = nn.Sequential(nn.Linear(in_features, n_ap), nn.Tanh())

        # Confidence Decoder
        self.confidence_decoder = nn.Sequential(nn.Linear(in_features, n_ap), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass through the decoder.

        Args:
            x: ResNet features tensor, shape (batch_size, resnet_output_dim)

        Returns:
            ModelOutput dataclass that contains the model output.
        """
        cos_aoa = self.cos_decoder(x)  # shape (batch_size, n_ap)
        sin_aoa = self.sin_decoder(x)  # shape (batch_size, n_ap)
        confidence = self.confidence_decoder(x)  # shape (batch_size, n_ap)

        return ModelOutput(
            cos_aoa=cos_aoa,
            sin_aoa=sin_aoa,
            confidence=confidence,
            aoa=None,
            location=None,
        )

class AoADecoder(nn.Module):
    """Decoder for AoA prediction"""
    
    def __init__(self, in_features: int, n_ap: int = 1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 256)
        self.linear2 = nn.Linear(256, n_ap)
        self.tanh = nn.Tanh()
        self.confidence_head = nn.Sequential(nn.Linear(in_features, n_ap), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass through the decoder.
        
        Args:
            x: ResNet features tensor, shape (batch_size, resnet_output_dim)
        
        Returns:
            ModelOutput dataclass that contains the model output.
        """
        aoa = self.linear1(x)
        aoa = nn.ReLU()(aoa)
        aoa = nn.Dropout(0.2)(aoa)
        aoa = self.linear2(aoa)  # shape (batch_size, n_ap)
        aoa = self.tanh(aoa)
        confidence = self.confidence_head(x)

        return ModelOutput(
            cos_aoa=None,
            sin_aoa=None,
            confidence=confidence,
            aoa=aoa,
            location=None,
        )
    
class TrigAOAResNetModel(pl.LightningModule):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.lr = config.lr

        # Initialize metrics calculator
        self.train_metrics = MetricCollection([AoAAccuracy(n_aps=self.config.dataset_config.train_n_aps), LocationAccuracy(), RSSIMetric()])
        self.val_metrics = MetricCollection([AoAAccuracy(n_aps=self.config.dataset_config.val_n_aps), LocationAccuracy(), RSSIMetric()])
        self.test_metrics = MetricCollection([AoAAccuracy(n_aps=self.config.dataset_config.test_n_aps), LocationAccuracy(), RSSIMetric()])

        # ResNet Encoder for single AP
        self.resnet_encoder_list = nn.ModuleList([ResNetEncoder(1) for _ in range(self.config.dataset_config.train_n_aps)])
        encoder_mlp_output_dim = self.resnet_encoder_list[0].mlp_output_dim

        # Initialize decoder
        self.decoder = AoADecoder(encoder_mlp_output_dim, n_ap=1)

    def forward(self, x: torch.Tensor, ap_metadata: torch.Tensor) -> ModelOutput:
        """Forward pass through the model - PriWiLoc paper version with geometric loss.
        
        Args:
            x: input tensor, shape (batch_size, n_aps, height, width)
            ap_metadata: tensor of shape (batch_size, n_aps, 5)
        
        Returns:
            A dataclass that contains the model output with cos_aoa, sin_aoa, and location.
        """
        ap_metadata_list = APMetadata.from_tensor(ap_metadata)
        batch_size = x.shape[0]
        n_aps = x.shape[1]
        
        # Initialize outputs
        aoa_preds = torch.zeros(batch_size, n_aps, device=self.device)
        confidence_preds = torch.zeros(batch_size, n_aps, device=self.device)
        
        # Process each AP across the ENTIRE batch (for batch norm)
        for ap_index in range(n_aps):
            x_ap = x[:, ap_index, :, :].unsqueeze(1)  # (batch_size, 1, H, W)
            encoder_idx = min(ap_index, len(self.resnet_encoder_list)-1)
            
            resnet_features = self.resnet_encoder_list[encoder_idx](x_ap)
            decoder_output = self.decoder(resnet_features)
            
            aoa_preds[:, ap_index] = decoder_output.aoa.squeeze(1) * torch.pi/2
            confidence_preds[:, ap_index] = decoder_output.confidence.squeeze(1)
        
        # Compute cos and sin from aoa predictions (for geometric loss)
        cos_aoa_preds = torch.cos(aoa_preds)
        sin_aoa_preds = torch.sin(aoa_preds)
        
        # Transform to map frame for triangulation
        cos_aoa_preds_map_frame = torch.zeros_like(aoa_preds)
        sin_aoa_preds_map_frame = torch.zeros_like(aoa_preds)
        
        for i in range(batch_size):
            ap_meta = ap_metadata_list[i]
            
            cos_aoa_preds_map_frame[i, :] = cos_angle_diff(
                torch.cos(aoa_preds[i, :]),
                torch.sin(aoa_preds[i, :]),
                ap_meta.cos_ap_orientations,
                ap_meta.sin_ap_orientations,
            )
            sin_aoa_preds_map_frame[i, :] = sin_angle_diff(
                torch.cos(aoa_preds[i, :]),
                torch.sin(aoa_preds[i, :]),
                ap_meta.cos_ap_orientations,
                ap_meta.sin_ap_orientations,
            )
        
        # Triangulation per sample (required for geometric loss)
        location_preds = torch.zeros(batch_size, 2, device=self.device)
        confidence_score = torch.ones_like(confidence_preds)
        
        for i in range(batch_size):
            ap_meta = ap_metadata_list[i]
            result = solve_ray_intersection_batch(
                ap_meta.ap_locations,
                cos_aoa_preds_map_frame[i, :].unsqueeze(0),
                sin_aoa_preds_map_frame[i, :].unsqueeze(0),
                confidence_score[i, :].unsqueeze(0)
            )
            location_preds[i, :] = result.squeeze(0)
        
        return ModelOutput(
            cos_aoa=cos_aoa_preds,      # For geometric loss
            sin_aoa=sin_aoa_preds,      # For geometric loss
            location=location_preds,     # For geometric loss
            confidence=confidence_preds,
            aoa=aoa_preds,              # For metrics/visualization
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
        self.log(f"{stage}_loss", loss_all.total_loss.item())
        
        # Log individual loss components
        if loss_all.cos_loss is not None:
            self.log(f"{stage}_cos_loss", loss_all.cos_loss.item())
        if loss_all.sin_loss is not None:
            self.log(f"{stage}_sin_loss", loss_all.sin_loss.item())
        if loss_all.location_loss is not None:
            self.log(f"{stage}_location_loss", loss_all.location_loss.item())

        return model_pred, gt_label, loss_all
    
    def training_step(self, batch: DLocBatchDataSample, batch_idx: int) -> torch.Tensor:
        model_pred, gt_label, train_loss = self._common_step(batch, "train")
        self.train_metrics.update(model_pred, gt_label)
        self.train_metrics['RSSIMetric'].set_rssi(batch.rssi)

        # log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr)

        return train_loss.total_loss

    def validation_step(self, batch: DLocBatchDataSample, batch_idx: int) -> Mapping[str, Any]:
        model_pred, gt_label, val_loss = self._common_step(batch, "val")
        self.val_metrics.update(model_pred, gt_label)
        self.val_metrics['RSSIMetric'].set_rssi(batch.rssi)

        return {
            "val_loss": val_loss.total_loss,
            "model_pred": model_pred,
            "gt_label": gt_label,
        }

    def test_step(self, batch: DLocBatchDataSample, batch_idx: int) -> Mapping[str, Any]:
        model_pred, gt_label, test_loss = self._common_step(batch, "test")
        self.test_metrics.update(model_pred, gt_label)
        self.test_metrics['RSSIMetric'].set_rssi(batch.rssi) 
        
        return {
            "test_loss": test_loss.total_loss,
            "model_pred": model_pred,
            "gt_label": gt_label,
        }

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay= 5e-5)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma = 0.9
            ),
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
