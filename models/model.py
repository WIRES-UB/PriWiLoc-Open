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


# """Model definition and training and validation logic."""

# from typing import Any, Mapping, Optional, Dict, Tuple

# import pytorch_lightning as pl
# import torch
# import torch.nn as nn
# from torchmetrics import MetricCollection
# from torchvision.models import resnet34
# from metrics_calculator import AoAAccuracy, LocationAccuracy, RSSIMetric
# from models.model_utils import compute_loss, compute_aoa_loss, get_batch_gt_label
# from utils.config import ExperimentConfig
# from utils.geometry_utils import cos_angle_diff, sin_angle_diff
# from utils.ray_intersection_solver import solve_2d_ray_intersection, solve_ray_intersection_batch
# from utils.schema import (
#     APMetadata,
#     DLocBatchDataSample,
#     GTlabel,
#     LossTerms,
#     ModelOutput,
# )

# class QuadraticActivation(nn.Module):
#     """y = -4/pi^2 * x^2 + 1 activation function. Resembles cos(x) when x is in the range of [-pi/2, pi/2]"""

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.relu((-4 / torch.pi**2) * x**2 + 1)


# class ResNetEncoder(nn.Module):
#     """ResNet Encoder for single AP"""

#     def __init__(self, in_channels: int = 1, dropout: float = 0.3, ap_context_dim: int = 4):
#         super().__init__()
#         self.resnet_encoder = resnet34(weights=None)
#         self.resnet_output_dim = self.resnet_encoder.fc.in_features

#         self.resnet_encoder.conv1 = nn.Conv2d(
#             in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False,
#         )
#         self.resnet_encoder.fc = nn.Identity()

#         self.mlp_output_dim = 64
#         self.mlp = nn.Sequential(
#             nn.Linear(self.resnet_output_dim + ap_context_dim, 256),  # ADD +ap_context_dim
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.Linear(128, self.mlp_output_dim),
#             nn.ReLU(),
#         )

   

#     def forward(self, x: torch.Tensor, ap_context: torch.Tensor = None) -> torch.Tensor:

#         """Forward pass through the ResNet encoder.

#         Args:
#             x: AoA-TOF data tensor for single AP, shape (batch_size, 1, height, width)

#         Returns:
#             ResNet features tensor, shape (batch_size, resnet_output_dim)
#         """
#         resnet_output = self.resnet_encoder(x)
        
#         if ap_context is not None:
#             resnet_output = torch.cat([resnet_output, ap_context], dim=1)
        
#         mlp_output = self.mlp(resnet_output)
#         return mlp_output


# class Decoder(nn.Module):
#     """Decoder for AoA prediction"""

#     def __init__(self, in_features: int, n_ap: int = 1):
#         super().__init__()

#         # Cos Decoder
#         self.cos_decoder = nn.Sequential(
#             nn.Linear(in_features, n_ap),
#             QuadraticActivation(),
#         )

#         # Sin Decoder
#         self.sin_decoder = nn.Sequential(nn.Linear(in_features, n_ap), nn.Tanh())

#         # Confidence Decoder
#         self.confidence_decoder = nn.Sequential(nn.Linear(in_features, n_ap), nn.Sigmoid())

#     def forward(self, x: torch.Tensor) -> ModelOutput:
#         """Forward pass through the decoder.

#         Args:
#             x: ResNet features tensor, shape (batch_size, resnet_output_dim)

#         Returns:
#             ModelOutput dataclass that contains the model output.
#         """
#         cos_aoa = self.cos_decoder(x)  # shape (batch_size, n_ap)
#         sin_aoa = self.sin_decoder(x)  # shape (batch_size, n_ap)
#         confidence = self.confidence_decoder(x)  # shape (batch_size, n_ap)

#         return ModelOutput(
#             cos_aoa=cos_aoa,
#             sin_aoa=sin_aoa,
#             confidence=confidence,
#             aoa=None,
#             location=None,
#         )

# class AoADecoder(nn.Module):
#     """Decoder for AoA prediction"""
    
#     def __init__(self, in_features: int, n_ap: int = 1):
#         super().__init__()
#         self.linear1 = nn.Linear(in_features, 256)
#         self.linear2 = nn.Linear(256, n_ap)
#         self.tanh = nn.Tanh()
#         self.confidence_head = nn.Sequential(nn.Linear(in_features, n_ap), nn.Sigmoid())

#     def forward(self, x: torch.Tensor) -> ModelOutput:
#         """Forward pass through the decoder.
        
#         Args:
#             x: ResNet features tensor, shape (batch_size, resnet_output_dim)
        
#         Returns:
#             ModelOutput dataclass that contains the model output.
#         """
#         aoa = self.linear1(x)
#         aoa = nn.ReLU()(aoa)
#         aoa = nn.Dropout(0.2)(aoa)
#         aoa = self.linear2(aoa)  # shape (batch_size, n_ap)
#         aoa = self.tanh(aoa)
#         confidence = self.confidence_head(x)

#         return ModelOutput(
#             cos_aoa=None,
#             sin_aoa=None,
#             confidence=confidence,
#             aoa=aoa,
#             location=None,
#         )
    
# class TrigAOAResNetModel(pl.LightningModule):
#     def __init__(self, config: ExperimentConfig):
#         super().__init__()
#         self.config = config
#         self.lr = config.lr

#         # Initialize metrics calculator
#         # Ref: https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metriccollection
#         self.train_metrics = MetricCollection([AoAAccuracy(n_aps=self.config.dataset_config.train_n_aps), LocationAccuracy(), RSSIMetric()])
#         self.val_metrics = MetricCollection([AoAAccuracy(n_aps=self.config.dataset_config.val_n_aps), LocationAccuracy(), RSSIMetric()])
#         self.test_metrics = MetricCollection([AoAAccuracy(n_aps=self.config.dataset_config.test_n_aps), LocationAccuracy(), RSSIMetric()])

#         # ResNet Encoder for single AP
#         self.resnet_encoder_list = nn.ModuleList([ResNetEncoder(1) for _ in range(self.config.dataset_config.train_n_aps)])
#         encoder_mlp_output_dim = self.resnet_encoder_list[0].mlp_output_dim

#         # Initialize decoder
#         self.decoder = AoADecoder(encoder_mlp_output_dim, n_ap=1)

#     def forward(self, x: torch.Tensor, ap_metadata: torch.Tensor) -> ModelOutput:
#         """Forward pass through the model.
        
#         Args:
#             x: input tensor, shape (batch_size, n_aps, height, width)
#             ap_metadata: tensor of shape (batch_size, n_aps, 5)
        
#         Returns:
#             A dataclass that contains the model output.
#         """
            
        
        
#         ap_metadata_list = APMetadata.from_tensor(ap_metadata)
#         batch_size = x.shape[0]
#         n_aps = x.shape[1]  # Assume all have same number (4)
        
#         # Initialize outputs
#         aoa_preds = torch.zeros(batch_size, n_aps, device=self.device)
#         confidence_preds = torch.zeros(batch_size, n_aps, device=self.device)
        
#         # Process each AP across the ENTIRE batch (for batch norm)
#         for ap_index in range(n_aps):
#             x_ap = x[:, ap_index, :, :].unsqueeze(1)
#             encoder_idx = min(ap_index, len(self.resnet_encoder_list)-1)
            
#             # CREATE AP CONTEXT
#             ap_contexts = []
#             for i in range(batch_size):
#                 ap_meta = ap_metadata_list[i]
#                 ap_context = torch.tensor([
#                     ap_meta.cos_ap_orientations[ap_index].item(),
#                     ap_meta.sin_ap_orientations[ap_index].item(),
#                     ap_meta.ap_locations[ap_index, 0].item() / 20.0,
#                     ap_meta.ap_locations[ap_index, 1].item() / 20.0,
#                 ], device=self.device)
#                 ap_contexts.append(ap_context)
#             ap_context_batch = torch.stack(ap_contexts)
            
#             # PASS CONTEXT TO ENCODER
#             resnet_features = self.resnet_encoder_list[encoder_idx](x_ap, ap_context_batch)
#             decoder_output = self.decoder(resnet_features)
            
#             aoa_preds[:, ap_index] = decoder_output.aoa.squeeze(1) * torch.pi/2
#             confidence_preds[:, ap_index] = decoder_output.confidence.squeeze(1)
    
        
#         # NOW do per-sample transformations
#         cos_aoa_preds_map_frame = torch.zeros_like(aoa_preds)
#         sin_aoa_preds_map_frame = torch.zeros_like(aoa_preds)
        
#         for i in range(batch_size):
#             ap_meta = ap_metadata_list[i]
            
#             cos_aoa_preds_map_frame[i, :] = cos_angle_diff(
#                 torch.cos(aoa_preds[i, :]),
#                 torch.sin(aoa_preds[i, :]),
#                 ap_meta.cos_ap_orientations,
#                 ap_meta.sin_ap_orientations,
#             )
#             sin_aoa_preds_map_frame[i, :] = sin_angle_diff(
#                 torch.cos(aoa_preds[i, :]),
#                 torch.sin(aoa_preds[i, :]),
#                 ap_meta.cos_ap_orientations,
#                 ap_meta.sin_ap_orientations,
#             )
        
#         # Triangulation per sample
#         location_preds = torch.zeros(batch_size, 2, device=self.device)
#         confidence_score = torch.ones_like(confidence_preds)
        
#         for i in range(batch_size):
#             ap_meta = ap_metadata_list[i]
#             result = solve_ray_intersection_batch(
#                 ap_meta.ap_locations,
#                 cos_aoa_preds_map_frame[i, :].unsqueeze(0),
#                 sin_aoa_preds_map_frame[i, :].unsqueeze(0),
#                 confidence_score[i, :].unsqueeze(0)
#             )
#             location_preds[i, :] = result.squeeze(0)
        
#         return ModelOutput(
#             cos_aoa=None,
#             sin_aoa=None,
#             location=location_preds,
#             confidence=confidence_preds,
#             aoa=aoa_preds,
#         )

#     def _common_step(self, batch: DLocBatchDataSample, stage: str) -> Tuple[ModelOutput, GTlabel, LossTerms]:
#         assert stage in ["train", "val", "test"], "Stage must be one of 'train', 'val', or 'test'."

#         # Forward pass
#         model_pred: ModelOutput = self.forward(batch.features_2d, batch.ap_metadata)
    
#         # Store AP metadata for visualization callback (use first sample's metadata)
#         ap_metadata_list = APMetadata.from_tensor(batch.ap_metadata)
#         self.ap_metadata = ap_metadata_list[0]  # Store first sample's metadata for visualization

#         # construct ground truth label
#         gt_label: GTlabel = get_batch_gt_label(batch)

#         # DEBUG: Check shapes before metrics
#         # print(f"\n=== DEBUG {stage} ===")
#         # print(f"Batch size: {batch.features_2d.shape[0]}")
#         # print(f"model_pred.aoa shape: {model_pred.aoa.shape}")
#         # print(f"gt_label.aoa shape: {gt_label.aoa.shape}")
#         # print(f"batch.features_2d shape: {batch.features_2d.shape}")
#         # print(f"batch.aoa_label shape: {batch.aoa_label.shape}")
#         # if isinstance(batch.ap_metadata, torch.Tensor):
#         #     print(f"batch.ap_metadata shape: {batch.ap_metadata.shape}")
#         # else:
#         #     print(f"batch.ap_metadata type: {type(batch.ap_metadata)}")
#         # print("==================\n")

#         # compute loss
#         loss_all: LossTerms = compute_aoa_loss(model_pred, gt_label)

#         # log the loss
#         self.log(f"{stage}_loss", loss_all.total_loss.item())

#         return model_pred, gt_label, loss_all
    
#     def training_step(self, batch: DLocBatchDataSample, batch_idx: int) -> torch.Tensor:


#         model_pred, gt_label, train_loss = self._common_step(batch, "train")
#         self.train_metrics.update(model_pred, gt_label)
#         self.train_metrics['RSSIMetric'].set_rssi(batch.rssi)  # Use string name

#         # log learning rate
#         lr = self.trainer.optimizers[0].param_groups[0]["lr"]
#         self.log("learning_rate", lr)

#         return train_loss.total_loss


#     def validation_step(self, batch: DLocBatchDataSample, batch_idx: int) -> Mapping[str, Any]:
#         model_pred, gt_label, val_loss = self._common_step(batch, "val")
#         self.val_metrics.update(model_pred, gt_label)
#         self.val_metrics['RSSIMetric'].set_rssi(batch.rssi)  # Use string name

#         return {
#             "val_loss": val_loss.total_loss,
#             "model_pred": model_pred,
#             "gt_label": gt_label,
#         }

#     def test_step(self, batch: DLocBatchDataSample, batch_idx: int) -> Mapping[str, Any]:
#         model_pred, gt_label, test_loss = self._common_step(batch, "test")
#         self.test_metrics.update(model_pred, gt_label)
#         self.test_metrics['RSSIMetric'].set_rssi(batch.rssi) 
        
#         return {
#             "test_loss": test_loss.total_loss,
#             "model_pred": model_pred,
#             "gt_label": gt_label,
#         }

#     def on_train_epoch_end(self):
#         self.train_metrics.reset()

#     def on_validation_epoch_end(self):
#         self.val_metrics.reset()

#     def on_test_epoch_end(self):
#         self.test_metrics.reset()

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay= 5e-5)
#         lr_scheduler = {
#             "scheduler": torch.optim.lr_scheduler.StepLR(
#                 optimizer,
#                 step_size=10,
#                 gamma = 0.9
#                 # max_lr=self.lr,
#                 # total_steps=self.trainer.estimated_stepping_batches,
#             ),
#             "interval": "epoch",  # or "epoch"
#             "frequency": 1,  # Update the scheduler every step
#         }
#         return [optimizer], [lr_scheduler]


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = TrigAOAResNetModel().to(device)
#     sample_input = torch.randn(5, 4, 224, 224).to(device)
#     model_output = model(sample_input)
#     print(f"device is {model.device}")
#     print(f"Model output cos aoa: {model_output.cos_aoa}")
#     print(f"Model output sin aoa: {model_output.sin_aoa}")
#     print(f"Model output location: {model_output.location.shape}")
#     print(f"Model output confidence: {model_output.confidence.shape}")

# def forward(self, x: torch.Tensor, ap_metadata: torch.Tensor) -> ModelOutput:
    #     """Forward pass through the model.

    #     Args:
    #         x: input tensor, shape (batch_size, n_aps, height, width)

    #     Returns:
    #         A dataclass that contains the model output.
    #     """

    #     ap_metadata = APMetadata.from_tensor(ap_metadata)
    #     # permute order of ap during training to improve robustness
    #     ap_locations = ap_metadata.ap_locations.to(self.device)
    #     ap_aoas = ap_metadata.ap_orientations.to(self.device)
    #     cos_ap_orientations = ap_metadata.cos_ap_orientations.to(self.device)
    #     sin_ap_orientations = ap_metadata.sin_ap_orientations.to(self.device)
    #     n_aps = ap_metadata.n_aps

    #     # Initialize lists to store the results
    #     batch_size = x.shape[0]
    #     # cos_aoa_preds = torch.zeros(batch_size, n_aps, device=self.device)
    #     # sin_aoa_preds = torch.zeros(batch_size, n_aps, device=self.device)
    #     aoa_preds = torch.zeros(batch_size, n_aps, device=self.device)
    #     confidence_preds = torch.zeros(batch_size, n_aps, device=self.device)

    #     for ap_index in range(n_aps):
    #         x_ap = x[:, ap_index, :, :].unsqueeze(1)  # shape (batch_size, 1, height, width)

    #         # Need to consider the case when n_aps > len(self.resnet_encoder_list). This happens when number of APs
    #         # in the val or test set is larger than the number of APs in the training set.
    #         encoder_idx = min(ap_index, len(self.resnet_encoder_list)-1)

    #         # Pass AOA-TOF data through the encoder.
    #         resnet_features_single_ap = self.resnet_encoder_list[encoder_idx](x_ap)

    #         # Decoders, output shape is (batch_size, 1)
    #         decoder_output: ModelOutput = self.decoder(resnet_features_single_ap)

    #         # store the results in original order
    #         # cos_aoa_preds[:, ap_index] = decoder_output.cos_aoa.squeeze(1)
    #         # sin_aoa_preds[:, ap_index] = decoder_output.sin_aoa.squeeze(1)
    #         aoa_preds[:, ap_index] = decoder_output.aoa.squeeze(1) * torch.pi/2
    #         confidence_preds[:, ap_index] = decoder_output.confidence.squeeze(1)

    #     # convert AoA from AP frame to map frame
    #     cos_aoa_preds_map_frame = torch.zeros(batch_size, n_aps, device=self.device)
    #     sin_aoa_preds_map_frame = torch.zeros(batch_size, n_aps, device=self.device)
    #     for i in range(batch_size):
    #         cos_aoa_preds_map_frame[i, :] = cos_angle_diff(
    #             torch.cos(aoa_preds[i, :]),
    #             torch.sin(aoa_preds[i, :]),
    #             cos_ap_orientations[i, :],
    #             sin_ap_orientations[i, :],
    #         )
    #         sin_aoa_preds_map_frame[i, :] = sin_angle_diff(
    #             torch.cos(aoa_preds[i, :]),
    #             torch.sin(aoa_preds[i, :]),
    #             cos_ap_orientations[i, :],
    #             sin_ap_orientations[i, :],
    #         )

    #     # compute AoA prediction
    #     # aoa_preds = torch.atan(sin_aoa_preds / cos_aoa_preds)

    #     confidence_score = torch.ones_like(confidence_preds)

    #     # compute the intersection point
    #     location_preds = torch.zeros(batch_size, 2, device=self.device)
    #     for i in range(batch_size):
    #         location_preds[i, :] = solve_ray_intersection_batch(
    #             ap_locations[i, :],
    #             cos_aoa_preds_map_frame[i, :],
    #             sin_aoa_preds_map_frame[i, :],
    #             confidence_score[i, :]
    #         )
    #     # location_preds = solve_ray_intersection_batch(
    #     #     ap_locations,
    #     #     cos_aoa_preds_map_frame,
    #     #     sin_aoa_preds_map_frame,
    #     #     confidence_score,
    #     # )

    #     # location_preds = solve_2d_ray_intersection(
    #     #     ap_aoas,
    #     #     ap_locations,
    #     #     aoa_preds,
    #     # )

        # return ModelOutput(
        #     cos_aoa=None,
        #     sin_aoa=None,
        #     location=location_preds,
        #     confidence=confidence_preds,
        #     aoa=aoa_preds,
        # )
