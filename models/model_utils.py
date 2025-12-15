"""Utility functions for model training and loss computation."""

import torch
from utils.schema import GTlabel, ModelOutput, LossTerms, DLocBatchDataSample
import torch.nn.functional as F

ANGLE_LOSS_MULTIPLIER = 5
    
def compute_geometric_loss(model_output: ModelOutput, gt_label: GTlabel) -> LossTerms:
    """Compute loss function given model output and ground truth label.

    Args:
        model_output: dataclass that hold output of the model
        gt_label: dataclass that hold ground truth label

    Returns:
        LossTerms dataclass that contain all loss terms.
    """
    # compute the loss term
    cos_loss = F.huber_loss(model_output.cos_aoa, gt_label.cos_aoa,delta=0.5)
    sin_loss = F.huber_loss(model_output.sin_aoa, gt_label.sin_aoa, delta=1.0)
    location_loss = F.huber_loss(model_output.location, gt_label.location, delta=2.0)

    # Total loss
    total_loss = cos_loss * ANGLE_LOSS_MULTIPLIER + sin_loss * ANGLE_LOSS_MULTIPLIER + location_loss

    return LossTerms(
        total_loss=total_loss,
        cos_loss=cos_loss,
        sin_loss=sin_loss,
        location_loss=location_loss,
    )

def get_batch_gt_label(batch: DLocBatchDataSample) -> GTlabel:
    """Get ground truth label from batch data sample.

    Args:
        batch: batch data sample.

    Returns:
        A GTlabel dataclass containing the ground truth label.
    """
    return GTlabel(
        cos_aoa=torch.cos(batch.aoa_label),
        sin_aoa=torch.sin(batch.aoa_label),
        location=batch.location_label,
        aoa=batch.aoa_label,
        velocity=batch.velocity,
        timestamps=batch.timestamps
    )
