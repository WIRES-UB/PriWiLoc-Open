import torch

from metrics_calculator import AoAAccuracy, LocationAccuracy
from utils.schema import GTlabel, ModelOutput


def test_metric_updates_do_not_mutate_inputs():
    pred = ModelOutput(
        aoa=torch.randn(2, 3, 4),
        cos_aoa=torch.randn(2, 3, 4),
        sin_aoa=torch.randn(2, 3, 4),
        location=torch.randn(2, 3, 2),
        confidence=torch.ones(2, 3, 4),
    )
    target = GTlabel(
        aoa=torch.randn(2, 3, 4),
        cos_aoa=torch.randn(2, 3, 4),
        sin_aoa=torch.randn(2, 3, 4),
        location=torch.randn(2, 3, 2),
        velocity=torch.randn(2, 3, 2),
        timestamps=torch.randn(2, 3, 1),
    )
    original_pred_shapes = (pred.aoa.shape, pred.location.shape)
    original_target_shapes = (
        target.aoa.shape,
        target.location.shape,
        target.velocity.shape,
        target.timestamps.shape,
    )

    AoAAccuracy(n_aps=4).update(pred, target)
    LocationAccuracy().update(pred, target)

    assert (pred.aoa.shape, pred.location.shape) == original_pred_shapes
    assert (
        target.aoa.shape,
        target.location.shape,
        target.velocity.shape,
        target.timestamps.shape,
    ) == original_target_shapes
