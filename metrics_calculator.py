"""Implement various metrics calculators for evaluating the performance of the model.
"""
from torchmetrics import Metric
import torch
from utils.schema import ModelOutput, GTlabel
from utils.geometry_utils import wrap_to_pi
from typing import Dict

class MetricNames:
    """Class to store all metrics that metrics calculator generates."""
    AOA_ERROR_MEAN_RADIAN = "aoa_error_mean_radian"
    AOA_ERROR_STD_RADIAN = "aoa_error_std_radian"
    AOA_ERROR_RMSE_RADIAN = "aoa_error_mse_radian"
    AOA_ERROR_ALL_RADIAN = "aoa_errors_all_radian"
    AOA_PREDS_RADIAN = "aoa_preds_radian"
    AOA_TARGETS_RADIAN = "aoa_targets_radian"
    LOCATION_ERROR_MEAN = "location_error_mean"
    LOCATION_ERROR_MEDIAN = "location_error_median"
    LOCATION_ERROR_STD = "location_error_std"
    LOCATION_ERROR_90_PERCENTILE = "location_error_90_percentile"
    LOCATION_ERROR_99_PERCENTILE = "location_error_99_percentile"
    LOCATION_ERROR_ALL = "location_error_all"
    LOCATION_PREDS = "location_preds"
    LOCATION_TARGETS = "location_targets"
    VELOCITY = "velocity"
    TIMESTAMPS = "timestamps"
    RSSI_ALL = "rssi"


class AoAAccuracy(Metric):
    """This class computes mean and standard deviation of AoA error.
    The error is defined as difference between predicted AoA and the ground truth AoA. Unit is all in radians.

    Reference: https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
    """
    def __init__(self, n_aps: int, **kwargs):
        """Initialize the class.

        Args:
            n_aps: number of APs.
        """
        super().__init__(**kwargs)
        self.add_state("sum_aoa_error", default=torch.zeros(n_aps), dist_reduce_fx="sum")
        self.add_state("sum_aoa_error_sq", default=torch.zeros(n_aps), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("preds", default=[])
        self.add_state("targets", default=[])
        self.add_state("aoa_error_all", default=[])

    def update(self, pred: ModelOutput, target: GTlabel) -> None:
        # Calculate the difference between prediction and ground truth
        # pred and target aoa shape is (batch_size, n_aps)
        assert pred.aoa.shape == target.aoa.shape
        if pred.aoa.dim() == 3: # batch x seq x n_aps
            pred.aoa = pred.aoa[:, -1, :]
            target.aoa = target.aoa[:, -1, :]
            
        aoa_diff_radian = wrap_to_pi(pred.aoa - target.aoa)
        n_sample = pred.aoa.shape[0]

        # Update the states
        self.sum_aoa_error += aoa_diff_radian.sum(dim=0)
        self.sum_aoa_error_sq += (aoa_diff_radian ** 2).sum(dim=0)
        self.total += n_sample

        # accumulate prediction and target aoa
        self.preds.append(pred.aoa)
        self.targets.append(target.aoa)
        self.aoa_error_all.append(aoa_diff_radian)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute mean and standard deviation of AoA error.

        Returns:
            A dictionary containing mean and standard deviation of AoA error in radian.
            the shape of mean is (n_aps,) and the shape of std is (n_aps,)
        """
        # Compute the mean AoA error
        aoa_error_mean = self.sum_aoa_error / self.total
        aoa_error_std = torch.sqrt(self.sum_aoa_error_sq / self.total - aoa_error_mean ** 2)
        aoa_error_rmse = torch.sqrt(self.sum_aoa_error_sq / self.total)

        # convert aoa prediction and target to tensor
        preds_tensor = torch.cat(self.preds, dim=0)
        targets_tensor = torch.cat(self.targets, dim=0)
        aoa_error_tensor = torch.cat(self.aoa_error_all, dim=0)

        return {MetricNames.AOA_ERROR_MEAN_RADIAN: aoa_error_mean,
                MetricNames.AOA_ERROR_STD_RADIAN: aoa_error_std,
                MetricNames.AOA_ERROR_RMSE_RADIAN: aoa_error_rmse,
                MetricNames.AOA_PREDS_RADIAN: preds_tensor,
                MetricNames.AOA_TARGETS_RADIAN: targets_tensor,
                MetricNames.AOA_ERROR_ALL_RADIAN: aoa_error_tensor}

class RSSIMetric(Metric):
    """This class accumulates RSSI values across batches."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("rssi_values", default=[])

    def update(self, pred: ModelOutput, target: GTlabel) -> None:
        """Update method that matches the signature of other metrics.
        
        Note: This metric doesn't use pred or target, but needs to accept
        them to be compatible with MetricCollection.update() calls.
        RSSI values are set separately via set_rssi() method.
        
        Args:
            pred: Model predictions (unused)
            target: Ground truth labels (unused)
        """
        # Do nothing here - RSSI will be set via set_rssi()
        pass
    
    def set_rssi(self, rssi: torch.Tensor) -> None:
        """Set RSSI values for the current batch.
        
        Args:
            rssi: RSSI tensor of shape (batch_size, n_aps)
        """
        self.rssi_values.append(rssi)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute and return accumulated RSSI values.
        
        Returns:
            A dictionary containing all RSSI values concatenated.
        """
        if len(self.rssi_values) == 0:
            return {MetricNames.RSSI_ALL: torch.tensor([])}
        
        rssi_tensor = torch.cat(self.rssi_values, dim=0)
        return {MetricNames.RSSI_ALL: rssi_tensor}
    
class LocationAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[])
        self.add_state("targets", default=[])
        self.add_state("velocity", default=[])
        self.add_state("timestamps", default=[])

    def update(self, pred: ModelOutput, target: GTlabel) -> None:
        # accumulate prediction and target location
        if pred.location.dim() == 3: # batch x seq x 2
            pred.location = pred.location[:, -1, :]
            target.location = target.location[:, -1, :]
            target.velocity = target.velocity[:, -1, :]
            target.timestamps = target.timestamps[:, -1, :]

        self.preds.append(pred.location)
        self.targets.append(target.location)
        self.velocity.append(target.velocity)
        self.timestamps.append(target.timestamps)

    def compute(self) -> Dict:
        """Compute mean, median, standard deviation, 90th percentile, and 99th percentile of location error.

        Returns:
            A dictionary containing the computed metrics. Also return the prediction and target tensors for
            plotting purpose.
        """
        preds_tensor = torch.cat(self.preds, dim=0) # shape: (n_samples, 2)
        targets_tensor = torch.cat(self.targets, dim=0) # shape: (n_samples, 2)
        errors = torch.norm(preds_tensor - targets_tensor, dim=1) # shape: (n_samples,)
        mean_error = errors.mean()
        median_error = errors.median()
        std_error = errors.std()
        percentile_90_error = torch.quantile(errors, 0.90)
        percentile_99_error = torch.quantile(errors, 0.99)

        velocity_tensor = torch.cat(self.velocity, dim=0)
        timestamps_tensor = torch.cat(self.timestamps, dim=0)

        return {
            MetricNames.LOCATION_ERROR_MEAN: mean_error,
            MetricNames.LOCATION_ERROR_MEDIAN: median_error,
            MetricNames.LOCATION_ERROR_STD: std_error,
            MetricNames.LOCATION_ERROR_90_PERCENTILE: percentile_90_error,
            MetricNames.LOCATION_ERROR_99_PERCENTILE: percentile_99_error,
            MetricNames.LOCATION_ERROR_ALL: errors,
            MetricNames.LOCATION_PREDS: preds_tensor,
            MetricNames.LOCATION_TARGETS: targets_tensor,
            MetricNames.VELOCITY: velocity_tensor,
            MetricNames.TIMESTAMPS: timestamps_tensor
        }

if __name__ == "__main__":
    from torchmetrics import MetricCollection
    # Test the AoAAccuracy class
    batch_size = 5
    n_batch = 6
    n_aps = 4
    metrics_collection = MetricCollection([AoAAccuracy(n_aps), LocationAccuracy()])

    # prepare fake data
    pred_list = [ModelOutput(aoa=torch.randn(batch_size, n_aps),
                             cos_aoa=None,
                             sin_aoa=None,
                             location=torch.rand(batch_size, 2),
                             confidence=None) for _ in range(n_batch)]
    target_list = [GTlabel(aoa=torch.rand(batch_size, n_aps),
                           cos_aoa=None,
                           sin_aoa=None,
                           location=torch.rand(batch_size, 2)) for _ in range(n_batch)]

    # extract values into tensor
    pred_value_tensor = torch.cat([pred.location for pred in pred_list], dim=0)
    target_value_tensor = torch.cat([target.location for target in target_list], dim=0)

    # compute mean and std of the difference
    diff_tensor = torch.norm(pred_value_tensor - target_value_tensor, dim=1)
    diff_mean = diff_tensor.mean(dim=0)
    diff_std = diff_tensor.std()
    diff_median = diff_tensor.median()
    print(f"diff_mean: {diff_mean}, diff_std: {diff_std}, diff_median: {diff_median}")

    # compute mean using the metric
    for idx,(pred, target) in enumerate(zip(pred_list, target_list)):
        metrics_collection.update(pred, target)
        # print(f"batch {idx} , total count: {metric.total}")

    result = metrics_collection.compute()
    print(f"metrics calculation results: {result.keys()}")
    print(f"aoa metrics shape: {result[MetricNames.AOA_ERROR_MEAN_RADIAN].shape}")
    # Output: {'mean': tensor([-0.1000, -0.1000, -0.1000, -0.1000]), 'std': tensor([0., 0., 0., 0.])}
    # The mean AoA error is -0.1 for all APs and the standard deviation is 0 for all APs.

    # test metrics collection
