"""Defines the schema for the dataset and other dataclasses used in the project."""
from dataclasses import dataclass, field
from enum import Enum, unique
import torch
import numpy as np
import os
from datetime import datetime
from typing import Optional

@unique
class DatasetKeys(Enum):
    """Keys for each individual dataset h5 file.
    """
    # 2D fft plot of CSI data. Shape is (H, W, n_ap, 1)
    FEATURES_2D: str = 'features_2d'

    # Angle of arrival ground truth of the signal. Shape is (n_ap, 1)
    AOA_GT_LABEL: str = 'aoa_gnd'

    # Location ground truth of the signal. Shape is (2, 1) representing (x, y)
    LOCATION_GT_LABEL: str = 'labels'

    # Time of flight ground truth of the signal. Shape is (n_ap, 1)
    TOF_GT_LABEL: str = 'tof_gnd'

    # Timestamps of sample. Shape is (1)
    TIMESTAMP: str = "timestamps"

    # Velocity of sample. Shape is (2, ) representing (vx, vy)
    VELOCITY: str = "velocity"

    # AP Orientation Offset, shape (n_ap, )
    AP_AOAS: str = 'ap_aoas'

    # AP locations in the dataset. Shape is (n_ap, 2) representing (x, y)
    AP_LOCATIONS: str = 'aps'

    # RSSI of the signal. Shape is (n_ap, )
    RSSI: str = 'rssi'



class APMetadata:
    """Metadata for the WiFi access points in the dataset.
    data read form ap.h5 file. The ap_orientation is not correct in the file. Use hard coded value
    in this class instead.
    """
    def __init__(self, ap_aoas, ap_locs) -> None:
        # AP locations in the dataset. Shape is (n_aps, 2) representing (x, y)
        # self._ap_locations = torch.tensor([[ 0. ,  2. ],
        #                                   [16.8,  7.6],
        #                                   [ 6.4,  7.6],
        #                                   [12.4,  0. ]])

        self._ap_locations = ap_locs
        assert self._ap_locations.shape[1] == 2, "AP locations should be of shape (n_aps, 2)"

        # AP AoA.
        # self._ap_orientations = torch.tensor([0, np.pi, np.pi/2, -np.pi/2])
        self._ap_orientations = ap_aoas
        assert self._ap_locations.shape[0] == self._ap_orientations.shape[0], f"Shape mismatch. AP Location Shape: {self._ap_locations.shape}, AP AoAs Shape: {self._ap_orientations.shape}"

        # cosine of ap orientation
        self._cos_ap_orientations = torch.cos(self.ap_orientations)

        # sine of ap orientation
        self._sin_ap_orientations = torch.sin(self.ap_orientations)

        # number of APs
        self._n_aps = self._ap_locations.shape[0]

    @property
    def ap_locations(self):
        return self._ap_locations

    @property
    def ap_orientations(self):
        return self._ap_orientations

    @property
    def cos_ap_orientations(self):
        return self._cos_ap_orientations

    @property
    def sin_ap_orientations(self):
        return self._sin_ap_orientations

    @property
    def n_aps(self):
        return self._n_aps

    def to_tensor(self):
        """Convert APMetadata to a single tensor.
        
        Returns:
            torch.Tensor: Concatenated tensor containing:
                - ap_locations: shape (n_aps, 2) 
                - ap_orientations: shape (n_aps,)
                - cos_ap_orientations: shape (n_aps,)
                - sin_ap_orientations: shape (n_aps,)
            Total shape: (n_aps, 5)
        """
        return torch.cat([
            self._ap_locations,  # (n_aps, 2)
            self._ap_orientations.unsqueeze(1),  # (n_aps, 1)
            self._cos_ap_orientations.unsqueeze(1),  # (n_aps, 1)
            self._sin_ap_orientations.unsqueeze(1)   # (n_aps, 1)
        ], dim=1)  # (n_aps, 5)
    
    @classmethod
    def from_tensor(cls, tensor):
        """Create APMetadata from a tensor.
        
        Args:
            tensor (torch.Tensor): Tensor of shape (n_aps, 5) or (batch_size, n_aps, 5)
        
        Returns:
            APMetadata or list of APMetadata
        """
        # Check if batched
        if tensor.dim() == 3:
            # Return list of APMetadata objects, one per batch sample
            return [cls._from_single_tensor(tensor[i]) for i in range(tensor.shape[0])]
        else:
            # Single sample
            return cls._from_single_tensor(tensor)
        
    @classmethod
    def _from_single_tensor(cls, tensor):
        """Helper to create APMetadata from a single 2D tensor."""
        if tensor.shape[1] != 5:
            raise ValueError(f"Expected tensor shape (n_aps, 5), got {tensor.shape}")
        
        ap_locs = tensor[:, :2]  # (n_aps, 2)
        ap_aoas = tensor[:, 2]   # (n_aps,)
        
        return cls(ap_aoas=ap_aoas, ap_locs=ap_locs)

@dataclass
class LoggerParameters:
    """Dataclass to hold parameters for the logger.

    Ref: https://lightning.ai/docs/pytorch/1.9.1/api/pytorch_lightning.loggers.comet.html?highlight=comet
    """
    # experiment name
    experiment_name: str

    # api key. Do not set this value in LoggerParameters constructor. Instead, set the COMET_API_KEY environment variable.
    api_key: str = field(init=False)

    # workspace. Do not set this value in LoggerParameters constructor. Instead, set the COMET_WORKSPACE environment variable.
    workspace: str = field(init=False)

    # project name
    project_name: str = "dloc"

    # directory to save log
    save_dir: str = "./logs"


    def __post_init__(self):
        if os.getenv('COMET_API_KEY'):
            self.api_key = os.getenv('COMET_API_KEY')
        else:
            raise ValueError("Please set the COMET_API_KEY environment variable by running `export COMET_API_KEY=<your_api_key>`.")

        if os.getenv('COMET_WORKSPACE'):
            self.workspace = os.getenv('COMET_WORKSPACE')
        else:
            raise ValueError("Please set the COMET_WORKSPACE environment variable by running `export COMET_WORKSPACE=<your_workspace_name>`.")

        # Get the current GPU number (if available)
        device_name = f"gpu_{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

        # add datetime to experiment name
        self.experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}:{device_name}:{self.experiment_name}"

class DatasetType:
    train: str = "train"
    val: str = "val"
    test: str = "test"
    
@dataclass
class ModelOutput:
    # cos of aoa in AP frame, shape (batch_size, n_aps)
    cos_aoa: torch.Tensor

    # sin of aoa in AP frame, shape (batch_size, n_aps)
    sin_aoa: torch.Tensor

    # AoA prediction in AP frame, shape (batch_size, n_aps)
    aoa: torch.Tensor

    # location prediction, shape (batch_size, 2)
    location: torch.Tensor

    # confidence for aoa prediction, shape (batch_size, n_aps)
    confidence: Optional[torch.Tensor] = None


@dataclass
class GTlabel:
    # cos of aoa in AP frame, shape (batch_size, n_aps)
    cos_aoa: torch.Tensor

    # sin of aoa in AP frame, shape (batch_size, n_aps)
    sin_aoa: torch.Tensor

    # aoa ground truth value in AP frame, shape (batch_size, n_aps)
    aoa: torch.Tensor

    # location prediction, shape (batch_size, 2)
    location: torch.Tensor

    # velocity, shape (batch_size, 2)
    velocity: torch.Tensor

    # timestamps, shape (batch_size, )
    timestamps: torch.Tensor


@dataclass
class LossTerms:
    # total loss
    total_loss: torch.Tensor

    # loss for cos of AoA
    cos_loss: torch.Tensor

    # loss for sin of AoA
    sin_loss: torch.Tensor

    # loss for location prediction
    location_loss: torch.Tensor


@dataclass
class AoAVisualizationSample:
    # AoA tof plots, shape (n_aps, H, W)
    aoa_tof_plots: torch.Tensor

    # AoA ground truth in radian, shape (n_aps,)
    aoa_gt: torch.Tensor

    # AoA prediction in radian, shape (n_aps,)
    aoa_pred: torch.Tensor

    # confidence prediction, shape (n_aps,). This is optional value.
    confidence: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Check that the first dimension of all tensors match
        n_aps = self.aoa_tof_plots.shape[0]
        if not (self.aoa_gt.shape[0] == n_aps and
                self.aoa_pred.shape[0] == n_aps and
                (self.confidence is None or self.confidence.shape[0] == n_aps)):
            error_message = (
                f"The first dimension of aoa_gt, aoa_pred, and confidence (if provided) must match the first dimension of aoa_tof_plots.\n"
                f"aoa_tof_plots shape: {self.aoa_tof_plots.shape}\n"
                f"aoa_gt shape: {self.aoa_gt.shape}\n"
                f"aoa_pred shape: {self.aoa_pred.shape}\n"
            )
            if self.confidence is not None:
                error_message += f"confidence shape: {self.confidence.shape}\n"

            raise ValueError(error_message)


@dataclass
class DLocDataSample:
    # CSI data, shape (n_aps, H, W)
    features_2d: torch.Tensor

    # AoA label data in radians, shape (n_aps,)
    aoa_label: torch.Tensor

    # Location label data, xy coordinates, shape (2,)
    location_label: torch.Tensor

    # Velocity data, shape (2, )
    velocity: torch.Tensor

    # Timestamps data, shape (1, )
    timestamps: torch.Tensor

    ap_metadata: torch.Tensor

    # RSSI values, shape (batch_size, n_aps)
    rssi: torch.Tensor = None
    # ap_metadata: APMetadata = field(default_factory=APMetadata)

    def is_empty(self) -> bool:
        """Check if the tensors are empty."""
        return self.features_2d.numel() == 0 or self.aoa_label.numel() == 0 or self.location_label.numel() == 0 or self.velocity.numel() == 0 or self.timestamps.numel() == 0

    def has_nan(self) -> bool:
        """Check if the tensors contain NaN values."""
        return torch.isnan(self.features_2d).any() or torch.isnan(self.aoa_label).any() or torch.isnan(self.location_label).any() or torch.isnan(self.velocity).any() or torch.isnan(self.timestamps).any()


@dataclass
class DLocBatchDataSample(DLocDataSample):
    # CSI data, shape (batch_size, n_aps, H, W)
    features_2d: torch.Tensor

    # AoA label data in radians, shape (batch_size, n_aps)
    aoa_label: torch.Tensor

    # Location label data, xy coordinates, shape (batch_size, 2)
    location_label: torch.Tensor

    # Velocity data, shape (2, 1)
    velocity: torch.Tensor

    # Timestamps data, shape (1, )
    timestamps: torch.Tensor

    # RSSI data, shape (batch_size, n_aps)
    rssi: torch.Tensor = None  


    ap_metadata: APMetadata = field(default_factory=APMetadata)

    def __post_init__(self):
        if self.is_empty():
            return

        # Check that the first dimension of all tensors match
        if not (self.features_2d.shape[0] == self.aoa_label.shape[0] == self.location_label.shape[0] == self.velocity.shape[0] == self.timestamps.shape[0]):
            raise ValueError(
                f"The first dimension of features_2d, aoa_label, location_label, velocity, and timestamps must match.\n"
                f"features_2d shape: {self.features_2d.shape}\n"
                f"aoa_label shape: {self.aoa_label.shape}\n"
                f"location_label shape: {self.location_label.shape}\n"
                f"velocity_label shape: {self.velocity.shape}\n"
                f"timestamps shape: {self.timestamps.shape}\n"
            )

    def get_batch_size(self) -> int:
        """Get the batch size."""
        return self.features_2d.shape[0]

    def partition_support_query_set(self, support_set_pct: float = 0.8) -> tuple:
        """Partition the data into support and query sets.

        Args:
            support_set_pct (float): Percentage of data to be used for the support set.

        Returns:
            tuple: Support and query sets. Both are DLocBatchDataSample objects.
        """
        assert 0 < support_set_pct < 1, "support_set_pct must be between 0 and 1"
        batch_size = self.get_batch_size()
        support_set_size = int(batch_size * support_set_pct)

        # Generate shuffled indices
        indices = torch.randperm(batch_size)

        # Split the indices into support and query sets
        support_indices = indices[:support_set_size]
        query_indices = indices[support_set_size:]

        support_set = self.__class__(
            features_2d=self.features_2d[support_indices],
            aoa_label=self.aoa_label[support_indices],
            location_label=self.location_label[support_indices],
            velocity=self.velocity[support_indices],
            timestamps=self.timestamps[support_indices]
        )
        query_set = self.__class__(
            features_2d=self.features_2d[query_indices],
            aoa_label=self.aoa_label[query_indices],
            location_label=self.location_label[query_indices],
            velocity=self.velocity[query_indices],
            timestamps=self.timestamps[query_indices]
        )
        return support_set, query_set

@dataclass
class DLocBatchSequenceDataSample(DLocDataSample):
    # CSI data, shape (batch_size, sequence_size, n_aps, H, W)
    features_2d: torch.Tensor

    # AoA label data in radians, shape (batch_size, sequence_size, n_aps)
    aoa_label: torch.Tensor

    # Location label data, xy coordinates, shape (batch_size, sequence_size, 2)
    location_label: torch.Tensor

    # Velocity data, shape (batch_size, sequence_size, 2)
    velocity: torch.Tensor

    # Timestamps data, shape (batch_size, sequence_size, 1)
    timestamps: torch.Tensor

    ap_metadata: APMetadata = field(default_factory=APMetadata)

    def __post_init__(self):
        if self.is_empty():
            return

        # Check that the first dimension of all tensors match
        if not (self.features_2d.shape[0] == self.aoa_label.shape[0] == self.location_label.shape[0] == self.velocity.shape[0] == self.timestamps.shape[0]):
            raise ValueError(
                f"The first dimension of features_2d, aoa_label, location_label, velocity, and timestamps must match.\n"
                f"features_2d shape: {self.features_2d.shape}\n"
                f"aoa_label shape: {self.aoa_label.shape}\n"
                f"location_label shape: {self.location_label.shape}\n"
                f"velocity_label shape: {self.velocity.shape}\n"
                f"timestamps shape: {self.timestamps.shape}\n"
            )
        
        # Check that the second dimension of all tensors match
        if not (self.features_2d.shape[1] == self.aoa_label.shape[1] == self.location_label.shape[1] == self.velocity.shape[1] == self.timestamps.shape[1]):
            raise ValueError(
                f"The first dimension of features_2d, aoa_label, location_label, velocity, and timestamps must match.\n"
                f"features_2d shape: {self.features_2d.shape}\n"
                f"aoa_label shape: {self.aoa_label.shape}\n"
                f"location_label shape: {self.location_label.shape}\n"
                f"velocity_label shape: {self.velocity.shape}\n"
                f"timestamps shape: {self.timestamps.shape}\n"
            )

    def get_batch_size(self) -> int:
        """Get the batch size."""
        return self.features_2d.shape[0]
    
    def get_sequence_size(self) -> int:
        """Get the sequence size"""
        return self.features_2d.shape[1]

    def partition_support_query_set(self, support_set_pct: float = 0.8) -> tuple:
        """Partition the data into support and query sets.

        Args:
            support_set_pct (float): Percentage of data to be used for the support set.

        Returns:
            tuple: Support and query sets. Both are DLocBatchDataSample objects.
        """
        assert 0 < support_set_pct < 1, "support_set_pct must be between 0 and 1"
        batch_size = self.get_batch_size()
        support_set_size = int(batch_size * support_set_pct)

        # Generate shuffled indices
        indices = torch.randperm(batch_size)

        # Split the indices into support and query sets
        support_indices = indices[:support_set_size]
        query_indices = indices[support_set_size:]

        support_set = self.__class__(
            features_2d=self.features_2d[support_indices],
            aoa_label=self.aoa_label[support_indices],
            location_label=self.location_label[support_indices],
            velocity=self.velocity[support_indices],
            timestamps=self.timestamps[support_indices]
        )
        query_set = self.__class__(
            features_2d=self.features_2d[query_indices],
            aoa_label=self.aoa_label[query_indices],
            location_label=self.location_label[query_indices],
            velocity=self.velocity[query_indices],
            timestamps=self.timestamps[query_indices]
        )
        return support_set, query_set