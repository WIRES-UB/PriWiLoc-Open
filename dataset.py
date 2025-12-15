"""Dataset class for DLocV2 dataset. Load aoa-tof plots and groundtruth labels."""
import torch
from utils.schema import DatasetType
from torch.utils.data import Dataset
from utils.data_utils import list_and_sort_files, read_ap_file
from pathlib import Path
import numpy as np
import h5py
from typing import Union, List, Optional, Callable, Sequence
from utils.schema import APMetadata, DatasetKeys, DLocDataSample, DLocBatchDataSample
import os
import re

class DLocDatasetV2(Dataset):
    """Dataset class for DLocV2 dataset."""

    def __init__(self, data_file_csv: str, transform: Optional[Callable] = None, shuffle=False, type=DatasetType.train) -> None:
        """Constructor for DLocDatasetV2.
        Args:
            data_file_csv: Path to the csv file containing the data files.
            transform: Torch transform to transform aoa-tof plot. Defaults to None.
            shuffle: Whether to shuffle the data files.
            type: Dataset type (train, val, or test).
        """
        self.shuffle = shuffle
        self.data_files_list = self._get_data(data_file_csv)
        self.transform = transform
        self.type = type
        self.datasets = {}
        self.dataset_names_list = []  # Track dataset names for each sample
        
        # Extract dataset names for all files
        self._extract_all_dataset_names()

    def _extract_all_dataset_names(self):
        """Extract dataset names for all files in the dataset."""
        self.dataset_names_list = []
        for file_path in self.data_files_list:
            dataset_name = self.extract_dataset_name(file_path)
            self.dataset_names_list.append(dataset_name)
            
            # Track unique datasets
            if dataset_name not in self.datasets:
                self.datasets[dataset_name] = len(self.datasets) + 1


    def __len__(self):
        return len(self.data_files_list)

    def __getitem__(self, idx: int) -> DLocDataSample:
        """Return data and ground truth given an index."""
        data_path = self.data_files_list[idx]
        dataset_name = self.dataset_names_list[idx]
        
        # Look for 'ap.h5' file in the parent directory of 'ind'
        features_aoa_dir = os.path.dirname(os.path.dirname(data_path))
        ap_file_path = os.path.join(features_aoa_dir, "ap.h5")
        ap_aoas, ap_locs = read_ap_file(ap_file_path)
        
        ap_aoas = torch.tensor(ap_aoas).squeeze()
        ap_locs = torch.tensor(ap_locs)

        with h5py.File(data_path, "r") as f:
            features_2d_np = np.transpose(np.array(f.get(DatasetKeys.FEATURES_2D.value), dtype=np.float32)).squeeze()
            aoa_label_np = np.array(f.get(DatasetKeys.AOA_GT_LABEL.value), dtype=np.float32).squeeze()
            location_label_np = np.array(f.get(DatasetKeys.LOCATION_GT_LABEL.value), dtype=np.float32).squeeze()
            velocity_np = np.array(f.get(DatasetKeys.VELOCITY.value), dtype=np.float32).squeeze()
            timestamp_np = np.array(f.get(DatasetKeys.TIMESTAMP.value), dtype=np.float32).squeeze()
            rssi_np = np.array(f.get(DatasetKeys.RSSI.value), dtype=np.float32).squeeze()

        # Ensure all AP-related arrays have the same length
        n_aps = len(ap_aoas)
        
        # Pad or truncate features_2d to match n_aps
        if features_2d_np.shape[0] < n_aps:
            pad_size = n_aps - features_2d_np.shape[0]
            features_2d_np = np.pad(features_2d_np, ((0, pad_size), (0, 0), (0, 0)), constant_values=0)
        elif features_2d_np.shape[0] > n_aps:
            features_2d_np = features_2d_np[:n_aps]
        
        # Pad or truncate aoa_label to match n_aps
        if len(aoa_label_np) < n_aps:
            aoa_label_np = np.pad(aoa_label_np, (0, n_aps - len(aoa_label_np)), constant_values=0)
        elif len(aoa_label_np) > n_aps:
            aoa_label_np = aoa_label_np[:n_aps]
        
        # Pad or truncate rssi to match n_aps - THIS IS THE KEY FIX
        if len(rssi_np) < n_aps:
            rssi_np = np.pad(rssi_np, (0, n_aps - len(rssi_np)), constant_values=-100)
        elif len(rssi_np) > n_aps:
            rssi_np = rssi_np[:n_aps]

        ap_metadata = APMetadata(ap_aoas=ap_aoas, ap_locs=ap_locs)
        
        # Convert to torch Tensors AFTER padding
        features_2d = torch.tensor(features_2d_np)
        aoa_label = torch.tensor(aoa_label_np)
        location_label = torch.tensor(location_label_np)
        velocity = torch.tensor(velocity_np)
        timestamps = torch.tensor(timestamp_np)
        rssi = torch.tensor(rssi_np)  # This will now be length 4

        if self.transform:
            features_2d = self.transform(features_2d)

        return DLocDataSample(
            features_2d=features_2d,
            aoa_label=aoa_label,
            location_label=location_label,
            velocity=velocity,
            timestamps=timestamps,
            ap_metadata=ap_metadata,
            rssi=rssi
        )

    @staticmethod
    def extract_dataset_name(file_path: str) -> str:
        """Extract dataset name (e.g., 'jacobs_july28') from file path.
        
        Args:
            file_path: Full path to data file
            
        Returns:
            Dataset name extracted from the path
        """
        path = Path(file_path)
        # Navigate up to find the dataset directory
        # Path structure: .../DLocData/jacobs_July28/features_aoa/ind/file.h5
        # We want to get 'jacobs_July28'
        
        # Method 1: Using path parts
        path_parts = path.parts
        try:
            # Find the index of 'DLocData' and get the next part
            dloc_data_idx = path_parts.index('DLocData')
            if dloc_data_idx + 1 < len(path_parts):
                return path_parts[dloc_data_idx + 1]
        except ValueError:
            pass
        
        # Method 2: Using string splitting as fallback
        if 'DLocData' in file_path:
            parts = file_path.split('DLocData/')
            if len(parts) > 1:
                dataset_part = parts[1].split('/')[0]
                return dataset_part
        
        # Method 3: Extract from any path structure
        # Look for pattern: something_something (like jacobs_july28)
        match = re.search(r'([a-zA-Z]+_[a-zA-Z0-9]+)', file_path)
        if match:
            return match.group(1)
        
        return "unknown_dataset"

    def _get_data(self, data_paths: str) -> List[str]:
        """Gets the data from either csv or directory depending on path
        **NO SUPPORT FOR MULTIPLE DATASETS YET**
        """
        data_path_obj = Path(data_paths)
        if data_path_obj.suffix == ".csv":
            return self._get_all_data_from_csv(data_paths)
        return self._get_all_data_from_directory(data_paths)

    def _get_all_data_from_directory(self, data_paths: Union[List[str], str]) -> List[str]:
        """Get all data files from the given data paths.
        Args:
            data_paths: List of data directories that contain individual data files.
                       Can also be a single data directory.
        Returns:
            A list containing absolute paths to all data files sorted by their filenames.
        """
        if isinstance(data_paths, str):
            if "ind" in data_paths:
                return list_and_sort_files(data_paths)
            else:
                return list_and_sort_files(os.path.join(data_paths, "ind"))

        data_files_list = []
        for data_path in set(data_paths):
            data_files_list += list_and_sort_files(data_path)

        if not self.shuffle:
            data_files_list.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        return data_files_list

    def _get_all_data_from_csv(self, data_file_csv: str) -> List[str]:
        """Get all data files from the given csv file.
        Args:
            data_file_csv: Path to the csv file containing the data files.
        Returns:
            A list containing absolute paths to all data files sorted by their filenames.
        """
        assert os.path.exists(data_file_csv), f"Data file csv {data_file_csv} does not exist."

        with open(data_file_csv, "r") as f:
            data_files_list = [line.strip() for line in f.readlines()]

        if len(data_files_list) == 0:
            raise ValueError(f"No data files found in {data_file_csv}.")

        if not self.shuffle:
            data_files_list.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        return data_files_list

    @staticmethod
    def collate_fn(batch: Sequence[DLocBatchDataSample]) -> DLocBatchDataSample:
        """Instruct data loader how to collate data samples into a batch.
        Args:
            batch: batch data samples.
        Returns:
            A DLocBatchDataSample dataclass containing the collated data.
        """
        max_n_aps = max(sample.features_2d.shape[0] for sample in batch)
        batch_size = len(batch)

        h, w = batch[0].features_2d.shape[1], batch[0].features_2d.shape[2]

        # Stack the features_2d tensors along the batch dimension
        features_2d = torch.stack([sample.features_2d for sample in batch], dim=0)
        # Stack the aoa_label tensors along the batch dimension
        aoa_label = torch.stack([sample.aoa_label for sample in batch], dim=0)
        # Stack the location_label tensors along the batch dimension
        location_label = torch.stack([sample.location_label for sample in batch], dim=0)
        # Stack the velocity tensors along the batch dimension
        velocity = torch.stack([sample.velocity for sample in batch], dim=0)
        # Stack the timestamps tensors along the batch dimension
        timestamps = torch.stack([sample.timestamps for sample in batch], dim=0)
        
        ap_metadata = torch.stack([sample.ap_metadata.to_tensor() for sample in batch], dim=0)
        # Stack the rssi tensors along the batch dimension
        rssi = torch.stack([sample.rssi for sample in batch], dim=0)

        # Return a DLocBatchDataSample object
        return DLocBatchDataSample(
            features_2d=features_2d,
            aoa_label=aoa_label,
            location_label=location_label,
            velocity=velocity,
            timestamps=timestamps,
            ap_metadata=ap_metadata,
            rssi=rssi
        )

    # @staticmethod
    # def collate_fn(batch: Sequence[DLocBatchDataSample]) -> DLocBatchDataSample:
    #     """Instruct data loader how to collate data samples into a batch.
    #     Handles variable number of APs by padding to max_n_aps in the batch.
    #     """
    #     # Find the maximum number of APs in this batch for each field
    #     max_n_aps_features = max(sample.features_2d.shape[0] for sample in batch)
    #     max_n_aps_aoa = max(sample.aoa_label.shape[0] for sample in batch)
    #     max_n_aps_rssi = max(sample.rssi.shape[0] for sample in batch)
    #     max_n_aps_metadata = max(sample.ap_metadata.to_tensor().shape[0] for sample in batch)
        
    #     # Use the maximum across all fields
    #     max_n_aps = max(max_n_aps_features, max_n_aps_aoa, max_n_aps_rssi, max_n_aps_metadata)
        
    #     batch_size = len(batch)
        
    #     # Get dimensions from first sample
    #     h, w = batch[0].features_2d.shape[1], batch[0].features_2d.shape[2]
        
    #     # Initialize tensors with zeros (padding)
    #     features_2d = torch.zeros(batch_size, max_n_aps, h, w)
    #     aoa_label = torch.zeros(batch_size, max_n_aps)
    #     rssi = torch.zeros(batch_size, max_n_aps)
        
    #     # Initialize with correct sizes for non-padded fields
    #     location_label = torch.stack([sample.location_label for sample in batch], dim=0)
    #     velocity = torch.stack([sample.velocity for sample in batch], dim=0)
    #     timestamps = torch.stack([sample.timestamps for sample in batch], dim=0)
        
    #     # Pad ap_metadata to max_n_aps (assuming 5 features per AP from to_tensor())
    #     ap_metadata = torch.zeros(batch_size, max_n_aps, 5)
        
    #     # Fill in actual data (non-padded regions)
    #     for i, sample in enumerate(batch):
    #         n_aps_features = sample.features_2d.shape[0]
    #         n_aps_aoa = sample.aoa_label.shape[0]
    #         n_aps_rssi = sample.rssi.shape[0]
    #         n_aps_metadata = sample.ap_metadata.to_tensor().shape[0]
            
    #         features_2d[i, :n_aps_features, :, :] = sample.features_2d
    #         aoa_label[i, :n_aps_aoa] = sample.aoa_label
    #         rssi[i, :n_aps_rssi] = sample.rssi
    #         ap_metadata[i, :n_aps_metadata, :] = sample.ap_metadata.to_tensor()

    #     return DLocBatchDataSample(
    #         features_2d=features_2d,
    #         aoa_label=aoa_label,
    #         location_label=location_label,
    #         velocity=velocity,
    #         timestamps=timestamps,
    #         ap_metadata=ap_metadata,
    #         rssi=rssi
    #     )