"""Utils functions for data loading and processing."""

import os
from typing import List
import h5py
import numpy as np
import subprocess
import sys

from utils.schema import DatasetKeys

def get_n_ap_from_path(data_path: str) -> int:
    """Find the number of access points (APs) from the given data path.

    Args:
        data_path: Path to the data file.

    Returns:
        Number of access points (APs) in the dataset.
    """
    assert os.path.exists(data_path), f"Data file csv {data_path} does not exist."

    parent_dir = os.path.join(os.path.dirname(data_path), "features_aoa")
    ap_file_path = os.path.join(parent_dir, "ap.h5")
    ap_aoas, _ = read_ap_file(ap_file_path)
    return len(ap_aoas)

def read_ap_file(ap_file_path):
    if not os.path.exists(ap_file_path):
        raise FileNotFoundError(f"'ap.h5' file not found in the directory: {ap_file_path}")
    with h5py.File(ap_file_path, "r") as f:
            ap_aoas = np.array(f.get(DatasetKeys.AP_AOAS.value), dtype=np.float32).squeeze()
            ap_locs = np.transpose(np.array(f.get(DatasetKeys.AP_LOCATIONS.value), dtype=np.float32))

    return ap_aoas, ap_locs

def list_and_sort_files(directory: str, extension: str = ".h5") -> List[str]:
    """list and sort files based on their filenames.

    Args:
        directory: Path to the directory containing data files.
        extension: File extension. Defaults to ".h5".

    Returns:
        A list containing absolute paths to the files sorted by their filenames.
    """
    assert os.path.exists(directory), f"Directory {directory} does not exist."

    # Filter files by the given extension
    files = [f for f in os.listdir(directory) if f.endswith(extension)]

    # Sort files based on their filenames
    files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    # Convert to absolute paths
    files = [os.path.abspath(os.path.join(directory, f)) for f in files]

    return files


def get_xy_label_from_file(file_path: str) -> np.ndarray:
    """get xy labels from h5 file

    Args:
        file_path: path to h5 file containing xy labels

    Returns:
        xy labels, shape (2,)
    """
    with h5py.File(file_path, 'r') as f:
        # xy_label_np.shape = (2,)
        location_label_np = np.array(f.get(DatasetKeys.LOCATION_GT_LABEL.value), dtype=np.float32).squeeze()
    return location_label_np


def partition_points_into_n_by_n_grid_cells(points_data: np.ndarray, n: int) -> list:
    """Given an array of points, partition the points into n x n grid cells.

    Args:
        points_data: xy coordinates of the points. Shape is (n_sample, 2).
        n: grid size to partition the points.

    Returns:
        A list of boolean masks, each mask indicating which points fall into a specific grid cell.
    """
    x_min, y_min = np.min(points_data, axis=0)
    x_max, y_max = np.max(points_data, axis=0)

    # Create grid boundaries
    x_bins = np.linspace(x_min, x_max, n + 1)
    y_bins = np.linspace(y_min, y_max, n + 1)

    grid_cell_masks = []
    for x_index in range(n):
        for y_index in range(n):
            x_lower_bound = x_bins[x_index]
            x_upper_bound = x_bins[x_index + 1]
            y_lower_bound = y_bins[y_index]
            y_upper_bound = y_bins[y_index + 1]

            # Get points within the grid
            grid_cell_mask = (points_data[:, 0] >= x_lower_bound) &\
                             (points_data[:, 0] < x_upper_bound) &\
                             (points_data[:, 1] >= y_lower_bound) &\
                             (points_data[:, 1] < y_upper_bound)
            grid_cell_masks.append(grid_cell_mask)
    return grid_cell_masks


def randomly_choose_numbers(K: int, n_numbers: int) -> tuple:
    """
    Randomly choose n integers from 0 (inclusive) to K-1 (inclusive), return the chosen numbers and the rest.

    Args:
        K (int): The upper limit of the range (not inclusive).
        n (int): The number of numbers to choose.

    Returns:
        tuple: Two sorted lists, the first containing the chosen numbers and the second containing the rest (K-n) numbers.
    """
    if n_numbers > K:
        raise ValueError(f"n_numbers cannot be greater than K. n_numbers: {n_numbers}, K: {K}")

    all_numbers = np.arange(K)
    chosen_numbers = np.random.choice(all_numbers, n_numbers, replace=False)
    rest_numbers = np.setdiff1d(all_numbers, chosen_numbers)

    return sorted(chosen_numbers), sorted(rest_numbers)