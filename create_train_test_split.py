#!/usr/bin/env python3
"""
Script to partition .h5 files into train and test CSV files.
Every 5th file goes to test.csv, the rest go to train.csv.
"""

import numpy as np
import h5py
import os
import glob
import argparse
from tqdm import tqdm


def get_numeric_part(filename):
    """Extract numeric part from filename for sorting."""
    basename = os.path.basename(filename)
    name_part = os.path.splitext(basename)[0]
    try:
        return int(name_part)
    except ValueError:
        print(f"Warning: Could not parse numeric part from filename: {basename}")
        return -1  # Handle cases where filename is not purely numeric


def load_and_validate_files(data_dirs):
    """
    Load all .h5 files from data_dirs and validate they have proper labels.
    
    Args:
        data_dirs: List of directories containing .h5 files
        
    Returns:
        numpy array of valid file paths
    """
    print("--- Starting Data Loading ---")
    
    all_h5_files = []
    
    # Collect files from all directories
    for data_dir in data_dirs:
        print(f"\nScanning directory: {data_dir}")
        file_pattern = os.path.join(data_dir, '*.h5')
        h5_files = glob.glob(file_pattern)
        
        if not h5_files:
            print(f"  Warning: No .h5 files found in {data_dir}")
        else:
            print(f"  Found {len(h5_files)} .h5 files")
            all_h5_files.extend(h5_files)
    
    if not all_h5_files:
        print(f"\nError: No .h5 files found in any of the provided directories")
        return np.array([])
    
    # Sort files numerically
    sorted_files = sorted(all_h5_files, key=get_numeric_part)
    # Filter out any files that couldn't be parsed
    sorted_files = [f for f in sorted_files if get_numeric_part(f) != -1]
    num_files = len(sorted_files)
    
    print(f"\nTotal files found across all directories: {num_files}")
    
    # Validate files have proper labels
    print("Validating files and labels...")
    valid_files = []
    
    for file_path in tqdm(sorted_files):
        try:
            with h5py.File(file_path, 'r') as f:
                if '/labels' in f:
                    label = f['/labels'][:].flatten()[:2]  # Ensure it's shape (2,)
                    if label.shape == (2,) and not np.isnan(label).any():
                        valid_files.append(file_path)
                    else:
                        print(f"Skipping file {file_path}: Invalid label shape or NaN value.")
                else:
                    print(f"Skipping file {file_path}: '/labels' dataset not found.")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    files_array = np.array(valid_files)
    print(f"Loaded {len(files_array)} valid files successfully.")
    print("--- Finished Data Loading ---\n")
    
    return files_array


def create_train_test_split(files_array, train_csv="train.csv", test_csv="test.csv"):
    """
    Split files into train and test sets.
    Every 5th file goes to test, the rest go to train.
    
    Args:
        files_array: Array of file paths
        train_csv: Output path for training set
        test_csv: Output path for test set
    """
    print("--- Starting Train/Test Partitioning ---")
    
    if files_array.size == 0:
        print("No files to partition. Exiting.")
        return
    
    train_files = []
    test_files = []
    
    for idx, file_path in enumerate(files_array):
        if (idx + 1) % 5 == 0:  # Every 5th file (1-indexed: 5, 10, 15, ...)
            test_files.append(file_path)
        else:
            train_files.append(file_path)
    
    print(f"Total files: {len(files_array)}")
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    
    # Save to CSV files (one path per line, no header)
    if train_files:
        with open(train_csv, 'w') as f:
            for file_path in train_files:
                f.write(f"{file_path}\n")
        print(f"Saved train partition to: {train_csv}")
    else:
        print("Warning: No train files to save.")
    
    if test_files:
        with open(test_csv, 'w') as f:
            for file_path in test_files:
                f.write(f"{file_path}\n")
        print(f"Saved test partition to: {test_csv}")
    else:
        print("Warning: No test files to save.")
    
    print("--- Finished Train/Test Partitioning ---")


def main():
    parser = argparse.ArgumentParser(
        description="Partition .h5 files into train and test CSV files. "
                    "Every 5th file goes to test, the rest go to train. "
                    "Supports multiple data directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single directory
  python create_train_test_split.py /path/to/data

  # Multiple directories
  python create_train_test_split.py /path/to/data1 /path/to/data2 /path/to/data3

  # Multiple directories with custom output files
  python create_train_test_split.py /path/to/data1 /path/to/data2 --train-csv my_train.csv --test-csv my_test.csv
"""
    )
    parser.add_argument(
        'data_dirs',
        type=str,
        nargs='+',
        help='One or more directories containing .h5 files'
    )
    parser.add_argument(
        '--train-csv',
        type=str,
        default='train.csv',
        help='Output path for training set CSV (default: train.csv)'
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default='test.csv',
        help='Output path for test set CSV (default: test.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate all data directories exist
    invalid_dirs = []
    for data_dir in args.data_dirs:
        if not os.path.isdir(data_dir):
            invalid_dirs.append(data_dir)
    
    if invalid_dirs:
        print("Error: The following directories were not found:")
        for d in invalid_dirs:
            print(f"  - {d}")
        return
    
    print(f"Processing {len(args.data_dirs)} director{'y' if len(args.data_dirs) == 1 else 'ies'}...")
    
    # Load and validate files
    files_array = load_and_validate_files(args.data_dirs)
    
    # Create train/test split
    create_train_test_split(files_array, args.train_csv, args.test_csv)


if __name__ == "__main__":
    main()

