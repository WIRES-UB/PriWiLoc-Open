#!/usr/bin/env python3
import csv, glob, os, re, random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless systems
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import pandas as pd
from pathlib import Path

# --------------------------
# Office datasets only
# --------------------------
OFFICE_GLOBS = {
    "Office_user1_w":   "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user1_w/features_aoa/ind/*.h5",
    "Office_user1_wo":  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user1_wo/features_aoa/ind/*.h5",
    "Office_user2_w":   "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user2_w/features_aoa/ind/*.h5",
    "Office_user2_wo":  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user2_wo/features_aoa/ind/*.h5",
    "Office_user3_w":   "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user3_w/features_aoa/ind/*.h5",
    "Office_user3_wo":  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user3_wo/features_aoa/ind/*.h5",
    "Office_user4_w":   "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user4_w/features_aoa/ind/*.h5",
    "Office_user4_wo":  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user4_wo/features_aoa/ind/*.h5",
    "Office_user5_w":   "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user5_w/features_aoa/ind/*.h5",
    "Office_user5_wo":  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user5_wo/features_aoa/ind/*.h5",
}

# Office room AP configuration
OFFICE_CONFIG = {
    'ap_locations': np.array([[5.7, 3.4], [1.4, 9.2], [3.2, -0.3], [-1.5, 4.8]]),
    'ap_names': ['sRE22', 'sRE5', 'sRE6', 'sRE7'],
    'title': 'Office Room'
}

TRAIN_OUT = "train_office_full_filtered.csv"
TEST_OUT = "test_office_rand_filtered.csv"

# --------------------------
# Helpers
# --------------------------
_num = re.compile(r"(\d+)(?=\.h5$)")
def numeric_key(path: str):
    """Sort by trailing integer in filename (e.g., .../ind/123.h5)."""
    m = _num.search(os.path.basename(path))
    return int(m.group(1)) if m else path

def get_room_bounds():
    """Calculate rectangular bounds based on AP locations"""
    ap_locations = OFFICE_CONFIG['ap_locations']
    x_min = np.min(ap_locations[:, 0])
    x_max = np.max(ap_locations[:, 0])
    y_min = np.min(ap_locations[:, 1])
    y_max = np.max(ap_locations[:, 1])
    return x_min, x_max, y_min, y_max

def get_out_of_bounds_files(dataset_name, pattern):
    """Get set of out-of-bounds file basenames for a dataset"""
    base_dir = Path(pattern).parent.parent.parent  # Go up from /features_aoa/ind/*
    out_of_bounds_dir = base_dir / "out_of_bounds" / "features_aoa" / "ind"
    
    if not out_of_bounds_dir.exists():
        return set()
    
    oob_files = glob.glob(str(out_of_bounds_dir / "*.h5"))
    return {os.path.basename(f) for f in oob_files}

def get_filtered_files(name, pattern):
    """Get only in-bounds files, excluding out-of-bounds samples"""
    files = sorted(glob.glob(pattern), key=numeric_key)
    if not files:
        print(f"[Error] No files found for dataset '{name}' - skipping")
        return []
    
    oob_basenames = get_out_of_bounds_files(name, pattern)
    if not oob_basenames:
        print(f"[{name}] No out-of-bounds directory found, keeping all {len(files)} files")
        return files
    
    filtered_files = [f for f in files if os.path.basename(f) not in oob_basenames]
    print(f"[{name}] Filtered dataset: {len(filtered_files)} in-bounds (removed {len(files)-len(filtered_files)})")
    return filtered_files

def collect_filtered_files(globs):
    """Collect all filtered files and verify they're in-bounds"""
    all_files = []
    report_lines = []
    
    for name, pattern in globs.items():
        files = get_filtered_files(name, pattern)
        if not files:
            print(f"[Error] No files found for dataset '{name}' - skipping")
            continue
        
        all_files.extend(files)
        report_lines.append(f"{name}: {len(files)} filtered samples")
    
    print("\n=== Filtered Dataset Summary ===")
    for line in report_lines:
        print(line)
    print(f"Total filtered files: {len(all_files)}")
    
    return all_files

def load_labels_from_csv(csv_file):
    """Load XY labels from files listed in CSV"""
    labels_list = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                h5_file = row[0]
                try:
                    with h5py.File(h5_file, 'r') as f:
                        labels = f['/labels'][:]
                        labels_list.append(labels)
                except Exception as e:
                    print(f"[Warning] Error reading {h5_file}: {e}")
                    continue
    return np.array(labels_list) if labels_list else np.array([])

def plot_train_dataset_sanity_check(train_csv, save_dir):
    """Plot ground truth XY coordinates from train CSV with bounds overlay"""
    print(f"\nCreating sanity check plot for training dataset...")
    train_labels = load_labels_from_csv(train_csv)
    if len(train_labels) == 0:
        print("[Error] No labels loaded from training CSV")
        return None
    
    print(f"Loaded {len(train_labels)} training samples for visualization")
    ap_locations = OFFICE_CONFIG['ap_locations']
    x_min, x_max, y_min, y_max = get_room_bounds()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.scatter(train_labels[:, 0], train_labels[:, 1],
               c='green', alpha=0.7, s=30,
               label=f'Training Data ({len(train_labels)} samples)')
    ax.scatter(ap_locations[:, 0], ap_locations[:, 1],
               c='black', s=200, marker='^', label='Access Points', zorder=5)
    
    for i, (x, y) in enumerate(ap_locations):
        ax.annotate(f'AP{i+1}\n({OFFICE_CONFIG["ap_names"][i]})',
                    (x, y), xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=3, edgecolor='red', facecolor='none',
                             linestyle='--', label='AP Bounds')
    ax.add_patch(rect)
    
    ax.text(0.02, 0.98, f'Bounds: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]\n'
                        f'Area: {(x_max - x_min) * (y_max - y_min):.1f} m²',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    ax.set_xlabel('X (meters)', fontsize=14)
    ax.set_ylabel('Y (meters)', fontsize=14)
    ax.set_title('Office Room - Training Dataset Sanity Check\n(Filtered In-Bounds Data Only)', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_aspect('equal')
    
    margin = 1.0
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    save_path = Path(save_dir) / 'office_train_sanity_check.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sanity check plot saved to: {save_path}")
    plt.close(fig)
    
    x_coords, y_coords = train_labels[:, 0], train_labels[:, 1]
    within_bounds = ((x_coords >= x_min) & (x_coords <= x_max) &
                     (y_coords >= y_min) & (y_coords <= y_max))
    violations = np.sum(~within_bounds)
    if violations > 0:
        print(f"[ERROR] {violations}/{len(train_labels)} training points are outside bounds!")
        print(f"Violating points (first 5): {train_labels[~within_bounds][:5]}")
    else:
        print(f"[OK] All {len(train_labels)} training points are within bounds ✓")
    return save_path

# --------------------------
# Main
# --------------------------
def main():
    random.seed(42)
    print("=== Office Room Dataset Processing ===")
    all_files = collect_filtered_files(OFFICE_GLOBS)
    if not all_files:
        print("[Error] No files collected. Check bounds filtering status.")
        return
    
    print(f"Total office files (filtered): {len(all_files)}")
    train_all = list(all_files)
    test_size = int(0.2 * len(all_files))
    test_all = random.sample(all_files, test_size)
    
    with open(TRAIN_OUT, "w", newline="") as f:
        w = csv.writer(f)
        for p in train_all: 
            w.writerow([p])
    
    with open(TEST_OUT, "w", newline="") as f:
        w = csv.writer(f)
        for p in test_all: 
            w.writerow([p])
    
    print(f"\nWrote {len(train_all)} train samples -> {TRAIN_OUT}")
    print(f"Wrote {len(test_all)} test samples -> {TEST_OUT}")
    
    plot_path = plot_train_dataset_sanity_check(TRAIN_OUT, ".")
    print(f"\n=== Office Room Processing Complete ===")
    print(f"Training CSV: {TRAIN_OUT}")
    print(f"Testing CSV: {TEST_OUT}")
    print(f"Sanity check plot: {plot_path}")

if __name__ == "__main__":
    main()
