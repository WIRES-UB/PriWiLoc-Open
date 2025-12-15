# CSI Data Processing for Deep Learning

This repository contains MATLAB scripts for processing Channel State Information (CSI) data for deep learning applications in wireless localization. The code processes raw channel data into features that can be used for angle-of-arrival (AoA) and time-of-flight (ToF) estimation.

## Repository Structure

- **analyze_images.m**: Script for analyzing and visualizing CSI features as images, including ground truth angle-of-arrival and position verification.
- **ap_locations_ref.m**: Reference file containing access point (AP) configurations for various datasets including positions, orientations, and angle-of-arrival settings.
- **channels_to_features.m**: Main processing script that converts raw channel data into 2D feature images for deep learning, saving them as HDF5 files.
- **update_mat_file.m**: Utility script to update AP orientation angles in dataset files.
- **p2slam_channels_to_features.mlx**: Processing script that converts data from P2SLAM into 2D feature images, saving them as HDF5 files.

## Dataset Information

The code supports multiple datasets collected in different environments:
- Indoor lab measurements (July datasets)
- Jacobs Hall measurements (August datasets)

Each dataset contains CSI data from N access points with antenna arrays, capturing wireless propagation data from different locations within the measurement space.

## Usage Instructions

### Converting Channel Data to Features

1. Set the `DATA_SAVE_TOP` variable in `channels_to_features.m` to your desired output directory.\
2. Set the `CHANNELS_LOCATION` variable in `channels_to_features.m` to the directory containing the raw data channels.\ 
3. Select the dataset you want to process by modifying the dataset_number array.
4. Run the script to extract CSI data and convert it to 2D feature images.
5. The script will generate HDF5 files containing:
   - 2D features (angle-of-arrival vs. time-of-flight images)
   - Ground truth angle-of-arrival data
   - Position labels
   - RSSI Labels (optional)
6. If you'd like to inspect the AP angle-of-arrival (ap-aoa), then set the 2nd argument in `extractCSIData` to `true`. You may also comment out the section titled `Calculate GT AoA and Verify` if you don't need to verify the labels. 

### Analyzing Generated Features

1. Configure dataset path and name in `analyze_images.m`.
2. Run the script to:
   - Visualize the feature images for each access point
   - View ground truth AoA lines overlaid on the images
   - Compare estimated vs. actual positions
   - Generate animations across multiple data points (optional)

### Modifying AP Configurations

Use `update_mat_file.m` to update access point orientation angles if needed.

## Feature Format

The extracted features are stored as:
- 2D images with dimensions representing:
  - Angle-of-arrival (theta): -90° to 90° (360 points)
  - Time-of-flight (distance): -10m to 30m (400 points)
- Each image represents the RF propagation pattern from a single access point
- Red lines in visualizations indicate ground truth angle-of-arrival

## Prerequisites

- MATLAB with Signal Processing Toolbox
- HDF5 support

## Reference Functions (Not Shown in Repository)

The code depends on several helper functions that should be in your path:
- `extractCSIData`: Extracts channel state information from dataset files
- `computeAngleOfArrival`: Calculates ground truth angles based on positions
- `lineIntersect2D_slope_point`: Verifies position reconstruction from angles
- `generate_aoa_tof_features`: Converts raw channel data to angle/time features

## Useful Knowledge
- `computeAngleOfArrival` produced the local AoA. To get the global AoA, simply subtract the ap-aoa from the computed aoa. 
- 0 ap-aoa means the antennas are aligned parallel to the positive y-axis with antenna 1 at the top, and antenna 4 on the bottom. 
- The left half (antenna 1) of the AP represents positive angles, the right half (antenna 4) represents negative angles
- If you'd like to use `atan` instead of `atan2`, then for APs that are parallel to the y-axis, always give 0 ap-aoa.
