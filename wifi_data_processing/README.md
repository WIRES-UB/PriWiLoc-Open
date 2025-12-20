# CSI Data Processing for Deep Learning

This repository contains MATLAB scripts for processing Channel State Information (CSI) data for deep learning applications in wireless localization. The code processes raw channel data into features that can be used for angle-of-arrival (AoA) and time-of-flight (ToF) estimation.

## Repository Structure

- **analyze_images.m**: Script for analyzing and visualizing CSI features as images, including ground truth angle-of-arrival and position verification.
- **ap_locations_ref.m**: Reference file containing access point (AP) configurations for various datasets including positions, orientations, and angle-of-arrival settings.
- **channels_to_features_aoa.m**: Main processing script that converts raw channel data from DLoc datasets into 2D feature images for deep learning, saving them as HDF5 files.
- **generate_rloc_data_edit.m**: Processing script that converts RLoc dataset format into the DLoc format and generates 2D feature images, saving them as HDF5 files.
- **update_mat_file.m**: Utility script to convert AP orientation angles from degrees to radians in DLoc dataset files.
- **utils/**: Directory containing helper functions for CSI processing, AoA computation, and feature generation.

## Dataset Information

The code supports two types of datasets:

### DLoc Datasets
- Indoor lab measurements (July datasets)
- Jacobs Hall measurements (August datasets)
- Each dataset contains CSI data from multiple access points with antenna arrays

### RLoc Datasets
- Human-held device WiFi indoor localization dataset
- Four room types: Conference, Laboratory, Lounge, and Office
- Multiple users and interference conditions (with/without interference)

## Usage Instructions

### For DLoc Paper Data

1. **Process the channel data** using `channels_to_features_aoa.m`:
   - Set `DATA_SAVE_TOP` to your desired output directory
   - Set `CHANNELS_LOCATION` to the directory containing the raw channel files
   - Select the dataset(s) to process by modifying the `dataset_number` array
   - Run the script to generate 2D feature images

2. **Verify Data Quality** using `analyze_images.m`:
   - We reccomend you do this step to ensure the data looks correct. 
   - You should see the XY labels match the AoA --> XY labels
   - You can verify ground truth AoA by manually checking the angle of the ground truth XY with respect to each AP


### For RLoc Data

1. **Process RLoc data** using `generate_rloc_data_edit.m`:
   - Set `DATA_SAVE_TOP` to your desired output directory
   - Set `RLOC_DATA_PATH` to the RLoc dataset directory
   - Select the rooms to process by modifying the `rooms` and `room_codes` arrays
   - Run the script to convert RLoc format to DLoc format and generate features

2. **Remove invalid data points**:
   - After processing, remove any files where the ground truth AoA exceeds ±90 degrees
   - This ensures all data points are within the valid angular range

### Analyzing Generated Features

1. Configure dataset path and name in `analyze_images.m`.
2. Run the script to:
   - Visualize the feature images for each access point
   - View ground truth AoA lines overlaid on the images
   - Compare estimated vs. actual positions
   - Generate animations across multiple data points (optional)

## Output Format

The processing scripts generate the following structure for each dataset:

```
dataset_name/
  features_aoa/
    ap.h5                    # AP locations and orientations
    ind/
      1.h5                   # Individual sample files
      2.h5
      ...
```

Each sample HDF5 file contains:
- `/features_2d`: 2D feature image (400 x 360 x N_AP)
  - Dimension 1: Distance/ToF (-10m to 30m for DLoc, 0 to 100m for RLoc)
  - Dimension 2: Angle-of-arrival (-90° to 90°, 360 points)
  - Dimension 3: Access point index
- `/aoa_gnd`: Ground truth angle-of-arrival for each AP (radians)
- `/labels`: Ground truth (x, y) position (meters)
- `/velocity`: Velocity vector (for DLoc datasets with timestamps)
- `/timestamps`: Time of measurement (for DLoc datasets)
- `/rssi`: Received Signal Strength Indicator for each AP

The `ap.h5` file contains:
- `/aps`: AP locations as (N_AP x 2) array with (x, y) coordinates
- `/ap_aoas`: AP orientation angles as (N_AP,) array in radians

## Prerequisites

- MATLAB R2019b or later
- Signal Processing Toolbox
- HDF5 support (included in MATLAB)

## Helper Functions (in utils/ directory)

The processing pipeline uses the following helper functions:
- `extractCSIData.m`: Extracts channel state information from dataset files
- `computeAngleOfArrival.m`: Calculates ground truth global angles based on positions (for DLoc)
- `computeAngleOfArrivalRLoc.m`: Calculates ground truth angles for RLoc format
- `lineIntersect2D_slope_point.m`: Verifies position reconstruction from angles
- `generate_aoa_tof_features.m`: Converts raw channel data to 2D angle/time features
- `getAngle.m`: Utility for angle calculations

## Important Notes

### Angle Conventions
- **Valid AoA Range**: Only angles between -90° and +90° are physically measurable by the AP antenna arrays
- **AP Orientation (ap_aoa)**:
  - 0 radians: Antennas aligned parallel to positive y-axis (antenna 1 at top, antenna 4 at bottom)
  - Positive angles: Counterclockwise rotation from y-axis
  - Negative angles: Clockwise rotation from y-axis
- **Local vs Global AoA**:
  - `computeAngleOfArrival` produces global AoA (relative to coordinate system)
  - Local AoA (relative to AP orientation) = global AoA - ap_aoa
- **Antenna Arrangement**:
  - Left half of AP array (antenna 1) detects positive angles
  - Right half of AP array (antenna 4) detects negative angles
  - 4 antennas with 2.6cm spacing

### Data Quality
- Remove samples with ground truth AoA > ±90° as these are outside the valid detection range
- The verification plot in `channels_to_features_aoa.m` helps identify potential labeling issues
- RLoc data is automatically converted to match DLoc format conventions
