# PriWiLoc-Open

A PyTorch Lightning-based deep learning framework for WiFi-based indoor localization using Channel State Information (CSI) with federated learning capabilities. The system processes angle-of-arrival (AoA) and time-of-flight (ToF) features from multiple access points to predict device locations.

## Project Structure

### Core Directories

- **`configs/`** - Hydra configuration files for all aspects of training
  - `config.yaml` - Main configuration file with experiment settings
  - `dataset/` - Dataset configurations (batch size, data paths, number of APs)
  - `model/` - Model architecture parameters (dropout, channels, federated learning settings)
  - `trainer/` - Training configurations (accelerator, devices, callbacks, checkpoints)
  - `logger/` - Logging backends (wandb, tensorboard, comet, csv, none)

- **`models/`** - Neural network model definitions
  - `model.py` - Base TrigAOAResNetModel with ResNet34 encoder for AoA prediction
  - `federated_learning.py` - FederatedLearningModel extending base model with parameter averaging
  - `model_utils.py` - Helper functions for loss computation and geometric calculations

- **`callbacks/`** - PyTorch Lightning callbacks
  - `visualization_callback.py` - Visualizes AoA predictions and localization results during training

- **`utils/`** - Utility functions and helper modules
  - `config_hydra.py` - Configuration schema and validation
  - `data_utils.py` - Data loading and preprocessing utilities
  - `geometry_utils.py` - Geometric calculations for angle and position estimation
  - `ray_intersection_solver.py` - Solves ray intersection for triangulation-based localization
  - `plot_utils.py` - Plotting utilities for visualization
  - `logger_factory.py` - Factory for creating different logger instances
  - `schema.py` - Data structures and type definitions
  - `tests/` - Unit tests for utility functions

- **`wifi_data_processing/`** - MATLAB scripts for preprocessing raw CSI data
  - Converts raw channel data into 2D AoA-ToF feature images
  - Generates HDF5 files with features, ground truth labels, and AP metadata
  - See `wifi_data_processing/README.md` for detailed usage

- **`outputs/`** - Training outputs (logs, checkpoints) organized by timestamp

- **`lightning_logs/`** - PyTorch Lightning default logs

### Core Files

- **`main.py`** - Main training script with Hydra configuration
- **`data_module.py`** - PyTorch Lightning DataModule for loading datasets
- **`dataset.py`** - Dataset class for loading HDF5 files with AoA-ToF features
- **`metrics_calculator.py`** - Custom metrics for AoA accuracy, location accuracy, and RSSI

## Prerequisites

### Python Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install pytorch-lightning hydra-core omegaconf h5py pandas numpy scipy matplotlib python-dotenv torchmetrics
```

### Optional Dependencies

For logging:
- **Weights & Biases**: `pip install wandb`
- **TensorBoard**: `pip install tensorboard` (included with PyTorch)
- **Comet ML**: `pip install comet-ml`

### MATLAB (for data preprocessing)

If you need to process raw CSI data, you'll need MATLAB with:
- Signal Processing Toolbox
- HDF5 support

## Getting Started

### 1. Prepare Your Dataset

#### Option A: Use Preprocessed Data

If you have preprocessed HDF5 files with AoA-ToF features, skip to step 2.

#### Option B: Process Raw CSI Data

1. Navigate to `wifi_data_processing/`
2. Follow instructions in `wifi_data_processing/README.md`
3. Run MATLAB scripts to generate HDF5 feature files

### 2. Configure Dataset Paths

Edit `configs/dataset/default.yaml`:

```yaml
dataset:
  # Update these paths to your data locations
  train_data_path: "/path/to/your/train_data.csv"
  val_data_path: "/path/to/your/val_data.csv"
  test_data_path: "/path/to/your/test_data.csv"
  
  # Number of access points in your setup
  train_n_aps: 4
  val_n_aps: 4
  test_n_aps: 4
  
  # Adjust based on your GPU memory
  batch_size: 16
  num_workers: 8
```

**Note**: The CSV files should contain paths to HDF5 files, one per line.

### 3. Configure Training Settings

Edit `configs/trainer/default.yaml`:

```yaml
trainer:
  # Set accelerator: "gpu" or "cpu"
  accelerator: "gpu"
  
  # GPU device IDs (e.g., [0], [0,1], or [1])
  devices: [0]
  
  # Strategy for multi-GPU training
  strategy: "ddp_find_unused_parameters_true"  # Use "auto" for single GPU
```

### 4. Configure Model Settings (Optional)

Edit `configs/model/default.yaml`:

```yaml
model:
  # Federated learning: average weights every N batches
  average_weight_every_n_batches: 10
  
  # Model architecture
  dropout: 0.3
  in_channels: 1
```

### 5. Configure Experiment Settings

Edit `configs/config.yaml`:

```yaml
experiment:
  name: "my_experiment"
  seed: 42
  max_epochs: 50
  learning_rate: 5e-5
```

### 6. Choose a Logger (Optional)

By default, no external logger is used. To enable logging:

**Weights & Biases:**
```bash
# Set up environment variable
export WANDB_API_KEY=your_api_key

# Run with wandb logger
python main.py logger=wandb
```

**TensorBoard:**
```bash
python main.py logger=tensorboard

# View logs
tensorboard --logdir=./logs
```

**No logger:**
```bash
python main.py logger=none
```

### 7. Run Training

```bash
python main.py
```

#### Override Configuration from Command Line

You can override any configuration parameter:

```bash
# Change batch size and learning rate
python main.py dataset.batch_size=32 experiment.learning_rate=1e-4

# Use different GPU
python main.py trainer.devices=[1]

# Change number of epochs
python main.py experiment.max_epochs=100

# Use different dataset
python main.py dataset.train_data_path=/path/to/other/data.csv
```

#### Run Multiple Experiments (Hyperparameter Sweep)

```bash
# Sweep over multiple learning rates
python main.py -m experiment.learning_rate=1e-4,5e-5,1e-5
```

## Training Output

Training outputs are saved in `outputs/YYYY-MM-DD/HH-MM-SS/`:
- `main.log` - Training logs
- `.hydra/` - Hydra configuration snapshots
- Model checkpoints (best model based on validation loss)

## Model Architecture

The system uses a **FederatedLearningModel** that:
1. Processes AoA-ToF features from multiple access points using separate ResNet34 encoders
2. Predicts AoA (angle-of-arrival) for each AP using trigonometric outputs (sin/cos)
3. Combines predictions from all APs to estimate device location via ray intersection
4. Implements federated learning by averaging encoder parameters across APs every N batches

**Loss Components:**
- AoA prediction loss (trigonometric loss using sin/cos differences)
- Location prediction loss (MSE between predicted and ground truth positions)
- Geometric consistency loss (ensures AoA predictions are geometrically consistent with location)

## Evaluation Metrics

- **AoA Accuracy**: Angular error between predicted and ground truth angles
- **Location Accuracy**: Euclidean distance error between predicted and ground truth positions
- **RSSI Metrics**: Signal strength-based metrics (if available)

## Environment Variables

Create a `.env` file in the project root for sensitive information:

```bash
# Weights & Biases
WANDB_API_KEY=your_api_key

# Comet ML
COMET_API_KEY=your_api_key
COMET_PROJECT_NAME=your_project
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `dataset.batch_size` in config
- Reduce `dataset.num_workers`
- Use smaller GPU device IDs or fewer devices

### Data Loading Issues
- Verify CSV files contain valid paths to HDF5 files
- Check HDF5 files contain required keys: `features_2d`, `aoa_label`, `location_label`, `ap_metadata`
- Ensure `train_n_aps`, `val_n_aps`, `test_n_aps` match your data

### Multi-GPU Training Issues
- Use `strategy: "ddp_find_unused_parameters_true"` for federated learning
- Ensure all GPUs are visible: `export CUDA_VISIBLE_DEVICES=0,1`

### Slow Training
- Increase `dataset.num_workers` (typically 4-8 per GPU)
- Increase `dataset.prefetch_factor` (default: 2)
- Enable `persistent_workers` in DataLoader (already enabled for training)

## Testing

Run unit tests:

```bash
# Test geometry utilities
python utils/tests/geometry_utils_test.py

# Test ray intersection solver
python utils/tests/ray_intersection_solver_test.py

# Test schema definitions
python utils/tests/schema_test.py
```

## Citation

If you use this code, please cite the relevant paper (add citation here when available).

## License

(Add license information here)

## Contact

(Add contact information here)

