# PriWiLoc Examples

This directory contains example scripts and configurations to help you get started with PriWiLoc using Hydra.

## Quick Start Examples

### 1. CSV Logger (No Setup Required)

```bash
bash examples/run_with_csv_logger.sh
```

This is the simplest way to get started. No API keys or accounts needed!

### 2. TensorBoard Logger

```bash
bash examples/run_with_tensorboard.sh
```

Then view results:
```bash
tensorboard --logdir logs/
```

### 3. Hyperparameter Sweep

```bash
bash examples/run_hyperparameter_sweep.sh
```

This will run multiple experiments with different hyperparameters.

## Custom Configuration

### Using a Custom Dataset Config

1. Copy `custom_dataset_config.yaml` to `configs/dataset/my_dataset.yaml`
2. Update the paths in the file
3. Run:

```bash
python main_hydra.py dataset=my_dataset
```

## Command Line Examples

### Basic Training

```bash
python main_hydra.py \
  logger=csv \
  dataset.train_data_path=/path/to/train.csv \
  dataset.val_data_path=/path/to/val.csv \
  dataset.test_data_path=/path/to/test.csv
```

### With Custom Parameters

```bash
python main_hydra.py \
  logger=tensorboard \
  experiment.max_epochs=100 \
  experiment.learning_rate=1e-4 \
  dataset.batch_size=32 \
  trainer.devices=[0,1]
```

### Quick Test (Fast)

```bash
python main_hydra.py \
  logger=none \
  experiment.max_epochs=1 \
  dataset.batch_size=2 \
  dataset.num_workers=0
```

## Using Different Loggers

### Comet ML

```bash
export COMET_API_KEY="your_key"
export COMET_WORKSPACE="your_workspace"
python main_hydra.py logger=comet
```

### Weights & Biases

```bash
wandb login
python main_hydra.py logger=wandb logger.entity=your_username
```

### TensorBoard

```bash
python main_hydra.py logger=tensorboard
tensorboard --logdir logs/
```

### CSV (No Setup)

```bash
python main_hydra.py logger=csv
```

## Tips

1. **Start with CSV logger** - no setup required
2. **Use small epochs first** - test with `experiment.max_epochs=5`
3. **Check your data paths** - make sure they're correct
4. **Use `logger=none`** for debugging
5. **Run sweeps** to find best hyperparameters

## Troubleshooting

### "No such file or directory"

Update the data paths in the example scripts to point to your actual data files.

### "COMET_API_KEY not set"

Use a different logger that doesn't require API keys:
```bash
python main_hydra.py logger=csv
```

### "Out of memory"

Reduce batch size:
```bash
python main_hydra.py dataset.batch_size=8
```

## More Information

- See [HYDRA_GUIDE.md](../HYDRA_GUIDE.md) for comprehensive documentation
- See [README_HYDRA.md](../README_HYDRA.md) for quick reference

