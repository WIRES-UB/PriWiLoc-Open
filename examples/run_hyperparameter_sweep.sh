#!/bin/bash
# Example: Run hyperparameter sweep
# This will run multiple experiments with different hyperparameters

python main_hydra.py -m \
  logger=csv \
  experiment.name=hyperparam_sweep \
  experiment.learning_rate=1e-5,5e-5,1e-4 \
  dataset.batch_size=16,32 \
  ukf.measurement_noise_std=0.5,0.7,1.0 \
  dataset.train_data_path=/path/to/your/train.csv \
  dataset.val_data_path=/path/to/your/val.csv \
  dataset.test_data_path=/path/to/your/test.csv

echo "Sweep complete! Check multirun/ directory for results"

