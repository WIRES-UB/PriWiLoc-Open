#!/bin/bash
# Example: Run training with CSV logger (no API keys needed)
# This is great for open-source users who want to try the code

python main_hydra.py \
  logger=csv \
  experiment.name=csv_example \
  experiment.max_epochs=10 \
  dataset.batch_size=16 \
  dataset.train_data_path=/path/to/your/train.csv \
  dataset.val_data_path=/path/to/your/val.csv \
  dataset.test_data_path=/path/to/your/test.csv

echo "Training complete! Check logs/ directory for results"

