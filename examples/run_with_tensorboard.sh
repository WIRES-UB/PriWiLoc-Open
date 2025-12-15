#!/bin/bash
# Example: Run training with TensorBoard logger
# View results with: tensorboard --logdir logs/

python main_hydra.py \
  logger=tensorboard \
  experiment.name=tensorboard_example \
  experiment.max_epochs=50 \
  dataset.batch_size=32 \
  dataset.train_data_path=/path/to/your/train.csv \
  dataset.val_data_path=/path/to/your/val.csv \
  dataset.test_data_path=/path/to/your/test.csv

echo "Training complete! Run 'tensorboard --logdir logs/' to view results"

