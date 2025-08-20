#!/bin/bash

# LoRA Length Extension Training Script for Radial Attention
# This script trains LoRA adapters for 4x video length extension

set -e

# Configuration
CONFIG_FILE="configs/lora_length_extension.json"
OUTPUT_DIR="lora_length_extension_checkpoints"
DATA_DIR="training_data"

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"

echo "Starting LoRA Length Extension Training..."
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Data: $DATA_DIR"

# Run training
python lora_length_extension_train.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR"

echo "Training completed!"
echo "LoRA checkpoints saved to: $OUTPUT_DIR"