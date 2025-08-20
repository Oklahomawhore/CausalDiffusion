#!/bin/bash

# LoRA Length Extension Evaluation Script
# This script evaluates trained LoRA models with different configurations

set -e

LORA_CHECKPOINT_DIR="$1"
OUTPUT_DIR="${2:-evaluation_outputs}"

if [ -z "$LORA_CHECKPOINT_DIR" ]; then
    echo "Usage: $0 <lora_checkpoint_dir> [output_dir]"
    echo "Example: $0 lora_checkpoints/best_model evaluation_outputs"
    exit 1
fi

if [ ! -d "$LORA_CHECKPOINT_DIR" ]; then
    echo "Error: LoRA checkpoint directory does not exist: $LORA_CHECKPOINT_DIR"
    exit 1
fi

echo "Evaluating LoRA model: $LORA_CHECKPOINT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Test prompts for evaluation
PROMPTS=(
    "A serene lake with mountains in the background, peaceful and calm"
    "A bustling city street with cars and people moving during rush hour"
    "A forest path with golden sunlight filtering through autumn trees"
    "Ocean waves crashing against rocky cliffs on a stormy day"
    "A field of colorful wildflowers swaying gently in the spring breeze"
)

# Evaluation configurations
echo "Running comprehensive evaluation..."

# 1. Base evaluation (69 -> 161 frames)
echo "=== Base Length Extension (69 -> 161 frames) ==="
python evaluate_lora_length_extension.py \
    --lora_checkpoint_dir "$LORA_CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR/base_extension" \
    --base_frames 69 \
    --extended_frames 161 \
    --height 768 \
    --width 1280 \
    --num_inference_steps 50 \
    --seed 42

# 2. Extreme extension (69 -> 273 frames)
echo "=== Extreme Length Extension (69 -> 273 frames) ==="
python evaluate_lora_length_extension.py \
    --lora_checkpoint_dir "$LORA_CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR/extreme_extension" \
    --base_frames 69 \
    --extended_frames 273 \
    --height 768 \
    --width 1280 \
    --num_inference_steps 50 \
    --dense_layers 3 \
    --dense_timesteps 1 \
    --decay_factor 0.15 \
    --seed 42

# 3. High resolution test
echo "=== High Resolution Test (161 frames, 1024x1536) ==="
python evaluate_lora_length_extension.py \
    --lora_checkpoint_dir "$LORA_CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR/high_resolution" \
    --base_frames 69 \
    --extended_frames 161 \
    --height 1024 \
    --width 1536 \
    --num_inference_steps 50 \
    --seed 42

# 4. Fast inference test
echo "=== Fast Inference Test (25 steps) ==="
python evaluate_lora_length_extension.py \
    --lora_checkpoint_dir "$LORA_CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR/fast_inference" \
    --base_frames 69 \
    --extended_frames 161 \
    --height 768 \
    --width 1280 \
    --num_inference_steps 25 \
    --seed 42

# 5. Individual prompt evaluation
echo "=== Individual Prompt Evaluation ==="
for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    echo "Evaluating prompt $((i+1)): $prompt"
    
    python evaluate_lora_length_extension.py \
        --lora_checkpoint_dir "$LORA_CHECKPOINT_DIR" \
        --output_dir "$OUTPUT_DIR/individual_prompts/prompt_$((i+1))" \
        --prompt "$prompt" \
        --base_frames 69 \
        --extended_frames 161 \
        --height 768 \
        --width 1280 \
        --num_inference_steps 50 \
        --seed $((42 + i))
done

# Generate evaluation summary
echo "=== Generating Evaluation Summary ==="
python -c "
import json
import os
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
summary = {
    'lora_checkpoint': '$LORA_CHECKPOINT_DIR',
    'evaluation_date': '$(date -u +"%Y-%m-%dT%H:%M:%SZ")',
    'test_configurations': [
        'base_extension',
        'extreme_extension', 
        'high_resolution',
        'fast_inference',
        'individual_prompts'
    ],
    'results': {}
}

# Collect results from each test
for config in summary['test_configurations']:
    config_dir = output_dir / config
    if config_dir.exists():
        results_file = config_dir / 'evaluation_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            successful = sum(1 for r in results if r.get('success', False))
            summary['results'][config] = {
                'total_samples': len(results),
                'successful_samples': successful,
                'success_rate': successful / len(results) if results else 0
            }

# Save summary
with open(output_dir / 'evaluation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Evaluation summary saved to: {output_dir / \"evaluation_summary.json\"}')
"

echo "=== Evaluation Complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo "Summary: $OUTPUT_DIR/evaluation_summary.json"

# Print quick summary
echo ""
echo "Quick Summary:"
echo "=============="
if [ -f "$OUTPUT_DIR/evaluation_summary.json" ]; then
    python -c "
import json
with open('$OUTPUT_DIR/evaluation_summary.json', 'r') as f:
    summary = json.load(f)

print(f\"LoRA Model: {summary['lora_checkpoint']}\")
print(f\"Evaluation Date: {summary['evaluation_date']}\")
print(\"\nResults:\")
for config, results in summary['results'].items():
    success_rate = results['success_rate'] * 100
    print(f\"  {config}: {results['successful_samples']}/{results['total_samples']} ({success_rate:.1f}% success)\")
"
else
    echo "Summary file not found"
fi

echo ""
echo "Check the output directory for generated videos and detailed results."