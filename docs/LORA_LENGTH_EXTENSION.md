# LoRA Length Extension Finetuning

This document describes the LoRA (Low-Rank Adaptation) finetuning implementation for video length extension using Radial Attention, as described in the paper.

## Overview

The LoRA length extension finetuning enables pre-trained video diffusion models (e.g., Wan2.1-14B, HunyuanVideo) to generate videos that are 4× longer than their original training length while maintaining video quality. This is achieved through:

1. **Lightweight LoRA Adapters**: Only a small number of parameters are trained, making the process efficient
2. **Radial Attention**: Sparse attention mechanism that scales to longer sequences with O(n log n) complexity
3. **Consistency Training**: The model learns to maintain temporal consistency when extending video length

## Key Features

- **Efficient Training**: Only LoRA parameters are trained, requiring minimal computational resources
- **Length Extension**: Supports extending videos from base length (69 frames) to extended length (161+ frames)
- **Quality Preservation**: Maintains video quality while extending temporal length
- **Radial Attention Integration**: Uses sparse attention patterns for efficient long video generation
- **Flexible Configuration**: Configurable LoRA rank, target modules, and training parameters

## Files Structure

```
├── lora_length_extension_train.py     # Main training script
├── evaluate_lora_length_extension.py  # Evaluation script
├── prepare_training_data.py           # Data preparation utilities
├── configs/
│   └── lora_length_extension.json     # Training configuration
└── scripts/
    └── train_lora_length_extension.sh # Training script
```

## Installation

Ensure you have the required dependencies installed:

```bash
pip install accelerate wandb tqdm peft
```

All other dependencies should already be available from the main requirements.txt.

## Usage

### 1. Prepare Training Data

First, prepare your training dataset:

```bash
# Create a sample dataset for testing
python prepare_training_data.py create --output_dir training_data --num_samples 100

# Or convert existing video dataset
python prepare_training_data.py convert --input_dir /path/to/videos --output_dir training_data

# Validate dataset format
python prepare_training_data.py validate --output_dir training_data
```

### 2. Configure Training

Edit the configuration file `configs/lora_length_extension.json` to adjust training parameters:

```json
{
  "model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
  "learning_rate": 1e-4,
  "batch_size": 1,
  "num_epochs": 100,
  "lora_rank": 64,
  "lora_alpha": 64,
  "base_num_frames": 69,
  "extended_num_frames": 161,
  "pattern": "radial",
  "dense_layers": 2,
  "dense_timesteps": 2,
  "decay_factor": 0.2
}
```

### 3. Train LoRA Model

Run the training script:

```bash
# Using the provided script
bash scripts/train_lora_length_extension.sh

# Or run directly with custom parameters
python lora_length_extension_train.py \
    --config configs/lora_length_extension.json \
    --output_dir lora_checkpoints \
    --data_dir training_data
```

### 4. Evaluate Trained Model

Evaluate the trained LoRA model:

```bash
python evaluate_lora_length_extension.py \
    --lora_checkpoint_dir lora_checkpoints/best_model \
    --output_dir evaluation_outputs \
    --base_frames 69 \
    --extended_frames 161
```

### 5. Use Trained LoRA for Inference

Use the trained LoRA with existing inference scripts:

```bash
python wan_t2v_inference.py \
    --prompt "A serene lake with mountains in the background" \
    --num_frames 161 \
    --pattern radial \
    --dense_layers 2 \
    --dense_timesteps 2 \
    --decay_factor 0.2 \
    --lora_checkpoint_dir lora_checkpoints/best_model \
    --output_file extended_video.mp4
```

## Training Configuration

### LoRA Parameters

- **lora_rank**: Rank of LoRA adaptation (default: 64)
- **lora_alpha**: LoRA scaling parameter (default: 64)
- **lora_dropout**: Dropout rate for LoRA layers (default: 0.1)
- **target_modules**: Which modules to apply LoRA to (attention layers by default)

### Video Configuration

- **base_num_frames**: Base video length for training (default: 69)
- **extended_num_frames**: Extended video length for training (default: 161)
- **height/width**: Video resolution (default: 768x1280)

### Radial Attention Configuration

- **pattern**: Attention pattern ("radial" or "dense")
- **dense_layers**: Number of dense attention layers (default: 2)
- **dense_timesteps**: Number of dense timesteps (default: 2)
- **decay_factor**: Exponential decay factor for radial attention (default: 0.2)

### Training Parameters

- **learning_rate**: Learning rate for optimizer (default: 1e-4)
- **batch_size**: Batch size for training (default: 1)
- **num_epochs**: Number of training epochs (default: 100)
- **gradient_accumulation_steps**: Steps for gradient accumulation (default: 4)

## Data Format

The training data should be organized as JSON metadata files:

```json
[
  {
    "video_id": "sample_000001",
    "prompt": "A serene lake with mountains in the background",
    "negative_prompt": "blurry, low quality, static",
    "base_frames": 69,
    "extended_frames": 161,
    "height": 768,
    "width": 1280
  }
]
```

## Training Process

The training process involves:

1. **Model Setup**: Load base Wan2.1 model and apply LoRA adapters
2. **Data Loading**: Load video-text pairs from metadata
3. **Loss Computation**: Compute denoising loss for both base and extended lengths
4. **Consistency Loss**: Ensure first N frames of extended video match base video
5. **Optimization**: Update only LoRA parameters using gradient descent

## Key Components

### TrainingConfig
Configuration dataclass containing all training parameters and hyperparameters.

### VideoTextDataset
Dataset class for loading video-text pairs from metadata files.

### LoRALengthExtensionTrainer
Main trainer class that handles:
- Model setup and LoRA application
- Training loop with consistency loss
- Validation and checkpointing
- Radial attention configuration

## Evaluation Metrics

The evaluation script generates videos at both base and extended lengths and saves them for visual comparison. Key aspects to evaluate:

1. **Temporal Consistency**: Extended videos should maintain smooth temporal flow
2. **Quality Preservation**: Video quality should not degrade with length extension
3. **Content Coherence**: Extended content should be semantically consistent
4. **Motion Continuity**: Object motion should continue naturally

## Tips for Training

1. **Start Small**: Begin with a small dataset to verify the training pipeline
2. **Monitor Loss**: Watch for convergence of both denoising and consistency losses
3. **Adjust LoRA Rank**: Higher rank = more parameters but potentially better adaptation
4. **Tune Decay Factor**: Affects the sparsity pattern of radial attention
5. **Use Gradient Checkpointing**: Enables larger batch sizes with limited GPU memory

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size, enable gradient checkpointing, or use model offloading
2. **Poor Convergence**: Adjust learning rate, increase LoRA rank, or check data quality
3. **Inconsistent Videos**: Increase consistency loss weight or adjust dense layers/timesteps
4. **Slow Training**: Use mixed precision, gradient accumulation, or multiple GPUs

### Performance Optimization

- Use `use_mixed_precision=True` for faster training
- Enable `use_gradient_checkpointing=True` for memory efficiency
- Use `use_sage_attention=True` if available for accelerated attention
- Consider `use_model_offload=True` for very large models

## Advanced Usage

### Custom Target Modules

Modify the target modules for LoRA adaptation:

```python
target_modules = [
    "to_q", "to_k", "to_v", "to_out.0",  # Self-attention
    "proj_in", "proj_out",               # Projections
    "ff.net.0", "ff.net.2"              # Feed-forward layers
]
```

### Custom Loss Functions

Implement custom loss functions by modifying `_compute_loss` in the trainer:

```python
def _compute_loss(self, batch):
    # Add custom loss terms
    temporal_consistency_loss = compute_temporal_consistency(...)
    motion_smoothness_loss = compute_motion_smoothness(...)
    
    total_loss = base_loss + extended_loss + temporal_consistency_loss + motion_smoothness_loss
    return total_loss
```

### Multi-GPU Training

Enable multi-GPU training with Accelerate:

```python
accelerate launch --num_processes 4 lora_length_extension_train.py \
    --config configs/lora_length_extension.json
```

## Integration with Existing Models

The trained LoRA adapters can be used with existing inference scripts:

- `wan_t2v_inference.py`: For Wan2.1 models
- `wan_22_t2v_inference.py`: For Wan2.2 models
- `hunyuan_t2v_inference.py`: For HunyuanVideo models

Simply provide the `--lora_checkpoint_dir` parameter pointing to your trained LoRA.

## Future Improvements

Potential enhancements to the LoRA training:

1. **Progressive Training**: Start with shorter extensions and gradually increase length
2. **Multi-Scale Training**: Train on videos of different resolutions simultaneously
3. **Adversarial Training**: Add discriminator for improved temporal consistency
4. **Curriculum Learning**: Order training data by difficulty/complexity
5. **Distributed Training**: Scale to larger datasets with multiple GPUs

## References

- [Radial Attention Paper](https://arxiv.org/abs/2506.19852)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [Diffusers Library](https://github.com/huggingface/diffusers)