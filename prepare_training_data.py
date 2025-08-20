#!/usr/bin/env python3
"""
Data Preparation Script for LoRA Length Extension Training
This script helps prepare video-text datasets for training LoRA adapters.
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_dataset(output_dir: str, num_samples: int = 100):
    """Create a sample dataset with dummy metadata for testing"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample prompts for video generation
    sample_prompts = [
        "A serene lake with mountains in the background, peaceful and calm",
        "A bustling city street with cars and people moving during rush hour",
        "A forest path with golden sunlight filtering through autumn trees",
        "Ocean waves crashing against rocky cliffs on a stormy day",
        "A field of colorful wildflowers swaying gently in the spring breeze",
        "A time-lapse of clouds moving across a bright blue sky",
        "A waterfall cascading down moss-covered rocks in a lush forest",
        "A desert landscape with sand dunes shifting in the wind",
        "A snowy mountain peak glistening under bright sunlight",
        "A busy marketplace with vendors and customers in vibrant colors",
        "A peaceful garden with butterflies flying among blooming flowers",
        "A river flowing through a valley with green hills on both sides",
        "A lighthouse standing tall against dramatic storm clouds",
        "A field of wheat swaying in golden afternoon light",
        "A campfire crackling in the darkness with sparks flying upward",
        "A modern city skyline reflecting in calm water at sunset",
        "A train moving through a picturesque countryside landscape",
        "A coral reef underwater with colorful fish swimming around",
        "A bird soaring high above mountain peaks and valleys",
        "A cozy cabin in the woods surrounded by tall pine trees"
    ]
    
    negative_prompts = [
        "blurry, low quality, static, bad composition, distorted",
        "overexposed, underexposed, noisy, pixelated, artifacts",
        "unrealistic, cartoonish, artificial, synthetic, fake",
        "low resolution, compressed, jpeg artifacts, poor lighting",
        "shaky camera, motion blur, out of focus, grainy, dark"
    ]
    
    def generate_metadata(split: str, start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """Generate metadata for a specific split"""
        metadata = []
        
        for i in range(start_idx, end_idx):
            prompt_idx = i % len(sample_prompts)
            neg_prompt_idx = i % len(negative_prompts)
            
            metadata.append({
                "video_id": f"sample_{i:06d}",
                "prompt": sample_prompts[prompt_idx],
                "negative_prompt": negative_prompts[neg_prompt_idx],
                "base_frames": 69,
                "extended_frames": 161,
                "height": 768,
                "width": 1280,
                "duration": 4.5,  # seconds
                "fps": 24
            })
        
        return metadata
    
    # Calculate split sizes
    train_size = int(num_samples * 0.9)
    val_size = num_samples - train_size
    
    # Generate training metadata
    train_metadata = generate_metadata("train", 0, train_size)
    val_metadata = generate_metadata("val", train_size, num_samples)
    
    # Save metadata files
    with open(output_path / "train_metadata.json", 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(output_path / "val_metadata.json", 'w') as f:
        json.dump(val_metadata, f, indent=2)
    
    # Create README with dataset information
    readme_content = f"""# LoRA Length Extension Training Dataset

This directory contains metadata for training LoRA adapters for video length extension.

## Dataset Structure

- `train_metadata.json`: Training split metadata ({len(train_metadata)} samples)
- `val_metadata.json`: Validation split metadata ({len(val_metadata)} samples)

## Metadata Format

Each entry in the metadata files contains:
- `video_id`: Unique identifier for the video
- `prompt`: Text prompt describing the video content
- `negative_prompt`: Negative prompt to avoid unwanted features
- `base_frames`: Number of frames for base length video (default: 69)
- `extended_frames`: Number of frames for extended length video (default: 161)
- `height`: Video height in pixels (default: 768)
- `width`: Video width in pixels (default: 1280)
- `duration`: Video duration in seconds
- `fps`: Frames per second

## Usage

This is a sample dataset for testing the LoRA training pipeline. For real training,
replace this with your actual video-text dataset following the same metadata format.

## Training

To train LoRA adapters with this dataset:

```bash
bash scripts/train_lora_length_extension.sh
```

The training script will use this metadata to generate synthetic training data for
learning length extension capabilities.
"""
    
    with open(output_path / "README.md", 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Sample dataset created with {num_samples} samples")
    logger.info(f"Training samples: {len(train_metadata)}")
    logger.info(f"Validation samples: {len(val_metadata)}")
    logger.info(f"Dataset saved to: {output_path}")


def validate_dataset(data_dir: str):
    """Validate an existing dataset"""
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Dataset directory does not exist: {data_dir}")
        return False
    
    # Check for required metadata files
    train_file = data_path / "train_metadata.json"
    val_file = data_path / "val_metadata.json"
    
    if not train_file.exists():
        logger.error(f"Training metadata file not found: {train_file}")
        return False
    
    if not val_file.exists():
        logger.error(f"Validation metadata file not found: {val_file}")
        return False
    
    # Validate metadata format
    try:
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        with open(val_file, 'r') as f:
            val_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metadata files: {e}")
        return False
    
    # Check required fields
    required_fields = ["video_id", "prompt", "base_frames", "extended_frames"]
    
    for split_name, data in [("train", train_data), ("val", val_data)]:
        if not isinstance(data, list):
            logger.error(f"{split_name} metadata should be a list")
            return False
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.error(f"{split_name} item {i} should be a dictionary")
                return False
            
            for field in required_fields:
                if field not in item:
                    logger.error(f"{split_name} item {i} missing required field: {field}")
                    return False
    
    logger.info(f"Dataset validation passed!")
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    
    return True


def convert_video_dataset(input_dir: str, output_dir: str, video_ext: str = ".mp4"):
    """Convert a directory of videos to training metadata format"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    video_files = list(input_path.glob(f"*{video_ext}"))
    
    if not video_files:
        logger.error(f"No video files found with extension {video_ext}")
        return
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Generate metadata
    metadata = []
    for i, video_file in enumerate(video_files):
        video_id = video_file.stem
        
        # Try to extract prompt from filename or create a generic one
        prompt = video_id.replace("_", " ").replace("-", " ").title()
        if len(prompt) < 10:
            prompt = f"A video showing {prompt}"
        
        metadata.append({
            "video_id": video_id,
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, static, bad composition",
            "video_path": str(video_file.relative_to(input_path)),
            "base_frames": 69,
            "extended_frames": 161,
            "height": 768,
            "width": 1280
        })
    
    # Split into train/val
    train_size = int(len(metadata) * 0.9)
    train_metadata = metadata[:train_size]
    val_metadata = metadata[train_size:]
    
    # Save metadata
    with open(output_path / "train_metadata.json", 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(output_path / "val_metadata.json", 'w') as f:
        json.dump(val_metadata, f, indent=2)
    
    logger.info(f"Dataset conversion completed!")
    logger.info(f"Training samples: {len(train_metadata)}")
    logger.info(f"Validation samples: {len(val_metadata)}")
    logger.info(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for LoRA length extension")
    parser.add_argument("command", choices=["create", "validate", "convert"], 
                       help="Command to execute")
    parser.add_argument("--output_dir", type=str, default="training_data",
                       help="Output directory for dataset")
    parser.add_argument("--input_dir", type=str,
                       help="Input directory containing videos (for convert command)")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to create (for create command)")
    parser.add_argument("--video_ext", type=str, default=".mp4",
                       help="Video file extension (for convert command)")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_sample_dataset(args.output_dir, args.num_samples)
    
    elif args.command == "validate":
        validate_dataset(args.output_dir)
    
    elif args.command == "convert":
        if not args.input_dir:
            parser.error("--input_dir is required for convert command")
        convert_video_dataset(args.input_dir, args.output_dir, args.video_ext)


if __name__ == "__main__":
    main()