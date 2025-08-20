#!/usr/bin/env python3
"""
LoRA Length Extension Finetuning Script for Radial Attention
This script implements training of LoRA adapters for video length extension
as described in the Radial Attention paper.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from transformers import UMT5EncoderModel
from peft import LoraConfig, get_peft_model, TaskType

from radial_attn.utils import set_seed
from radial_attn.models.wan.inference import replace_wan_attention
from radial_attn.models.wan.sparse_transformer import replace_sparse_forward

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for LoRA length extension training"""
    # Model configuration
    model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    
    # Training configuration
    output_dir: str = "lora_length_extension_checkpoints"
    learning_rate: float = 1e-4
    batch_size: int = 1
    num_epochs: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # LoRA configuration
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",  # Self-attention modules
        "proj_in", "proj_out"  # Projection layers
    ])
    
    # Video configuration
    base_num_frames: int = 69  # Base video length (1x)
    extended_num_frames: int = 161  # Extended video length (2.3x)
    height: int = 768
    width: int = 1280
    
    # Radial attention configuration
    pattern: str = "radial"
    dense_layers: int = 2
    dense_timesteps: int = 2
    decay_factor: float = 0.2
    
    # Training data
    data_dir: str = "training_data"
    validation_split: float = 0.1
    
    # Logging and checkpointing
    save_every: int = 10
    validate_every: int = 5
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "radial-attention-lora"
    
    # Hardware optimization
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_sage_attention: bool = False
    
    # Flow configuration
    flow_shift: float = 5.0
    num_inference_steps: int = 50
    guidance_scale: float = 5.0


class VideoTextDataset(Dataset):
    """Dataset for video-text pairs for length extension training"""
    
    def __init__(self, data_dir: str, config: TrainingConfig, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        
        # Load metadata
        metadata_file = self.data_dir / f"{split}_metadata.json"
        if not metadata_file.exists():
            logger.warning(f"Metadata file {metadata_file} not found. Creating dummy data.")
            self.metadata = self._create_dummy_metadata()
        else:
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        logger.info(f"Loaded {len(self.metadata)} samples for {split} split")
    
    def _create_dummy_metadata(self) -> List[Dict]:
        """Create dummy metadata for testing purposes"""
        dummy_prompts = [
            "A serene lake with mountains in the background, peaceful and calm",
            "A bustling city street with cars and people moving",
            "A forest path with sunlight filtering through trees",
            "Ocean waves crashing against rocks on a beach",
            "A field of flowers swaying in the wind",
        ]
        
        metadata = []
        for i, prompt in enumerate(dummy_prompts):
            metadata.append({
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, static, bad composition",
                "video_id": f"dummy_{i:04d}",
                "base_frames": self.config.base_num_frames,
                "extended_frames": self.config.extended_num_frames
            })
        
        return metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.metadata[idx]
        return {
            "prompt": item["prompt"],
            "negative_prompt": item.get("negative_prompt", ""),
            "video_id": item["video_id"],
            "base_frames": item.get("base_frames", self.config.base_num_frames),
            "extended_frames": item.get("extended_frames", self.config.extended_num_frames)
        }


class LoRALengthExtensionTrainer:
    """Trainer for LoRA length extension finetuning"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision="fp16" if config.use_mixed_precision else None,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb" if config.use_wandb else None,
            project_config={"project_name": config.wandb_project} if config.use_wandb else None
        )
        
        # Set random seed
        set_seed(42)
        
        # Setup models
        self._setup_models()
        
        # Setup LoRA
        self._setup_lora()
        
        # Setup data
        self._setup_data()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup logging
        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(project=config.wandb_project, config=config.__dict__)
    
    def _setup_models(self):
        """Setup the base models"""
        logger.info("Loading base models...")
        
        # Replace sparse forward for radial attention
        replace_sparse_forward()
        
        # Load models
        self.vae = AutoencoderKLWan.from_pretrained(
            self.config.model_id, 
            subfolder="vae", 
            torch_dtype=torch.float32
        )
        
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.config.model_id, 
            subfolder="text_encoder", 
            torch_dtype=torch.bfloat16
        )
        
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.config.model_id, 
            subfolder="transformer", 
            torch_dtype=torch.bfloat16
        )
        
        self.scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction', 
            use_flow_sigmas=True, 
            num_train_timesteps=1000, 
            flow_shift=self.config.flow_shift
        )
        
        # Create pipeline
        self.pipe = WanPipeline.from_pretrained(
            self.config.model_id,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            vae=self.vae,
            torch_dtype=torch.bfloat16
        )
        self.pipe.scheduler = self.scheduler
        
        # Enable gradient checkpointing if specified
        if self.config.use_gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        
        # Freeze non-LoRA parameters
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        logger.info("Base models loaded successfully")
    
    def _setup_lora(self):
        """Setup LoRA adapters"""
        logger.info("Setting up LoRA adapters...")
        
        # Define LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.DIFFUSION,
        )
        
        # Apply LoRA to transformer
        self.transformer = get_peft_model(self.transformer, lora_config)
        self.transformer.print_trainable_parameters()
        
        logger.info("LoRA adapters setup successfully")
    
    def _setup_data(self):
        """Setup training and validation datasets"""
        logger.info("Setting up datasets...")
        
        # Create datasets
        self.train_dataset = VideoTextDataset(
            self.config.data_dir, 
            self.config, 
            split="train"
        )
        
        self.val_dataset = VideoTextDataset(
            self.config.data_dir, 
            self.config, 
            split="val"
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        logger.info("Setting up optimizer and scheduler...")
        
        # Get trainable parameters (only LoRA parameters)
        trainable_params = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        logger.info(f"Optimizer setup with {len(trainable_params)} trainable parameters")
    
    def _generate_latents(self, batch: Dict[str, Any], num_frames: int) -> torch.Tensor:
        """Generate latents for a given number of frames"""
        # Setup radial attention for the specific frame count
        replace_wan_attention(
            self.pipe,
            self.config.height,
            self.config.width,
            num_frames,
            self.config.dense_layers,
            self.config.dense_timesteps,
            self.config.decay_factor,
            self.config.pattern,
            self.config.use_sage_attention,
        )
        
        # Generate noise latents
        batch_size = len(batch["prompt"])
        latent_shape = (
            batch_size,
            self.pipe.transformer.config.in_channels,
            num_frames,
            self.config.height // 8,
            self.config.width // 8
        )
        latents = torch.randn(latent_shape, device=self.accelerator.device, dtype=torch.bfloat16)
        
        return latents
    
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute the training loss"""
        # Encode text prompts
        with torch.no_grad():
            prompt_embeds = self.pipe._encode_prompt(
                batch["prompt"],
                self.accelerator.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=batch["negative_prompt"]
            )
        
        # Generate latents for base and extended frames
        base_latents = self._generate_latents(batch, self.config.base_num_frames)
        extended_latents = self._generate_latents(batch, self.config.extended_num_frames)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (base_latents.shape[0],), device=self.accelerator.device
        ).long()
        
        # Add noise to latents
        noise_base = torch.randn_like(base_latents)
        noise_extended = torch.randn_like(extended_latents)
        
        noisy_base_latents = self.scheduler.add_noise(base_latents, noise_base, timesteps)
        noisy_extended_latents = self.scheduler.add_noise(extended_latents, noise_extended, timesteps)
        
        # Predict noise for base frames
        noise_pred_base = self.transformer(
            noisy_base_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds[0],
            return_dict=False
        )[0]
        
        # Predict noise for extended frames  
        noise_pred_extended = self.transformer(
            noisy_extended_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds[0],
            return_dict=False
        )[0]
        
        # Compute consistency loss between base and extended predictions
        # The LoRA should learn to maintain consistency when extending length
        base_loss = F.mse_loss(noise_pred_base, noise_base, reduction="mean")
        extended_loss = F.mse_loss(noise_pred_extended, noise_extended, reduction="mean")
        
        # Consistency loss: first N frames of extended should be similar to base
        if extended_latents.shape[2] >= base_latents.shape[2]:
            base_portion = noise_pred_extended[:, :, :base_latents.shape[2]]
            consistency_loss = F.mse_loss(base_portion, noise_pred_base, reduction="mean")
        else:
            consistency_loss = torch.tensor(0.0, device=self.accelerator.device)
        
        # Total loss
        total_loss = base_loss + extended_loss + 0.5 * consistency_loss
        
        return total_loss, {
            "base_loss": base_loss.item(),
            "extended_loss": extended_loss.item(), 
            "consistency_loss": consistency_loss.item(),
            "total_loss": total_loss.item()
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.transformer.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.transformer):
                loss, loss_dict = self._compute_loss(batch)
                
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.transformer.parameters(),
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Log metrics
                if step % self.config.log_every == 0 and self.accelerator.is_main_process:
                    metrics = {
                        "train/loss": loss.item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step": step + epoch * len(self.train_loader)
                    }
                    metrics.update({f"train/{k}": v for k, v in loss_dict.items()})
                    
                    if self.config.use_wandb:
                        wandb.log(metrics)
                    
                    logger.info(f"Step {step}: {metrics}")
        
        avg_loss = total_loss / num_batches
        return {"avg_loss": avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.transformer.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", disable=not self.accelerator.is_main_process):
                loss, loss_dict = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if self.accelerator.is_main_process:
            metrics = {"val/loss": avg_loss, "val/epoch": epoch}
            if self.config.use_wandb:
                wandb.log(metrics)
            logger.info(f"Validation - Epoch {epoch}: {metrics}")
        
        return {"avg_loss": avg_loss}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        checkpoint_path = output_dir / f"lora_epoch_{epoch}.safetensors"
        self.transformer.save_pretrained(checkpoint_path.parent / f"epoch_{epoch}")
        
        # Save config
        config_path = output_dir / "lora_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "lora_params": {
                    "lora_rank": self.config.lora_rank,
                    "lora_alpha": self.config.lora_alpha,
                    "lora_dropout": self.config.lora_dropout,
                    "target_modules": self.config.target_modules
                },
                "training_config": self.config.__dict__
            }, f, indent=2)
        
        if is_best:
            best_path = output_dir / "best_model"
            self.transformer.save_pretrained(best_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Prepare models for training with accelerator
        self.transformer, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.transformer, self.optimizer, self.train_loader, self.val_loader
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config.validate_every == 0:
                val_metrics = self.validate(epoch)
                
                # Check if this is the best model
                is_best = val_metrics["avg_loss"] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics["avg_loss"]
                
                # Save checkpoint
                if epoch % self.config.save_every == 0:
                    self.save_checkpoint(epoch, is_best)
        
        # Save final checkpoint
        self.save_checkpoint(self.config.num_epochs - 1, False)
        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA for video length extension")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--output_dir", type=str, default="lora_length_extension_checkpoints")
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="radial-attention-lora")
    
    args = parser.parse_args()
    
    # Create training configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig(
            model_id=args.model_id,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
        )
    
    # Create trainer and start training
    trainer = LoRALengthExtensionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()