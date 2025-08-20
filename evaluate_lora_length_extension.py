#!/usr/bin/env python3
"""
LoRA Length Extension Evaluation Script
This script evaluates trained LoRA adapters for video length extension.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from transformers import UMT5EncoderModel
from termcolor import colored

from radial_attn.utils import set_seed
from radial_attn.models.wan.inference import replace_wan_attention
from radial_attn.models.wan.sparse_transformer import replace_sparse_forward

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LoRALengthExtensionEvaluator:
    """Evaluator for LoRA length extension models"""
    
    def __init__(
        self,
        model_id: str,
        lora_checkpoint_dir: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.model_id = model_id
        self.lora_checkpoint_dir = lora_checkpoint_dir
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load LoRA configuration
        self.lora_config = self._load_lora_config()
        
        # Setup models
        self._setup_models()
        
        logger.info("LoRA Length Extension Evaluator initialized")
    
    def _load_lora_config(self) -> Dict[str, Any]:
        """Load LoRA configuration from checkpoint directory"""
        config_path = Path(self.lora_checkpoint_dir) / "lora_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded LoRA config from {config_path}")
            return config
        else:
            logger.warning(f"LoRA config not found at {config_path}, using defaults")
            return {
                "lora_params": {
                    "lora_rank": 64,
                    "lora_alpha": 64
                },
                "training_config": {
                    "pattern": "radial",
                    "dense_layers": 2,
                    "dense_timesteps": 2,
                    "decay_factor": 0.2,
                    "flow_shift": 5.0
                }
            }
    
    def _setup_models(self):
        """Setup the base models and load LoRA weights"""
        logger.info("Loading base models...")
        
        # Replace sparse forward for radial attention
        replace_sparse_forward()
        
        # Load models
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32
        )
        
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder", 
            torch_dtype=self.torch_dtype
        )
        
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            torch_dtype=self.torch_dtype
        )
        
        # Setup scheduler with training config
        training_config = self.lora_config.get("training_config", {})
        flow_shift = training_config.get("flow_shift", 5.0)
        
        self.scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=flow_shift
        )
        
        # Create pipeline
        self.pipe = WanPipeline.from_pretrained(
            self.model_id,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            vae=self.vae,
            torch_dtype=self.torch_dtype
        )
        self.pipe.scheduler = self.scheduler
        
        # Load LoRA weights
        self._load_lora_weights()
        
        # Move to device
        self.pipe.to(self.device)
        
        logger.info("Models loaded successfully")
    
    def _load_lora_weights(self):
        """Load LoRA weights from checkpoint directory"""
        lora_dir = Path(self.lora_checkpoint_dir)
        
        # Look for LoRA checkpoint files
        safetensors_files = list(lora_dir.glob("*.safetensors"))
        best_model_dir = lora_dir / "best_model"
        
        if best_model_dir.exists():
            logger.info(f"Loading LoRA weights from best model: {best_model_dir}")
            self.pipe.load_lora_weights(best_model_dir)
        elif safetensors_files:
            # Use the latest checkpoint
            latest_checkpoint = max(safetensors_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading LoRA weights from: {latest_checkpoint}")
            self.pipe.load_lora_weights(str(lora_dir), weight_name=latest_checkpoint.name)
        else:
            raise FileNotFoundError(f"No LoRA checkpoint found in {lora_dir}")
        
        # Set LoRA scale
        lora_params = self.lora_config.get("lora_params", {})
        lora_alpha = lora_params.get("lora_alpha", 64)
        lora_rank = lora_params.get("lora_rank", 64)
        lora_scale = lora_alpha / lora_rank
        
        if hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters(["default"], [lora_scale])
        
        logger.info(f"LoRA weights loaded with scale: {lora_scale}")
    
    def setup_radial_attention(
        self,
        height: int,
        width: int,
        num_frames: int,
        dense_layers: Optional[int] = None,
        dense_timesteps: Optional[int] = None,
        decay_factor: Optional[float] = None,
        use_sage_attention: bool = False
    ):
        """Setup radial attention for the specified video dimensions"""
        
        # Use training config as defaults
        training_config = self.lora_config.get("training_config", {})
        
        dense_layers = dense_layers or training_config.get("dense_layers", 2)
        dense_timesteps = dense_timesteps or training_config.get("dense_timesteps", 2)
        decay_factor = decay_factor or training_config.get("decay_factor", 0.2)
        pattern = training_config.get("pattern", "radial")
        
        logger.info(f"Setting up {pattern} attention for {num_frames} frames")
        logger.info(f"Dense layers: {dense_layers}, Dense timesteps: {dense_timesteps}")
        logger.info(f"Decay factor: {decay_factor}")
        
        replace_wan_attention(
            self.pipe,
            height,
            width,
            num_frames,
            dense_layers,
            dense_timesteps,
            decay_factor,
            pattern,
            use_sage_attention,
        )
    
    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 768,
        width: int = 1280,
        num_frames: int = 161,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate a video with the LoRA model"""
        
        if seed is not None:
            set_seed(seed)
        
        # Setup radial attention for the video dimensions
        self.setup_radial_attention(height, width, num_frames, **kwargs)
        
        logger.info(f"Generating video: {num_frames} frames, {height}x{width}")
        logger.info(f"Prompt: {prompt}")
        
        # Generate video
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames[0]
        
        return output
    
    def evaluate_length_extension(
        self,
        prompts: list,
        base_frames: int = 69,
        extended_frames: int = 161,
        output_dir: str = "evaluation_outputs",
        **generate_kwargs
    ):
        """Evaluate length extension capability with multiple prompts"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Evaluating prompt {i+1}/{len(prompts)}: {prompt}")
            
            try:
                # Generate base length video
                base_video = self.generate_video(
                    prompt=prompt,
                    num_frames=base_frames,
                    **generate_kwargs
                )
                
                # Generate extended length video
                extended_video = self.generate_video(
                    prompt=prompt,
                    num_frames=extended_frames,
                    **generate_kwargs
                )
                
                # Save videos
                base_output_path = output_path / f"base_{i:03d}_frames_{base_frames}.mp4"
                extended_output_path = output_path / f"extended_{i:03d}_frames_{extended_frames}.mp4"
                
                export_to_video(base_video, str(base_output_path), fps=24)
                export_to_video(extended_video, str(extended_output_path), fps=24)
                
                result = {
                    "prompt": prompt,
                    "base_frames": base_frames,
                    "extended_frames": extended_frames,
                    "base_video_path": str(base_output_path),
                    "extended_video_path": str(extended_output_path),
                    "success": True
                }
                
                logger.info(f"Successfully generated videos for prompt {i+1}")
                
            except Exception as e:
                logger.error(f"Failed to generate videos for prompt {i+1}: {e}")
                result = {
                    "prompt": prompt,
                    "base_frames": base_frames,
                    "extended_frames": extended_frames,
                    "error": str(e),
                    "success": False
                }
            
            results.append(result)
        
        # Save evaluation results
        results_path = output_path / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {results_path}")
        
        # Print summary
        successful = sum(1 for r in results if r["success"])
        logger.info(f"Evaluation Summary: {successful}/{len(results)} successful")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA length extension model")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                       help="Base model ID")
    parser.add_argument("--lora_checkpoint_dir", type=str, required=True,
                       help="Directory containing LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs",
                       help="Output directory for generated videos")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str,
                       help="Single prompt to evaluate (if not provided, uses default prompts)")
    parser.add_argument("--negative_prompt", type=str, default="",
                       help="Negative prompt")
    parser.add_argument("--height", type=int, default=768,
                       help="Video height")
    parser.add_argument("--width", type=int, default=1280,
                       help="Video width")
    parser.add_argument("--base_frames", type=int, default=69,
                       help="Base number of frames")
    parser.add_argument("--extended_frames", type=int, default=161,
                       help="Extended number of frames")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Radial attention parameters
    parser.add_argument("--dense_layers", type=int, default=None,
                       help="Number of dense layers")
    parser.add_argument("--dense_timesteps", type=int, default=None,
                       help="Number of dense timesteps")
    parser.add_argument("--decay_factor", type=float, default=None,
                       help="Decay factor")
    parser.add_argument("--use_sage_attention", action="store_true",
                       help="Use SAGE attention")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = LoRALengthExtensionEvaluator(
        model_id=args.model_id,
        lora_checkpoint_dir=args.lora_checkpoint_dir
    )
    
    # Prepare generation parameters
    generate_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "dense_layers": args.dense_layers,
        "dense_timesteps": args.dense_timesteps,
        "decay_factor": args.decay_factor,
        "use_sage_attention": args.use_sage_attention,
    }
    
    # Prepare prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        # Default evaluation prompts
        prompts = [
            "A serene lake with mountains in the background, peaceful and calm",
            "A bustling city street with cars and people moving during rush hour",
            "A forest path with golden sunlight filtering through autumn trees",
            "Ocean waves crashing against rocky cliffs on a stormy day",
            "A field of colorful wildflowers swaying gently in the spring breeze",
        ]
    
    print(colored(f"Evaluating LoRA model: {args.lora_checkpoint_dir}", "green"))
    print(colored(f"Base frames: {args.base_frames}, Extended frames: {args.extended_frames}", "blue"))
    print(colored(f"Output directory: {args.output_dir}", "blue"))
    
    # Run evaluation
    results = evaluator.evaluate_length_extension(
        prompts=prompts,
        base_frames=args.base_frames,
        extended_frames=args.extended_frames,
        output_dir=args.output_dir,
        **generate_kwargs
    )
    
    print(colored("Evaluation completed!", "green"))
    print(colored(f"Check results in: {args.output_dir}", "yellow"))


if __name__ == "__main__":
    main()