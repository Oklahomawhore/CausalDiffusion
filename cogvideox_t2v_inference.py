import os
import argparse

import torch
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler
from diffusers.utils import export_to_video

from radial_attn.utils import set_seed
from radial_attn.models.cogvideox.inference import replace_cogvideox_attention
from radial_attn.models.cogvideox.sparse_transformer import replace_sparse_forward


def main():
    parser = argparse.ArgumentParser(description="Generate video from text prompt using CogVideoX with radial attention")
    parser.add_argument("--model_id", type=str, default="THUDM/CogVideoX-2b", help="Model ID to use for generation")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")
    parser.add_argument("--height", type=int, default=480, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=720, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--output_file", type=str, default="cogvideox_output.mp4", help="Output video file name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--pattern", type=str, default="dense", choices=["radial", "dense"], help="Attention pattern to use")
    parser.add_argument("--dense_layers", type=int, default=0, help="Number of dense layers to use in the attention")
    parser.add_argument("--dense_timesteps", type=int, default=0, help="Number of dense timesteps to use in the attention")
    parser.add_argument("--decay_factor", type=float, default=1.0, help="Decay factor for the radial attention")
    parser.add_argument("--use_sage_attention", action="store_true", help="Use SAGE attention for optimized inference")
    parser.add_argument("--use_fused_attention", action="store_true", help="Use fused attention processor")
    parser.add_argument("--enable_model_cpu_offload", action="store_true", help="Enable model CPU offloading for memory efficiency")
    parser.add_argument("--enable_sequential_cpu_offload", action="store_true", help="Enable sequential CPU offloading")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 如果使用径向注意力，替换前向传播方法
    if args.pattern == "radial":
        replace_sparse_forward()
    
    # 加载模型
    print(f"Loading CogVideoX model: {args.model_id}")
    pipe = CogVideoXPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16
    )
    
    # 设置调度器
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config)
    
    # 启用内存优化
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
        print("Model CPU offloading enabled")
    elif args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
        print("Sequential CPU offloading enabled")
    else:
        pipe = pipe.to("cuda")
    
    # 设置默认提示词
    if args.prompt is None:
        # print(colored("Using default prompt", "red"))
        args.prompt = "A young woman with beautiful and clear eyes and blonde hair stands and looks at the camera. A man walks up and speaks to her but she turns away. The light is dim but the girl's eyes are clearly visible."
    
    if args.negative_prompt is None:
        args.negative_prompt = "Very blurry, bad quality, distorted, low resolution, watermark, signature, text"

    print("=" * 20 + " Prompts " + "=" * 20)
    print(f"Prompt: {args.prompt}\n\n" + f"Negative Prompt: {args.negative_prompt}")

    # 如果使用径向注意力，替换注意力机制
    if args.pattern == "radial":
        replace_cogvideox_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            args.dense_layers,
            args.dense_timesteps,
            args.decay_factor,
            args.pattern,
            args.use_sage_attention,
            args.use_fused_attention,
        )
    
    print(f"Generating video with {args.pattern} attention...")
    
    # 生成视频
    with torch.no_grad():
        video = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).frames[0]
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # 导出视频
    export_to_video(video, args.output_file, fps=8)
    print(f"Video saved to: {args.output_file}")


if __name__ == "__main__":
    main()
