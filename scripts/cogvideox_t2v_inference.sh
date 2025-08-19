#!/bin/bash

# CogVideoX T2V 推理脚本，使用径向注意力

MODEL_ID="/data/wangshu/wangshu_code/CogVideo/finetune/checkpoints/CogVideoX-2b"
PROMPT="A young woman with beautiful and clear eyes and blonde hair stands and looks at the camera. A man walks up and speaks to her but she turns away. The light is dim but the girl's eyes are clearly visible."
OUTPUT_FILE="outputs/cogvideox_radial_output.mp4"

CUDA_VISIBLE_DEVICES=5 python cogvideox_t2v_inference.py \
    --model_id "$MODEL_ID" \
    --prompt "$PROMPT" \
    --height 480 \
    --width 720 \
    --num_frames 49 \
    --num_inference_steps 50 \
    --output_file "$OUTPUT_FILE" \
    --seed 42 \
    --guidance_scale 6.0 \
    --pattern "radial" \
    --dense_layers 2 \
    --dense_timesteps 5 \
    --decay_factor 1.0 \
    --use_sage_attention \
    --enable_model_cpu_offload
