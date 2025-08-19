# this is the setting for 1x length T2V inference
# dense_layers=1
# dense_timesteps=12

prompt=$(cat examples/prompt.txt)

# CUDA_VISIBLE_DEVICES=1 python wan_t2v_inference.py \
#     --model_id "/data/wangshu/wangshu_code/Wan2.1/Wan2.1-T2V-1.3B-Diffusers/" \
#     --prompt "$prompt" \
#     --height 768 \
#     --width 1280 \
#     --num_frames 129 \
#     --dense_layers $dense_layers \
#     --dense_timesteps $dense_timesteps \
#     --decay_factor 0.2 \
#     --pattern "radial" \
#     --use_model_offload

# this is the setting for 2x length T2V inference
dense_layers=2
dense_timesteps=2

CUDA_VISIBLE_DEVICES=1 python wan_t2v_inference.py \
    --model_id "/data/wangshu/wangshu_code/Wan2.1/Wan2.1-T2V-1.3B-Diffusers/" \
    --prompt "$prompt" \
    --height 720 \
    --width 1280 \
    --num_frames 161 \
    --pattern "radial" \
    --dense_layers $dense_layers \
    --dense_timesteps $dense_timesteps \
    --use_model_offload