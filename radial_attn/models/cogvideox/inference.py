import torch
from diffusers.models.attention_processor import Attention
from .attention import CogVideoXSparseAttnProcessor2_0, FusedCogVideoXSparseAttnProcessor2_0
from .sparse_transformer import replace_sparse_forward
from ...attn_mask import MaskMap

def replace_cogvideox_attention(
    pipe,
    height,
    width,
    num_frames,
    dense_layers=0,
    dense_timesteps=0,
    decay_factor=1.0,
    sparsity_type="radial",
    use_sage_attention=False,
    use_fused_attention=False,
):
    """
    替换CogVideoX模型中的注意力机制为径向稀疏注意力。
    
    Args:
        pipe: CogVideoX推理管道
        height: 视频帧高度
        width: 视频帧宽度
        num_frames: 视频帧数
        dense_layers: 使用密集注意力的层数
        dense_timesteps: 使用密集注意力的时间步数
        decay_factor: 径向衰减因子
        sparsity_type: 稀疏类型 ("radial" 或 "dense")
        use_sage_attention: 是否使用SAGE注意力
        use_fused_attention: 是否使用融合注意力处理器
    """
    
    # 计算视频token数量
    # CogVideoX的patch size通常是2，vae scale factor是8
    vae_scale_factor_spatial = getattr(pipe, 'vae_scale_factor_spatial', 8)
    vae_scale_factor_temporal = getattr(pipe, 'vae_scale_factor_temporal', 4)
    
    # 获取transformer配置
    transformer_config = pipe.transformer.config
    patch_size = getattr(transformer_config, 'patch_size', 2)
    if isinstance(patch_size, (list, tuple)):
        patch_size_t, patch_size_h, patch_size_w = patch_size
    else:
        patch_size_t = patch_size_h = patch_size_w = patch_size
    
    # 计算处理后的帧数和空间尺寸
    processed_frames = 1 + num_frames // (vae_scale_factor_temporal * patch_size_t)
    processed_height = height // (vae_scale_factor_spatial * patch_size_h)
    processed_width = width // (vae_scale_factor_spatial * patch_size_w)
    
    frame_size = processed_height * processed_width
    video_token_num = frame_size * processed_frames
    
    # 选择注意力处理器类
    if use_fused_attention:
        AttnModule = FusedCogVideoXSparseAttnProcessor2_0
        print("Using fused CogVideoX sparse attention processor")
    else:
        AttnModule = CogVideoXSparseAttnProcessor2_0
        print("Using standard CogVideoX sparse attention processor")
    
    # 设置类属性
    AttnModule.dense_block = dense_layers
    AttnModule.dense_timestep = dense_timesteps
    AttnModule.mask_map = MaskMap(video_token_num=video_token_num, num_frame=processed_frames)
    AttnModule.decay_factor = decay_factor
    AttnModule.sparse_type = sparsity_type
    AttnModule.use_sage_attention = use_sage_attention
    
    print(f"Replacing CogVideoX attention with {sparsity_type} attention")
    print(f"Video token num: {video_token_num}, processed frames: {processed_frames}")
    print(f"Frame size: {frame_size} ({processed_height}x{processed_width})")
    print(f"Dense layers: {dense_layers}, dense timesteps: {dense_timesteps}, decay factor: {decay_factor}")
    
    # 为transformer blocks中的注意力层设置layer_idx
    for layer_idx, block in enumerate(pipe.transformer.transformer_blocks):
        if hasattr(block, 'attn1') and hasattr(block.attn1, 'processor'):
            if hasattr(block.attn1.processor, 'layer_idx'):
                block.attn1.processor.layer_idx = layer_idx
    
    # 替换注意力处理器
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Attention):
            # 获取或设置layer_idx
            if hasattr(module.processor, 'layer_idx'):
                layer_idx = module.processor.layer_idx
            else:
                # 从模块名称中推断layer_idx
                layer_idx = 0
                if 'transformer_blocks' in name:
                    try:
                        layer_idx = int(name.split('transformer_blocks.')[1].split('.')[0])
                    except (IndexError, ValueError):
                        layer_idx = 0
            
            # 设置新的处理器
            module.set_processor(AttnModule(layer_idx))
            
    return pipe
