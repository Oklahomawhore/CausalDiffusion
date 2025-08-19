from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import rearrange
from ...attn_mask import RadialAttention
from torch.nn.attention import sdpa_kernel, SDPBackend

class CogVideoXSparseAttnProcessor2_0:
    """
    径向注意力处理器，专为CogVideoX模型设计。
    实现了基于径向衰减的稀疏注意力机制，可以显著减少计算复杂度。
    """
    mask_map = None
    dense_timestep = 0
    dense_block = 0
    decay_factor = 1.0
    sparse_type = "radial"  # 默认使用径向注意力，可设置为"dense"使用密集注意力
    use_sage_attention = False
    
    def __init__(self, layer_idx):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXSparseAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self.layer_idx = layer_idx
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        numeral_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # 将文本和图像特征拼接
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 应用旋转位置编码（RoPE）
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # 根据timestep决定使用密集注意力还是稀疏注意力
        if (timestep is None or 
            numeral_timestep < self.dense_timestep or 
            self.layer_idx < self.dense_block or 
            self.sparse_type == "dense"):
            # 使用密集注意力
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
        else:
            # 使用径向稀疏注意力
            batch_size_orig = query.shape[0]
            
            # 重新排列张量形状以适应RadialAttention函数
            # (batch_size, num_heads, seq_len, head_dim) -> (seq_len * batch_size, num_heads, head_dim)
            query = rearrange(query, "b h s d -> (b s) h d")
            key = rearrange(key, "b h s d -> (b s) h d")
            value = rearrange(value, "b h s d -> (b s) h d")
            
            # 应用径向注意力
            hidden_states = RadialAttention(
                query=query, 
                key=key, 
                value=value, 
                mask_map=self.mask_map, 
                sparsity_type="radial", 
                block_size=128, 
                decay_factor=self.decay_factor, 
                model_type="cogvideox", 
                pre_defined_mask=None, 
                use_sage_attention=self.use_sage_attention
            )
            
            # 重新排列回原始形状
            # (seq_len * batch_size, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
            hidden_states = rearrange(hidden_states, "(b s) h d -> b h s d", b=batch_size_orig)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # 线性投影
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # 分离文本和图像特征
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class FusedCogVideoXSparseAttnProcessor2_0:
    """
    融合版本的CogVideoX径向注意力处理器，使用融合的QKV投影。
    """
    mask_map = None
    dense_timestep = 0
    dense_block = 0
    decay_factor = 1.0
    sparse_type = "radial"
    use_sage_attention = False
    
    def __init__(self, layer_idx):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FusedCogVideoXSparseAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        numeral_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # 使用融合的QKV投影
        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = torch.split(qkv, split_size, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 应用旋转位置编码
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # 根据timestep决定使用密集注意力还是稀疏注意力
        if (timestep is None or 
            numeral_timestep < self.dense_timestep or 
            self.layer_idx < self.dense_block or 
            self.sparse_type == "dense"):
            # 使用密集注意力
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            # 使用径向稀疏注意力
            batch_size_orig = query.shape[0]
            
            # 重新排列张量形状
            query = rearrange(query, "b h s d -> (b s) h d")
            key = rearrange(key, "b h s d -> (b s) h d")
            value = rearrange(value, "b h s d -> (b s) h d")
            
            # 应用径向注意力
            hidden_states = RadialAttention(
                query=query, 
                key=key, 
                value=value, 
                mask_map=self.mask_map, 
                sparsity_type="radial", 
                block_size=128, 
                decay_factor=self.decay_factor, 
                model_type="cogvideox", 
                pre_defined_mask=None, 
                use_sage_attention=self.use_sage_attention
            )
            
            # 重新排列回原始形状
            hidden_states = rearrange(hidden_states, "(b s) h d -> b h s d", b=batch_size_orig)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # 线性投影
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states
