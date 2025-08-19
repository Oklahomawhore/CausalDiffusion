# 从CogVideoX项目借鉴并修改的稀疏transformer实现

from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock, CogVideoXTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)


class CogVideoXBlock_Sparse(CogVideoXBlock):
    """
    支持稀疏注意力的CogVideoX Transformer块。
    扩展了原始的CogVideoXBlock以支持timestep和numeral_timestep参数。
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        timestep: Optional[int] = None,
        numeral_timestep: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，支持稀疏注意力参数。
        
        Args:
            hidden_states: 输入的隐藏状态
            encoder_hidden_states: 编码器的隐藏状态
            temb: 时间嵌入
            image_rotary_emb: 图像旋转位置编码
            attention_kwargs: 注意力相关参数
            timestep: 当前时间步（用于决定是否使用稀疏注意力）
            numeral_timestep: 数值时间步
        """
        text_seq_length = encoder_hidden_states.size(1)
        
        # 准备注意力参数
        attn_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        attn_kwargs.update({
            "timestep": timestep,
            "numeral_timestep": numeral_timestep,
        })
        
        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attn_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class CogVideoXTransformer3DModel_Sparse(CogVideoXTransformer3DModel):
    """
    支持稀疏注意力的CogVideoX 3D Transformer模型。
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        numeral_timestep: Optional[int] = None,
    ):
        """
        前向传播，支持稀疏注意力。
        
        Args:
            numeral_timestep: 数值时间步，用于稀疏注意力调度
        """
        
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # 为每个PEFT层设置lora_scale权重
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. 时间嵌入
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps不包含权重，总是返回f32张量
        # 但time_embedding可能在fp16中运行，所以需要在这里转换类型
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch嵌入
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer块
        for i, block in enumerate(self.transformer_blocks):
            # 准备注意力参数，包含稀疏注意力相关信息
            sparse_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
            sparse_attention_kwargs.update({
                "timestep": i,
                "numeral_timestep": numeral_timestep,
            })
            
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    sparse_attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=sparse_attention_kwargs,
                )

        hidden_states = self.norm_final(hidden_states)

        # 4. 最终块
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. 去patch化
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # 移除每个PEFT层的lora_scale
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def replace_sparse_forward():
    """
    替换CogVideoX模型的前向传播方法以支持稀疏注意力。
    """
    # 替换CogVideoXBlock的forward方法
    CogVideoXBlock.forward = CogVideoXBlock_Sparse.forward
    
    # 替换CogVideoXTransformer3DModel的forward方法
    CogVideoXTransformer3DModel.forward = CogVideoXTransformer3DModel_Sparse.forward
    
    print("CogVideoX sparse forward methods replaced successfully!")
