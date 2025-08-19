# CogVideoX 径向注意力实现

本实现为CogVideoX模型添加了径向稀疏注意力机制，可以显著减少视频生成时的计算复杂度和内存使用。

## 特性

- **径向稀疏注意力**: 基于帧间距离的衰减注意力模式
- **自适应密集/稀疏切换**: 根据时间步和层数动态选择注意力模式
- **SAGE注意力支持**: 可选的量化注意力优化
- **内存优化**: 支持模型CPU卸载和分片

## 文件结构

```
radial_attn/models/cogvideox/
├── __init__.py              # 模块初始化
├── attention.py             # 径向注意力处理器
├── inference.py             # 推理接口
└── sparse_transformer.py    # 稀疏transformer实现
```

## 主要组件

### 1. CogVideoXSparseAttnProcessor2_0

径向注意力处理器，基于CogVideoX的标准注意力处理器扩展：

- 支持timestep参数控制稀疏/密集注意力切换
- 实现径向衰减的注意力mask
- 兼容CogVideoX的旋转位置编码

### 2. 径向注意力策略

```python
def get_window_width(i, j, token_per_frame, sparse_type, num_frame, decay_factor=1, block_size=128, model_type="cogvideox"):
    dist = abs(i - j)
    if model_type == "cogvideox":
        if dist == 0:
            return token_per_frame  # 同一帧内的全连接
        elif dist == 1:
            return token_per_frame // 2  # 相邻帧半连接
        else:
            # 远距离帧使用径向衰减
            group = dist.bit_length()
            decay_length = 2 ** token_per_frame.bit_length() / 2 ** group * decay_factor
            return max(decay_length, block_size)
```

### 3. 稀疏Transformer块

扩展的CogVideoXBlock，支持：
- 稀疏注意力参数传递
- 与原始CogVideoXBlock的兼容性
- 正确的双输出格式（hidden_states, encoder_hidden_states）

## 使用方法

### 基本使用

```python
from radial_attn.models.cogvideox import replace_cogvideox_attention, replace_sparse_forward
from diffusers import CogVideoXPipeline

# 替换稀疏前向传播
replace_sparse_forward()

# 加载模型
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.bfloat16)

# 替换注意力机制
replace_cogvideox_attention(
    pipe,
    height=480,
    width=720,
    num_frames=49,
    dense_layers=2,      # 前2层使用密集注意力
    dense_timesteps=5,   # 前5个时间步使用密集注意力
    decay_factor=1.0,    # 径向衰减因子
    sparsity_type="radial",
    use_sage_attention=True,
)

# 生成视频
video = pipe(prompt="A beautiful landscape", num_frames=49).frames[0]
```

### 命令行使用

```bash
python cogvideox_t2v_inference.py \
    --model_id "THUDM/CogVideoX-2b" \
    --prompt "A cat walking in the garden" \
    --pattern "radial" \
    --dense_layers 2 \
    --dense_timesteps 5 \
    --decay_factor 1.0 \
    --use_sage_attention \
    --enable_model_cpu_offload
```

## 参数说明

### replace_cogvideox_attention 参数

- `pipe`: CogVideoX推理管道
- `height/width`: 视频分辨率
- `num_frames`: 视频帧数
- `dense_layers`: 使用密集注意力的层数（从第0层开始）
- `dense_timesteps`: 使用密集注意力的时间步数（从第0步开始）
- `decay_factor`: 径向衰减因子，控制远距离帧的注意力强度
- `sparsity_type`: "radial" 或 "dense"
- `use_sage_attention`: 是否使用SAGE量化注意力
- `use_fused_attention`: 是否使用融合注意力处理器

### 径向注意力机制

径向注意力通过以下策略减少计算复杂度：

1. **帧内注意力**: 同一帧内的tokens之间保持全连接
2. **相邻帧注意力**: 相邻帧之间使用部分连接
3. **远距离帧注意力**: 基于距离的指数衰减连接模式

这种设计保证了：
- 帧内的空间一致性
- 相邻帧的时间连续性
- 远距离帧的全局上下文（以较低的计算成本）

## 性能优化

### 内存优化选项

```python
# 模型CPU卸载
pipe.enable_model_cpu_offload()

# 或者顺序CPU卸载
pipe.enable_sequential_cpu_offload()
```

### SAGE注意力

启用SAGE注意力可以进一步优化量化推理：

```python
replace_cogvideox_attention(
    pipe, 
    use_sage_attention=True,  # 启用SAGE优化
    # 其他参数...
)
```

## 扩展功能

### 场景感知稀疏性

支持基于场景边界的动态稀疏性：

```python
from radial_attn.attn_mask import SceneAwareMaskMap

# 定义场景边界
scene_boundaries = [0, 15, 30, 45]  # 3个场景
scene_adjacency_matrix = [
    [True, True, False],   # 场景0可以关注场景0,1
    [True, True, True],    # 场景1可以关注所有场景
    [False, True, True]    # 场景2可以关注场景1,2
]

# 使用场景感知mask
mask_map = SceneAwareMaskMap(
    video_token_num=video_token_num,
    num_frame=num_frames,
    scene_boundaries=scene_boundaries,
    scene_adjacency_matrix=scene_adjacency_matrix
)
```

## 注意事项

1. **模型兼容性**: 当前实现针对CogVideoX-2b和CogVideoX-5b优化
2. **内存需求**: 径向注意力显著减少内存使用，但仍需要足够的GPU内存
3. **质量vs效率**: 通过调整`dense_layers`和`decay_factor`平衡生成质量和计算效率
4. **SAGE依赖**: 使用SAGE注意力需要安装相应的CUDA扩展

## 故障排除

### 常见问题

1. **CUDA内存不足**: 尝试启用CPU卸载或减少批次大小
2. **SAGE注意力错误**: 确保安装了spas_sage_attn扩展
3. **模型加载失败**: 检查模型路径和网络连接

### 性能调优

- 对于短视频（<30帧）：`dense_layers=1, dense_timesteps=3`
- 对于长视频（>50帧）：`dense_layers=3, dense_timesteps=8`
- 内存受限环境：启用CPU卸载和SAGE注意力

## 更新日志

- v1.0: 基础径向注意力实现
- v1.1: 添加SAGE注意力支持
- v1.2: 场景感知稀疏性功能
- v1.3: CogVideoX兼容性改进
