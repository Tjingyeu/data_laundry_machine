# M1 Pro LoRA 训练指南 (16GB 内存)

## 环境概览

| 项目 | 配置 |
|------|------|
| 芯片 | Apple M1 Pro |
| 内存 | 16GB Unified |
| 加速 | Metal GPU (MPS) |
| 方案 | 推荐量化 + LoRA |

---

## 快速开始

### Step 1: 安装依赖

```bash
pip install torch transformers datasets accelerate
pip install bitsandbytes safetensors peft
pip install scipy scikit-learn
```

> 注意：M1 Pro 不需要安装 CUDA 相关包

### Step 2: 生成训练数据

```bash
python build_lora_dataset.py
```

这会从你的 PDF chunks 生成 `train.jsonl`

### Step 3: 开始训练

```bash
python train_lora_m1.py
```

选择方案：
- **方案1 (MLX)**: 苹果官方加速，需要安装 `mlx` 库
- **方案2 (量化)**: 推荐，使用 4bit 量化 + LoRA
- **方案3 (CPU)**: 最保守，但慢

### Step 4: 接入系统

训练完成后：

```bash
python lora_navigation.py
```

---

## M1 Pro 内存优化技巧

### 1. 使用小模型起步

| 模型 | 内存需求 | 推荐度 |
|------|---------|--------|
| Qwen2.5-0.5B | ~2GB | ⭐⭐⭐⭐⭐ |
| Qwen2.5-1.5B | ~4GB | ⭐⭐⭐⭐ |
| Qwen2.5-3B | ~8GB | ⭐⭐⭐ |

### 2. 4bit 量化

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

### 3. 梯度累积

batch_size=1 但 gradient_accumulation_steps=8
效果等同 batch_size=8

---

## 预期训练时间

| 模型 | 100 steps | 500 steps |
|------|-----------|-----------|
| 0.5B | ~15分钟 | ~1.5小时 |
| 1.5B | ~30分钟 | ~3小时 |
| 3B | ~1小时 | ~6小时 |

---

## 验证训练效果

### 对比测试

```bash
python compare_systems.py
```

观察：
- LoRA关键词是否更"像书"
- 检索是否更精准
- 模糊问题是否改善

---

## 常见问题

### Q: 内存不足怎么办？

A: 使用更小的模型 (0.5B) 或增加 gradient_accumulation_steps

### Q: MLX 安装失败？

A: 使用方案2 (量化方案)

### Q: 训练很慢？

A: 正常，CPU训练本来就是这样。可以减少 max_steps 先测试。

---

## 下一步

训练成功后，可以升级到：

1. **概率导航**: LoRA输出 confidence score
2. **多路径验证**: top1不匹配时验证top2
3. **混合检索**: 向量 + LoRA关键词 组合
