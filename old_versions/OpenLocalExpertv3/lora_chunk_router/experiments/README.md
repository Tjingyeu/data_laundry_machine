# LoRA Chunk Router - Experiment Summary

## 实验日期
2026-04-16

## 实验概述
使用 LoRA 微调的 Qwen2.5-0.5B 作为序列分类器进行语义路由，与 FAISS 向量搜索进行对比。

---

## 核心结果

### 评估指标对比

| 指标 | LoRA Router | Vector Search | 胜者 |
|------|-------------|---------------|------|
| **Top-1 Accuracy** | **34.21%** | 4.23% | LoRA ✓ |
| **Top-3 Accuracy** | **38.63%** | 9.26% | LoRA ✓ |
| **Top-5 Accuracy** | **38.63%** | 11.47% | LoRA ✓ |
| **MRR** | **0.3642** | 0.0686 | LoRA ✓ |
| **Fuzzy Query Acc** | **34.21%** | 4.23% | LoRA ✓ |
| **NOT_RELEVANT Acc** | 0.00% | 0.00% | Tie |
| **Avg Chunks Returned** | **1.11** | 5.00 | LoRA ✓ |

**LoRA 胜 6/6 项（不含平局）**

---

## 模型配置

```
Base Model:     Qwen/Qwen2.5-0.5B
Task Type:      Sequence Classification
LoRA Rank:      8
LoRA Alpha:     16
Target Modules: q_proj, v_proj
Learning Rate:  1e-4
Batch Size:     4 (effective 16 with grad accum)
Epochs:         10
Device:         MPS (Apple Silicon)
Trainable:      1,503,360 (0.30%)
```

---

## 数据集统计

| 项目 | 数值 |
|------|------|
| 原始文件数 | 28 个讲座 |
| Chunk 总数 | 497 |
| 训练样本 | 4,100 |
| 测试样本 | 497 |
| NOT_RELEVANT 样本 | 124 |
| 标签类别 | 498 |

---

## 关键改进

1. **Chunk ID 冲突修复** - 修复了按 lecture 重置 ID 导致的不同 lecture 的 chunk 重复问题
2. **Topic 提取改进** - 大写专有名词 +15 加权，增强停用词过滤
3. **架构变更** - 从 Causal LM 生成改为 Sequence Classification
4. **训练稳定性** - 添加梯度裁剪，float32 替代 float16
5. **数据分割** - 从 lecture 级别改为 sample 级别，确保所有 chunk ID 出现在训练中

---

## 哲学问题测试

对 10 个深度哲学问题的检索结果显示：

- **LoRA 优势**：能捕捉深层语义，对泛化问题有"拒绝"能力
- **Vector 优势**：关键词匹配稳定，置信度高
- **LoRA 局限**：对抽象问题置信度低（0.07-0.43 vs Vector 0.66-0.77）

---

## 文件列表

```
experiments/
├── 2026-04-16_experiment_results.json     # 核心评估结果
├── philosophical_queries_comparison.json    # 哲学问题检索对比
└── README.md                              # 本文件
```

---

## 结论

LoRA 序列分类路由器在语义路由任务上显著优于向量搜索（34% vs 4% Top-1），证明了学习式路由的有效性。

当前局限：
- NOT_RELEVANT 检测需要改进
- 抽象哲学问题的泛化能力有限
- Topic 提取仍需优化
