# 实验一：语境蒸馏 (Contextual Distillation) - 进度记录
**日期**: 2026-04-16
**状态**: 调试中（Phase 1 未完成）

---

## 目标
打破 34% Top-1 准确率天花板，预期达到 55-65%

## 已完成的工作

### Phase 1: 蒸馏工厂基础设施 ✅
- `src/distill_factory.py` 已创建（~32KB，32401 bytes）
- 核心功能：
  - 异步批处理 LLM 调用（subprocess ollama CLI 模式）
  - 4 种维度问题生成：factual、fuzzy、inferential、hard_negative
  - LCS 字面重合度过滤（阈值 70%）
  - JSON 健壮性解析（处理 markdown 包裹、废话前缀）
  - Hard Negative 专门 enrichment 流程
  - 标签平衡逻辑（NOT_RELEVANT 目标 15-20%）

### 修复的问题
1. 语法错误：单引号嵌套正则表达式 → 改用双引号包裹
2. aiohttp 兼容性问题 → 改用 subprocess 调用 ollama CLI
3. Qwen reasoning 模型 content 为空 → 尝试从 reasoning 字段读取

---

## 当前阻塞问题

### Ollama / 模型兼容性
```
ollama serve 运行中，模型: qwen3.5:4b
Qwen3.5 将输出放在 "reasoning" 字段而非 "content"
subprocess 调用 `ollama generate --json` 尚未验证成功
```

**可选方案：**
1. 继续调试 qwen3.5:4b（4B 参数，可能质量不足）
2. 拉取 qwen2.5:7b-instruct（推荐，需要 ~4GB 下载）
3. 使用 OpenAI API（需 API key）

---

## 下一步操作

### 立即执行（恢复时）
```bash
# 1. 测试 ollama CLI 调用的正确方式
ollama generate --json -p '{"model":"qwen3.5:4b","messages":[{"role":"user","content":"Hi"}],"options":{"temperature":0.8,"num_predict":100}}' 2>&1

# 2. 如果上面失败，下载 qwen2.5:7b
ollama pull qwen2.5:7b-instruct

# 3. 运行 preview 测试
cd lora_chunk_router
python3 -m src.distill_factory --mode preview --num_samples 3

# 4. 全量蒸馏（确认质量后）
python3 -m src.distill_factory --mode distill --batch_size 4 --concurrency 4
```

---

## 项目结构（蒸馏相关）

```
lora_chunk_router/
├── src/
│   ├── distill_factory.py   # ⭐ 核心蒸馏工厂（新）
│   ├── preprocess.py         # 原数据预处理
│   ├── train_lora.py         # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   ├── infer_lora.py         # 推理脚本
│   └── vector_search.py      # 向量搜索基线
├── data/
│   ├── train_distilled.jsonl  # ⭐ 蒸馏数据输出（待生成）
│   ├── train.jsonl             # 原训练数据
│   ├── test.json              # 测试数据
│   └── chunks.json            # chunk 数据
├── experiments/
│   ├── 2026-04-16_experiment_results.json
│   ├── chunks.json            # 497 chunks
│   └── PROGRESS_CONTEXTUAL_DISTILLATION.md  # 本文件
└── configs/
    └── config.yaml
```

---

## 蒸馏流水线设计（供参考）

### 4 种问题维度
1. **factual** - 关于事实的具体问题
2. **fuzzy** - 口语化/模糊问题
3. **inferential** - 需要推理的问题
4. **hard_negative** - 使用文本关键词但主题无关的问题 → 标签为 NOT_RELEVANT

### 训练配置变更（Phase 4）
- Label Smoothing: 0.1
- LoRA Rank: 16（原来 8）
- 置信度闸门: if confidence < 0.6 → NOT_RELEVANT

### 预期指标提升
| 指标 | 当前 | 目标 |
|------|------|------|
| Top-1 | 34.21% | 55-65% |
| NOT_RELEVANT | 0% | >70% |
