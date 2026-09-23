# LoRA导航增强RAG系统

## 系统概述

本项目实现**双系统对比实验**，验证"LoRA语义压缩层是否能提升RAG检索效果"。

### 系统A（实验组）：LoRA导航 + RAG

```
问题 → LoRA语义压缩 → 关键词生成 → 向量检索 → 原文 → 主模型总结
```

### 系统B（对照组）：传统RAG

```
问题 → 直接向量检索 → 原文 → 主模型总结
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 处理PDF

```bash
python pdf_to_chunks.py
```

### 3. 构建向量索引

```bash
python build_index.py
```

### 4. 运行系统

**传统RAG（对照组）：**
```bash
python rag_baseline.py
```

**LoRA导航RAG（实验组）：**
```bash
python lora_navigation.py
```

---

## 实验方法

### 测试问题示例

准备10个测试问题，包括：

| 类型 | 示例问题 |
|------|---------|
| 模糊问题 | "how water becomes clouds" |
| 同义改写 | "process of condensation in atmosphere" |
| 跨章节问题 | 信息检索与信息行为的联系 |

### 对比记录表

| 问题 | RAG命中 | LoRA命中 | 备注 |
|------|---------|----------|------|
| Q1 | ❌/✅  | ❌/✅   |      |

---

## 观察指标

### ✅ LoRA优势信号

- 能处理"表达不同但语义相同"的问题
- 更容易命中隐藏信息
- Top-1 结果更准确

### ❌ 失败信号

- LoRA乱生成关键词
- 检索结果变差
- 引入噪声

---

## 文件结构

```
OpenLocalExpert/
├── pdf_to_chunks.py      # PDF加载与分块
├── build_index.py        # 向量索引构建
├── rag_baseline.py       # 系统B：传统RAG
├── lora_navigation.py     # 系统A：LoRA导航+RAG
├── requirements.txt      # 依赖
└── README.md             # 本文档
```

---

## 技术栈

- **向量模型**: `all-MiniLM-L6-v2`
- **LoRA替身**: `google/flan-t5-small`
- **向量数据库**: FAISS (CPU版本)
- **PDF处理**: PyMuPDF

所有组件均可在**无GPU环境**下运行。
