#!/usr/bin/env python3
# compare_all.py - 三系统对比测试

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, '.')

# 测试问题
TEST_QUESTIONS = [
    "what is information retrieval",
    "semantic web technologies",
    "how does indexing work",
]

print("="*70)
print("🧪 三系统对比测试")
print("="*70)

# =====================
# 系统B: 传统RAG
# =====================
print("\n📊 系统B: 传统RAG (对照组)")
print("-"*50)

from rag_baseline import RAGBaseline
rag_baseline = RAGBaseline()

# =====================
# 系统A (v1): 关键词提取
# =====================
print("\n📊 系统A-v1: 关键词提取")
print("-"*50)

from lora_navigation import LoRANavigationRAG
rag_keyword = LoRANavigationRAG(use_lora=False)

# =====================
# 系统A (v2): MLX LoRA
# =====================
print("\n📊 系统A-v2: MLX LoRA导航")
print("-"*50)

rag_lora = LoRANavigationRAG(use_lora=True)

# =====================
# 对比测试
# =====================
print("\n" + "="*70)
print("📋 对比结果")
print("="*70)

for q in TEST_QUESTIONS:
    print(f"\n❓ 问题: {q}")
    print("="*50)

    # 系统B
    results_b = rag_baseline.retrieve(q, top_k=1)
    print(f"\n【传统RAG】")
    print(f"   → {results_b[0][:80]}...")

    # 系统A-v1
    kw1 = rag_keyword.lora_intuition(q)
    results_a1 = rag_keyword.retrieve(kw1, top_k=1)
    print(f"\n【关键词提取】关键词: {kw1}")
    print(f"   → {results_a1[0][:80]}...")

    # 系统A-v2
    kw2 = rag_lora.lora_intuition(q)
    results_a2 = rag_lora.retrieve(kw2, top_k=1)
    print(f"\n【MLX LoRA】关键词: {kw2}")
    print(f"   → {results_a2[0][:80]}...")

    # 差异分析
    same_b_a1 = results_b[0] == results_a1[0]
    same_b_a2 = results_b[0] == results_a2[0]
    print(f"\n【差异】RAG vs 关键词: {'相同' if same_b_a1 else '不同'}")
    print(f"       RAG vs LoRA: {'相同' if same_b_a2 else '不同'}")
