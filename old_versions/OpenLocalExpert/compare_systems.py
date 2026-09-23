#!/usr/bin/env python3
# compare_systems.py - 多问题对比测试

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, '.')

# 测试问题列表
TEST_QUESTIONS = [
    "what is information retrieval",
    "how does condensation work",
    "what is the information search process",
    "semantic web technologies",
]

print("="*70)
print("🧪 双系统对比测试 - 多问题")
print("="*70)

# 加载系统
from rag_baseline import RAGBaseline
from lora_navigation import LoRANavigationRAG

rag_baseline = RAGBaseline()
rag_lora = LoRANavigationRAG()

results_log = []

for q in TEST_QUESTIONS:
    print(f"\n{'='*70}")
    print(f"❓ 问题: {q}")
    print("="*70)

    # 系统B
    results_b = rag_baseline.retrieve(q, top_k=1)
    print(f"\n📊 传统RAG Top-1: ...{results_b[0][:80]}...")

    # 系统A
    keywords = rag_lora.lora_intuition(q)
    results_a = rag_lora.retrieve(keywords, top_k=1)
    print(f"🧠 LoRA关键词: {keywords}")
    print(f"🧠 LoRA导航 Top-1: ...{results_a[0][:80]}...")

    # 记录
    same = results_b[0] == results_a[0]
    results_log.append({
        'question': q,
        'lora_keywords': keywords,
        'same': same
    })

print("\n" + "="*70)
print("📋 测试总结")
print("="*70)
for r in results_log:
    status = "相同" if r['same'] else "不同"
    print(f"[{status}] {r['question']}")
    print(f"       LoRA关键词: {r['lora_keywords']}")
