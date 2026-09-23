#!/usr/bin/env python3
# eval_experiment.py - 正式评估实验 v2

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

from rag_baseline import RAGBaseline
from lora_navigation import LoRANavigationRAG
from sentence_transformers import SentenceTransformer
import numpy as np

print("="*70)
print("🧪 正式评估实验：LoRA导航RAG vs 传统RAG")
print("="*70)

# 加载评估模型
print("\n📦 加载评估模型...")
eval_model = SentenceTransformer('all-MiniLM-L6-v2')

# 加载系统
rag_baseline = RAGBaseline()
rag_lora = LoRANavigationRAG(use_lora=True)

# 测试问题集 - 针对信息科学领域
TEST_CASES = [
    # 核心概念
    {"q": "what is information retrieval", "domain": "核心概念", "related": ["information retrieval", "retrieval"]},
    {"q": "information retrieval definition", "domain": "核心概念", "related": ["information retrieval", "retrieval"]},

    # 检索技术
    {"q": "boolean retrieval model", "domain": "检索技术", "related": ["boolean", "retrieval model"]},
    {"q": "vector space model", "domain": "检索技术", "related": ["vector", "space model"]},
    {"q": "probabilistic retrieval", "domain": "检索技术", "related": ["probabilistic", "retrieval"]},

    # 系统组件
    {"q": "how does indexing work", "domain": "系统组件", "related": ["indexing", "index"]},
    {"q": "what is a thesaurus in IR", "domain": "系统组件", "related": ["thesaurus", "vocabulary"]},
    {"q": "relevance feedback", "domain": "系统组件", "related": ["relevance", "feedback"]},

    # 信息行为
    {"q": "information seeking behavior", "domain": "信息行为", "related": ["information seeking", "behavior"]},
    {"q": "information needs of users", "domain": "信息行为", "related": ["information needs", "users"]},

    # 网络/语义
    {"q": "semantic web", "domain": "网络技术", "related": ["semantic web", "ontology"]},
    {"q": "metadata standards", "domain": "网络技术", "related": ["metadata", "standards"]},

    # 应用系统
    {"q": "digital library", "domain": "应用", "related": ["digital library", "digital"]},
    {"q": "web search engines", "domain": "应用", "related": ["search engine", "web"]},
]

print(f"📋 测试用例: {len(TEST_CASES)} 个\n")

# 评估函数
def semantic_similarity(text1, text2, model):
    """计算两个文本的语义相似度"""
    emb1 = model.encode([text1])
    emb2 = model.encode([text2])
    similarity = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
    return float(similarity)

def relevance_score(retrieved, related_terms, model):
    """计算检索结果与期望主题的相关性"""
    scores = []
    for term in related_terms:
        sim = semantic_similarity(retrieved, term, model)
        scores.append(sim)
    return max(scores) if scores else 0

# 执行评估
results = []
print("🔄 开始评估...\n")

for i, case in enumerate(TEST_CASES):
    q = case["q"]
    domain = case["domain"]
    related = case["related"]

    # 获取检索结果
    baseline_results = rag_baseline.retrieve(q, top_k=3)
    lora_keywords = rag_lora.lora_intuition(q)
    lora_results = rag_lora.retrieve(lora_keywords, top_k=3)

    # 合并结果（去重）
    baseline_text = " ".join(baseline_results)
    lora_text = " ".join(lora_results)

    # 计算相关性分数
    baseline_score = relevance_score(baseline_text, related, eval_model)
    lora_score = relevance_score(lora_text, related, eval_model)

    # 计算关键词质量
    kw_similarity = semantic_similarity(q, lora_keywords, eval_model)

    # 结果是否改变
    changed = baseline_text != lora_text

    # 分数变化
    score_diff = lora_score - baseline_score

    result = {
        "id": i + 1,
        "question": q,
        "domain": domain,
        "lora_keywords": lora_keywords[:50],
        "baseline_score": baseline_score,
        "lora_score": lora_score,
        "score_diff": score_diff,
        "kw_quality": kw_similarity,
        "changed": changed,
    }
    results.append(result)

    # 打印
    diff_symbol = "↑" if score_diff > 0.01 else ("↓" if score_diff < -0.01 else "=")
    print(f"[{i+1:2d}] {domain}: {q}")
    print(f"    LoRA关键词: {lora_keywords[:45]}...")
    print(f"    相关性: {baseline_score:.3f} → {lora_score:.3f} ({diff_symbol}{abs(score_diff):.3f})")
    print(f"    改变: {'✅' if changed else '❌'}")
    print()

# 统计
print("="*70)
print("📊 评估统计")
print("="*70)

total = len(results)
changed_count = sum(1 for r in results if r["changed"])
improved_count = sum(1 for r in results if r["score_diff"] > 0.01)
degraded_count = sum(1 for r in results if r["score_diff"] < -0.01)

avg_baseline = np.mean([r["baseline_score"] for r in results])
avg_lora = np.mean([r["lora_score"] for r in results])
avg_kw_quality = np.mean([r["kw_quality"] for r in results])

print(f"\n总测试用例: {total}")
print(f"结果改变: {changed_count} ({100*changed_count/total:.1f}%)")
print(f"相关性提升: {improved_count} ({100*improved_count/total:.1f}%)")
print(f"相关性下降: {degraded_count} ({100*degraded_count/total:.1f}%)")

print(f"\n平均相关性分数:")
print(f"  基线: {avg_baseline:.4f}")
print(f"  LoRA: {avg_lora:.4f} ({'+' if avg_lora > avg_baseline else ''}{avg_lora - avg_baseline:.4f})")

print(f"\n关键词质量 (与问题语义相似度): {avg_kw_quality:.4f}")

# 按领域统计
print(f"\n按领域统计:")
domain_stats = {}
for r in results:
    d = r["domain"]
    if d not in domain_stats:
        domain_stats[d] = {"total": 0, "improved": 0, "changed": 0}
    domain_stats[d]["total"] += 1
    if r["score_diff"] > 0.01:
        domain_stats[d]["improved"] += 1
    if r["changed"]:
        domain_stats[d]["changed"] += 1

for d, stats in domain_stats.items():
    improved_rate = 100 * stats["improved"] / stats["total"]
    print(f"  {d}: {stats['improved']}/{stats['total']} 提升 ({improved_rate:.0f}%)")

# 详细结果表
print(f"\n" + "="*70)
print("📋 详细结果")
print("="*70)
print(f"{'ID':<3} {'领域':<8} {'问题':<30} {'基线':<8} {'LoRA':<8} {'变化'}")
print("-"*70)
for r in results:
    diff = r["score_diff"]
    symbol = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "=")
    print(f"{r['id']:<3} {r['domain']:<8} {r['question'][:28]:<30} {r['baseline_score']:<8.3f} {r['lora_score']:<8.3f} {symbol}{abs(diff):.3f}")

# 结论
print(f"\n" + "="*70)
print("📝 结论")
print("="*70)
if avg_lora > avg_baseline:
    print(f"✅ LoRA导航RAG 平均相关性提升: {avg_lora - avg_baseline:.4f}")
else:
    print(f"⚠️ LoRA导航RAG 平均相关性下降: {avg_baseline - avg_lora:.4f}")

if improved_count > degraded_count:
    print(f"✅ 提升案例多于下降案例 ({improved_count} vs {degraded_count})")
else:
    print(f"⚠️ 下降案例多于提升案例 ({degraded_count} vs {improved_count})")

print(f"\n关键词生成质量: {avg_kw_quality:.4f}")
if avg_kw_quality > 0.5:
    print("✅ 关键词与问题语义高度相关")
elif avg_kw_quality > 0.3:
    print("⚠️ 关键词与问题语义中等相关")
else:
    print("❌ 关键词与问题语义较弱")
