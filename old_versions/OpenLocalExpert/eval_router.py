#!/usr/bin/env python3
# eval_router.py - 路由LoRA vs 传统RAG 正式评估

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

from rag_baseline import RAGBaseline
from lora_navigation import LoRANavigationRAG
from sentence_transformers import SentenceTransformer
import numpy as np

print("="*70)
print("🧪 正式评估：路由LoRA vs 传统RAG")
print("="*70)

# 加载评估模型
print("\n📦 加载评估模型...")
eval_model = SentenceTransformer('all-MiniLM-L6-v2')

# 加载系统
rag_baseline = RAGBaseline()
rag_lora = LoRANavigationRAG(use_lora=True)

# 测试问题集
TEST_CASES = [
    {"q": "what is information retrieval", "related": ["information retrieval", "retrieval"]},
    {"q": "information retrieval definition", "related": ["information retrieval", "retrieval"]},
    {"q": "boolean retrieval model", "related": ["boolean", "retrieval"]},
    {"q": "vector space model", "related": ["vector", "space model"]},
    {"q": "probabilistic retrieval", "related": ["probabilistic", "retrieval"]},
    {"q": "how does indexing work", "related": ["indexing", "index"]},
    {"q": "what is a thesaurus", "related": ["thesaurus", "vocabulary"]},
    {"q": "relevance feedback", "related": ["relevance", "feedback"]},
    {"q": "information seeking behavior", "related": ["information seeking", "behavior"]},
    {"q": "semantic web", "related": ["semantic web", "ontology"]},
    {"q": "metadata standards", "related": ["metadata", "standards"]},
    {"q": "digital library", "related": ["digital library"]},
    {"q": "web search engines", "related": ["search engine"]},
    {"q": "information needs", "related": ["information needs", "users"]},
]

print(f"📋 测试用例: {len(TEST_CASES)} 个\n")

def semantic_similarity(text1, text2, model):
    emb1 = model.encode([text1])
    emb2 = model.encode([text2])
    sim = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
    return float(sim)

def relevance_score(retrieved, related_terms, model):
    scores = []
    for term in related_terms:
        sim = semantic_similarity(retrieved, term, model)
        scores.append(sim)
    return max(scores) if scores else 0

results = []
print("🔄 开始评估...\n")

for i, case in enumerate(TEST_CASES):
    q = case["q"]
    related = case["related"]

    # 传统RAG
    baseline_results = rag_baseline.retrieve(q, top_k=3)
    baseline_text = " ".join(baseline_results)

    # LoRA路由
    routed_content, chunk_id = rag_lora.ask(q, top_k=1)

    if chunk_id is not None:
        routed_text = routed_content if routed_content else ""
    else:
        routed_text = baseline_text  # fallback

    # 计算相关性
    baseline_score = relevance_score(baseline_text, related, eval_model)
    routed_score = relevance_score(routed_text, related, eval_model)

    score_diff = routed_score - baseline_score

    result = {
        "question": q,
        "chunk_id": chunk_id,
        "baseline_score": baseline_score,
        "routed_score": routed_score,
        "score_diff": score_diff,
    }
    results.append(result)

    diff_symbol = "↑" if score_diff > 0.01 else ("↓" if score_diff < -0.01 else "=")
    print(f"[{i+1:2d}] {q}")
    print(f"    Chunk: {f'Chunk_{chunk_id:04d}' if chunk_id else 'N/A'} | 基线: {baseline_score:.3f} | 路由: {routed_score:.3f} ({diff_symbol}{abs(score_diff):.3f})")
    print()

# 统计
print("="*70)
print("📊 评估统计")
print("="*70)

total = len(results)
routed_count = sum(1 for r in results if r["chunk_id"] is not None)
improved = sum(1 for r in results if r["score_diff"] > 0.01)
degraded = sum(1 for r in results if r["score_diff"] < -0.01)

avg_baseline = np.mean([r["baseline_score"] for r in results])
avg_routed = np.mean([r["routed_score"] for r in results])

print(f"\n总测试用例: {total}")
print(f"成功路由: {routed_count} ({100*routed_count/total:.1f}%)")
print(f"相关性提升: {improved} ({100*improved/total:.1f}%)")
print(f"相关性下降: {degraded} ({100*degraded/total:.1f}%)")

print(f"\n平均相关性分数:")
print(f"  传统RAG: {avg_baseline:.4f}")
print(f"  路由LoRA: {avg_routed:.4f} ({'+' if avg_routed > avg_baseline else ''}{avg_routed - avg_baseline:.4f})")

# 分析路由分布
chunk_ids = [r["chunk_id"] for r in results if r["chunk_id"] is not None]
if chunk_ids:
    from collections import Counter
    chunk_freq = Counter(chunk_ids)
    print(f"\n路由Chunk分布:")
    for chunk, count in chunk_freq.most_common(5):
        print(f"  Chunk_{chunk:04d}: {count}次 ({100*count/len(chunk_ids):.1f}%)")

print(f"\n结论:")
if avg_routed > avg_baseline + 0.02:
    print(f"✅ 路由LoRA优于传统RAG (+{avg_routed - avg_baseline:.4f})")
elif avg_routed < avg_baseline - 0.02:
    print(f"⚠️ 传统RAG优于路由LoRA ({avg_routed - avg_baseline:.4f})")
else:
    print(f"➡️ 两者效果接近")
