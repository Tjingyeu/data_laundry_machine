# build_router_dataset.py
# 路由式训练数据生成器

import json
import os
import re
from collections import Counter
import random

CHUNKS_FILE = "chunks.txt"
OUTPUT_FILE = "train_router.jsonl"

# 停用词
STOPWORDS = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
             'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
             'would', 'could', 'should', 'may', 'might', 'must', 'shall',
             'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
             'by', 'from', 'as', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'between', 'under', 'again',
             'further', 'then', 'once', 'what', 'which', 'who', 'whom',
             'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
             'their', 'we', 'our', 'you', 'your', 'he', 'she', 'him',
             'her', 'his', 'i', 'my', 'me', 'and', 'or', 'but', 'if',
             'because', 'until', 'while', 'how', 'why', 'where', 'when',
             'not', 'no', 'nor', 'so', 'too', 'very', 'just', 'also',
             'now', 'here', 'there', 'about', 'such', 'only', 'other',
             'some', 'any', 'all', 'each', 'both', 'few', 'more', 'most',
             'than', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
             'once', 'here', 'there', 'when', 'where', 'why', 'how',
             'is', 'it', 'are', 'was', 'were', 'be', 'been', 'being'}


def load_chunks(filepath=CHUNKS_FILE):
    """加载chunks"""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    parts = raw.split("===")
    chunks = []
    for part in parts:
        part = part.strip()
        if part:
            lines = part.split("\n")
            content_lines = [l for l in lines if not l.startswith("--- Chunk")]
            content = " ".join(content_lines).strip()
            if content and len(content) > 100:
                chunks.append(content)
    return chunks


def extract_key_terms(text, top_n=5):
    """从文本中提取关键术语"""
    # 提取词
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    # 过滤停用词
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 3]
    # 词频统计
    word_freq = Counter(filtered)
    # 取最高频的词
    top_terms = [w for w, _ in word_freq.most_common(top_n)]
    return top_terms


def generate_queries_for_chunk(chunk, chunk_id, num_queries=3):
    """为单个chunk生成多个查询"""
    key_terms = extract_key_terms(chunk, top_n=5)

    queries = []

    # 1. 直接型问题
    if len(key_terms) >= 2:
        queries.append({
            "instruction": "Route the query to the correct knowledge chunk.",
            "input": f"What is the relationship between {key_terms[0]} and {key_terms[1]}?",
            "output": f"Chunk_{chunk_id:04d}"
        })

    # 2. 定义型问题
    if key_terms:
        queries.append({
            "instruction": "Route the query to the correct knowledge chunk.",
            "input": f"Explain the concept of {key_terms[0]} in information science.",
            "output": f"Chunk_{chunk_id:04d}"
        })

    # 3. 机制型问题
    if len(key_terms) >= 2:
        queries.append({
            "instruction": "Route the query to the correct knowledge chunk.",
            "input": f"How does {key_terms[0]} work in information retrieval systems?",
            "output": f"Chunk_{chunk_id:04d}"
        })

    # 4. 应用型问题
    if key_terms:
        queries.append({
            "instruction": "Route the query to the correct knowledge chunk.",
            "input": f"What are the applications of {key_terms[0]} in modern information systems?",
            "output": f"Chunk_{chunk_id:04d}"
        })

    # 5. 对比型问题
    if len(key_terms) >= 3:
        queries.append({
            "instruction": "Route the query to the correct knowledge chunk.",
            "input": f"Compare {key_terms[0]} and {key_terms[1]} in the context of information science.",
            "output": f"Chunk_{chunk_id:04d}"
        })

    return queries[:num_queries]


def generate_negative_samples(chunks, num_negatives=50):
    """生成负样本（错误路由）"""
    negatives = []

    for _ in range(num_negatives):
        # 随机选择两个不同的chunk
        correct_idx = random.randint(0, len(chunks) - 1)
        wrong_idx = (correct_idx + random.randint(1, len(chunks) - 1)) % len(chunks)

        correct_chunk = chunks[correct_idx]
        wrong_terms = extract_key_terms(chunks[wrong_idx], top_n=2)

        if wrong_terms:
            # 生成一个与其他chunk更相关的问题
            queries = [
                f"Tell me about {wrong_terms[0]}",
                f"What is {wrong_terms[0]} and how is it used?",
                f"Explain {wrong_terms[0]} in information science"
            ]
            neg_query = random.choice(queries)

            negatives.append({
                "instruction": "Route the query to the correct knowledge chunk.",
                "input": neg_query,
                "output": f"Chunk_{(correct_idx + 1) % len(chunks):04d}"  # 故意用错误的chunk
            })

    return negatives


def build_dataset():
    """构建路由训练数据集"""
    print("📂 加载chunks...")

    if not os.path.exists(CHUNKS_FILE):
        print(f"❌ 文件不存在: {CHUNKS_FILE}")
        print("请先运行: python pdf_to_chunks.py")
        return

    chunks = load_chunks()
    print(f"✅ 加载 {len(chunks)} 个chunks")

    print("🔄 生成训练样本...")

    # 生成正样本
    samples = []
    for i, chunk in enumerate(chunks):
        chunk_samples = generate_queries_for_chunk(chunk, i, num_queries=3)
        samples.extend(chunk_samples)

        if (i + 1) % 100 == 0:
            print(f"   已处理 {i+1}/{len(chunks)} chunks")

    print(f"✅ 生成 {len(samples)} 个正样本")

    # 生成负样本 (10%)
    num_negatives = len(samples) // 10
    negatives = generate_negative_samples(chunks, num_negatives)
    print(f"✅ 生成 {len(negatives)} 个负样本")

    # 合并
    all_samples = samples + negatives
    random.shuffle(all_samples)

    # 保存为JSONL
    print(f"💾 保存到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"✅ 完成! 共 {len(all_samples)} 个训练样本")
    print(f"\n📋 样本示例:")
    print(f"   Input: {samples[0]['input'][:60]}...")
    print(f"   Output: {samples[0]['output']}")


if __name__ == "__main__":
    build_dataset()
