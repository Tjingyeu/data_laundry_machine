# build_lora_dataset.py
# 构建LoRA训练数据集 - 针对信息科学书籍

import json
import os
import re
from collections import Counter

CHUNKS_FILE = "chunks.txt"
OUTPUT_FILE = "train.jsonl"

# 书籍领域关键词（用于半监督生成）
DOMAIN_KEYWORDS = [
    "information", "retrieval", "document", "metadata", "database",
    "semantic", "ontology", "indexing", "classification", "query",
    "search", "filter", "relevance", "information science",
    "knowledge", "representation", "access", "organization"
]


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
            if content and len(content) > 50:
                chunks.append(content)
    return chunks


def extract_key_concepts(text):
    """从chunk中提取关键概念（无监督方式）"""
    # 停用词
    stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
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
                 'some', 'any', 'all', 'each', 'both', 'few', 'more', 'most'}

    # 提取词
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

    # 过滤停用词，保留有意义的词
    filtered = [w for w in words if w not in stopwords]

    # 词频统计
    word_freq = Counter(filtered)

    # 取最高频的词作为概念
    top_concepts = [w for w, _ in word_freq.most_common(5)]

    return top_concepts


def generate_training_sample(chunk):
    """为单个chunk生成训练样本"""
    concepts = extract_key_concepts(chunk)

    # 方法A: 无监督 - output留空，让模型学压缩表示
    sample_unsupervised = {
        "instruction": "Compress this text into semantic search keywords for information retrieval",
        "input": chunk[:512],  # 限制长度
        "output": ""  # 留空
    }

    # 方法B: 半监督 - 用提取的概念作为output
    sample_supervised = {
        "instruction": "Extract semantic search keywords from this academic text",
        "input": chunk[:512],
        "output": ", ".join(concepts)
    }

    return sample_supervised


def build_dataset():
    """构建完整训练数据集"""
    print("📂 加载chunks...")

    if not os.path.exists(CHUNKS_FILE):
        print(f"❌ 文件不存在: {CHUNKS_FILE}")
        print("请先运行: python pdf_to_chunks.py")
        return

    chunks = load_chunks()
    print(f"✅ 加载 {len(chunks)} 个chunks")

    print("🔄 生成训练样本...")
    samples = []
    for i, chunk in enumerate(chunks):
        sample = generate_training_sample(chunk)
        samples.append(sample)

        if (i + 1) % 500 == 0:
            print(f"   已处理 {i+1}/{len(chunks)}")

    # 保存为JSONL
    print(f"💾 保存到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"✅ 完成! 生成 {len(samples)} 个训练样本")
    print(f"\n📋 样本示例:")
    print(f"   Input: {samples[0]['input'][:100]}...")
    print(f"   Output: {samples[0]['output']}")


if __name__ == "__main__":
    build_dataset()
