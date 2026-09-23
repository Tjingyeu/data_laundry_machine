# build_expert_dataset.py
# Expert LoRA训练数据生成器
# 每Chunk生成5-8个query，去除负样本

import json
import os
import re
from collections import Counter
import random

CHUNKS_FILE = "chunks.txt"
OUTPUT_FILE = "train_expert.jsonl"

# 停用词
STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
             'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
             'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
             'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
             'during', 'before', 'after', 'above', 'below', 'between', 'under',
             'again', 'further', 'then', 'once', 'what', 'which', 'who', 'whom',
             'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
             'we', 'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his', 'i',
             'my', 'me', 'and', 'or', 'but', 'if', 'because', 'until', 'while',
             'how', 'why', 'where', 'when', 'not', 'no', 'nor', 'so', 'too', 'very',
             'just', 'also', 'now', 'here', 'there', 'about', 'such', 'only', 'other'}

# 领域关键词
DOMAIN_KEYWORDS = {
    'information_retrieval': ['retrieval', 'search', 'query', 'index', 'relevance', 'boolean', 'vector', 'probabilistic', 'ranking'],
    'semantic_web': ['semantic', 'ontology', 'rdf', 'web', 'linked', 'graph'],
    'digital_libraries': ['library', 'digital', 'archive', 'repository', 'preservation', 'museum'],
    'information_behavior': ['seeking', 'behavior', 'needs', 'users', 'information need', 'search process'],
    'classification': ['classification', 'categorization', 'taxonomy', 'facet', 'organization'],
    'database': ['database', 'sql', 'relational', 'dbms', 'data'],
    'metadata': ['metadata', 'schema', 'standard', 'dublin', 'marc', 'mod'],
}


def load_chunks(filepath=CHUNKS_FILE):
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


def extract_domain(text):
    text_lower = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return domain
    return "general"


def extract_key_terms(text, top_n=5):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 3]
    return [w for w, _ in Counter(filtered).most_common(top_n)]


def extract_intent(text):
    patterns = [
        r'is (?:a|an|the) ([a-zA-Z\s]+?)[\.,]',
        r'(?:means|refers to|describes|explains) ([a-zA-Z\s]+?)[\.,]',
        r'(?:process of|method of|approach to) ([a-zA-Z\s]+?)[\.,]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:300], re.IGNORECASE)
        if match:
            return match.group(1).strip()[:40]
    return None


def generate_queries_for_chunk(chunk, chunk_id, num_queries=6):
    """为chunk生成多个查询"""
    domain = extract_domain(chunk)
    key_terms = extract_key_terms(chunk, top_n=5)
    intent = extract_intent(chunk) or key_terms[0] if key_terms else "topic"

    queries = []

    # 1. 定义型
    queries.append({
        "instruction": "Route to the correct knowledge chunk.",
        "input": f"What is {key_terms[0] if key_terms else intent} in {domain}?",
        "output": f"{domain}/{key_terms[0] if key_terms else intent} -> Chunk_{chunk_id:04d}"
    })

    # 2. 关系型
    if len(key_terms) >= 2:
        queries.append({
            "instruction": "Route to the correct knowledge chunk.",
            "input": f"How do {key_terms[0]} and {key_terms[1]} relate?",
            "output": f"{domain}/{key_terms[0]} -> Chunk_{chunk_id:04d}"
        })

    # 3. 机制型
    queries.append({
        "instruction": "Route to the correct knowledge chunk.",
        "input": f"Explain how {key_terms[0] if key_terms else intent} works.",
        "output": f"{domain} -> Chunk_{chunk_id:04d}"
    })

    # 4. 应用型
    queries.append({
        "instruction": "Route to the correct knowledge chunk.",
        "input": f"What are the applications of {key_terms[0] if key_terms else intent}?",
        "output": f"{domain}/{key_terms[0] if key_terms else intent} -> Chunk_{chunk_id:04d}"
    })

    # 5. 概念对比型
    if len(key_terms) >= 3:
        queries.append({
            "instruction": "Route to the correct knowledge chunk.",
            "input": f"Compare {key_terms[0]} and {key_terms[2]} in information science.",
            "output": f"{domain}/{key_terms[0]} -> Chunk_{chunk_id:04d}"
        })

    # 6. 历史发展型
    queries.append({
        "instruction": "Route to the correct knowledge chunk.",
        "input": f"What is the history and development of {key_terms[0] if key_terms else intent}?",
        "output": f"{domain}/{key_terms[0] if key_terms else intent} -> Chunk_{chunk_id:04d}"
    })

    return queries[:num_queries]


def build_dataset():
    print("="*60)
    print("📚 Expert LoRA训练数据生成器")
    print("="*60)

    print("\n📂 加载chunks...")
    chunks = load_chunks()
    print(f"✅ 加载 {len(chunks)} 个chunks")

    print("\n🔄 生成训练数据...")
    all_samples = []

    for i, chunk in enumerate(chunks):
        samples = generate_queries_for_chunk(chunk, i, num_queries=6)
        all_samples.extend(samples)

        if (i + 1) % 100 == 0:
            print(f"   已处理 {i+1}/{len(chunks)} chunks")

    print(f"✅ 生成 {len(all_samples)} 个训练样本")

    # 打乱
    random.shuffle(all_samples)

    # 保存
    print(f"\n💾 保存到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    # 统计
    domains = [s["output"].split("/")[0] for s in all_samples]
    domain_counts = Counter(domains)
    print(f"\n📊 领域分布:")
    for domain, count in domain_counts.most_common(5):
        print(f"   {domain}: {count} ({100*count/len(all_samples):.1f}%)")

    print(f"\n✅ 完成!")


if __name__ == "__main__":
    build_dataset()
