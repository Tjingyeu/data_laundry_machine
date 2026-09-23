# build_distilled_dataset.py
# 优化版训练数据生成器
# 策略：语义蒸馏 + 聚类去重 + 锚点提取 + 变长采样

import json
import os
import re
from collections import Counter, defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
import random

CHUNKS_FILE = "chunks.txt"
OUTPUT_FILE = "train_distilled.jsonl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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
             'just', 'also', 'now', 'here', 'there', 'about', 'such', 'only', 'other',
             'some', 'any', 'all', 'each', 'both', 'few', 'more', 'most', 'than',
             'up', 'down', 'out', 'off', 'over', 'under', 'again', 'once', 'here',
             'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
             'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
             'just', 'should', 'now', 'into', 'from', 'with', 'without', 'within'}


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


def extract_entities_and_intent(text):
    """
    语义蒸馏：从文本中提取核心实体和意图
    返回: {"entities": [...], "intent": "...", "domain": "..."}
    """
    # 提取词
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in STOPWORDS]

    # 词频统计
    word_freq = Counter(filtered)
    top_words = [w for w, _ in word_freq.most_common(10)]

    # 识别实体（名词性的高频词）
    entities = top_words[:5]

    # 识别意图（从常见模式中提取）
    intent_patterns = [
        r'is (?:a|an|the) (.+?)[\.,]',
        r'(?:refers to|means|describes|explains) (.+?)[\.,]',
        r'(?:process of|method of|approach to) (.+?)[\.,]',
    ]

    intent = None
    for pattern in intent_patterns:
        match = re.search(pattern, text[:200], re.IGNORECASE)
        if match:
            intent = match.group(1).strip()[:50]
            break

    if not intent:
        # 用最重要的词作为意图
        intent = entities[0] if entities else "general topic"

    # 识别领域（基于关键词）
    domain_keywords = {
        'information_retrieval': ['retrieval', 'search', 'query', 'index', 'relevance', 'boolean', 'vector'],
        'semantic_web': ['semantic', 'ontology', 'rdf', 'web', 'linked data'],
        'digital_libraries': ['library', 'digital', 'archive', 'repository', 'preservation'],
        'information_behavior': ['seeking', 'behavior', 'needs', 'users', 'information need'],
        'classification': ['classification', 'categorization', 'taxonomy', 'facet'],
        'database': ['database', 'sql', 'query', 'relational', 'dbms'],
    }

    text_lower = text.lower()
    detected_domain = "general"
    for domain, keywords in domain_keywords.items():
        if any(kw in text_lower for kw in keywords):
            detected_domain = domain
            break

    return {
        "entities": entities,
        "intent": intent,
        "domain": detected_domain,
        "key_terms": top_words[:3]  # 最关键的3个词
    }


def compute_chunk_embeddings(chunks, model_name=EMBEDDING_MODEL):
    """计算所有chunk的向量表示"""
    print("🔄 计算chunk嵌入...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)
    return embeddings


def cluster_chunks(chunks, embeddings, threshold=0.85):
    """
    聚类相似的chunk
    threshold越高=越严格的相似度要求
    """
    print("🔄 聚类chunks...")

    n = len(chunks)
    clusters = []
    assigned = [False] * n

    for i in range(n):
        if assigned[i]:
            continue

        # 从这个chunk开始建立新簇
        cluster = [i]
        assigned[i] = True

        for j in range(i + 1, n):
            if assigned[j]:
                continue

            # 计算余弦相似度
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )

            if sim > threshold:
                cluster.append(j)
                assigned[j] = True

        clusters.append(cluster)

    print(f"✅ 生成 {len(clusters)} 个簇")

    # 统计
    cluster_sizes = [len(c) for c in clusters]
    print(f"   平均簇大小: {np.mean(cluster_sizes):.1f}")
    print(f"   最大簇: {max(cluster_sizes)}, 最小簇: {min(cluster_sizes)}")

    return clusters


def is_anchor_chunk(chunk_text):
    """
    判断是否为锚点chunk（包含关键信息）
    锚点特征：定义、专有名词、数值、转折句
    """
    # 定义模式
    definition_patterns = [
        r'is (?:defined as|defined|referred to as|called)',
        r'(?:means|refers to|describes|explains)',
        r'(?:first|second|third|finally|conclusion)',
        r'\d+[-–]\d+',  # 数值范围
        r'\d+\s*(?:percent|%|years?|pages?|items?)',  # 具体数值
    ]

    text_lower = chunk_text.lower()

    # 统计锚点信号
    anchor_score = 0

    for pattern in definition_patterns:
        if re.search(pattern, text_lower):
            anchor_score += 1

    # 如果包含多个专有名词（大写词）
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', chunk_text)
    if len(proper_nouns) > 3:
        anchor_score += 1

    return anchor_score >= 1


def generate_distilled_queries(chunk, chunk_id, distilled_info, num_queries=2):
    """
    基于蒸馏信息生成高质量查询
    """
    entities = distilled_info["entities"]
    intent = distilled_info["intent"]
    domain = distilled_info["domain"]
    key_terms = distilled_info["key_terms"]

    queries = []

    # 1. 定义型查询
    queries.append({
        "instruction": "Route to the correct knowledge chunk.",
        "input": f"What is {key_terms[0] if key_terms else intent} in {domain}?",
        "output": f"{domain}/{intent} → Chunk_{chunk_id:04d}"
    })

    # 2. 实体关系查询
    if len(entities) >= 2:
        queries.append({
            "instruction": "Route to the correct knowledge chunk.",
            "input": f"How do {entities[0]} and {entities[1]} relate to each other?",
            "output": f"{domain}/{entities[0]} → Chunk_{chunk_id:04d}"
        })

    # 3. 意图验证查询
    queries.append({
        "instruction": "Route to the correct knowledge chunk.",
        "input": f"Explain the concept of {intent} in information science.",
        "output": f"{domain} → Chunk_{chunk_id:04d}"
    })

    return queries[:num_queries]


def build_distilled_dataset():
    """构建蒸馏后的训练数据集"""
    print("="*60)
    print("📚 优化版训练数据生成器")
    print("策略: 语义蒸馏 + 聚类去重 + 锚点提取")
    print("="*60)

    # 1. 加载chunks
    print("\n📂 Step 1: 加载chunks...")
    chunks = load_chunks()
    print(f"✅ 加载 {len(chunks)} 个chunks")

    # 2. 语义蒸馏
    print("\n📂 Step 2: 语义蒸馏...")
    distilled_info = []
    for i, chunk in enumerate(chunks):
        info = extract_entities_and_intent(chunk)
        distilled_info.append(info)

        if (i + 1) % 100 == 0:
            print(f"   已处理 {i+1}/{len(chunks)}")

    print(f"✅ 蒸馏完成")

    # 显示示例
    print(f"\n📋 示例 (Chunk_0000):")
    print(f"   Intent: {distilled_info[0]['intent']}")
    print(f"   Entities: {distilled_info[0]['entities']}")
    print(f"   Domain: {distilled_info[0]['domain']}")

    # 3. 聚类分析
    print("\n📂 Step 3: 聚类分析...")
    embeddings = compute_chunk_embeddings(chunks)
    clusters = cluster_chunks(chunks, embeddings, threshold=0.85)

    # 4. 生成训练数据（锚点优先，变长采样）
    print("\n📂 Step 4: 生成训练数据...")

    # 采样策略：锚点chunk优先，非锚点稀疏采样
    SAMPLING_RATE_ANCHOR = 1.0  # 锚点100%采样
    SAMPLING_RATE_NORMAL = 0.3   # 普通30%采样
    SAMPLING_RATE_SIMILAR = 0.1 # 相邻相似10%采样

    samples = []
    sampled_chunks = set()

    # 为每个簇选择一个代表
    for cluster in clusters:
        # 找锚点chunk
        anchor_idx = None
        for idx in cluster:
            if is_anchor_chunk(chunks[idx]):
                anchor_idx = idx
                break

        # 如果没找到锚点，选择簇中间那个
        if anchor_idx is None:
            anchor_idx = cluster[len(cluster) // 2]

        # 为代表chunk生成数据
        if anchor_idx not in sampled_chunks:
            info = distilled_info[anchor_idx]
            chunk_queries = generate_distilled_queries(
                chunks[anchor_idx], anchor_idx, info, num_queries=2
            )
            samples.extend(chunk_queries)
            sampled_chunks.add(anchor_idx)

    # 额外采样一些非锚点chunk
    for i, (chunk, info) in enumerate(zip(chunks, distilled_info)):
        if i in sampled_chunks:
            continue

        # 检查是否在高度相似的簇中
        in_similar_cluster = False
        for cluster in clusters:
            if len(cluster) > 5 and i in cluster[1:]:  # 簇中非第一个位置
                in_similar_cluster = True
                break

        if in_similar_cluster and random.random() > SAMPLING_RATE_SIMILAR:
            continue

        if not is_anchor_chunk(chunk) and random.random() > SAMPLING_RATE_NORMAL:
            continue

        # 生成查询
        chunk_queries = generate_distilled_queries(chunk, i, info, num_queries=1)
        samples.extend(chunk_queries)
        sampled_chunks.add(i)

    print(f"✅ 生成 {len(samples)} 个训练样本")

    # 5. 添加负样本
    print("\n📂 Step 5: 添加负样本...")
    num_negatives = len(samples) // 10
    for _ in range(num_negatives):
        # 随机选择两个不同chunk
        idx1 = random.randint(0, len(chunks) - 1)
        idx2 = (idx1 + random.randint(1, len(chunks) - 1)) % len(chunks)

        info1 = distilled_info[idx1]
        info2 = distilled_info[idx2]

        # 生成误导查询
        neg_queries = [
            f"Tell me about {info2['entities'][0] if info2['entities'] else info2['intent']}",
            f"What is {info2['intent']}?",
        ]

        samples.append({
            "instruction": "Route to the correct knowledge chunk.",
            "input": random.choice(neg_queries),
            "output": f"NOT {info1['domain']} → Chunk_{idx1:04d}"  # 故意路由错误
        })

    print(f"✅ 添加 {num_negatives} 个负样本")

    # 6. 保存
    print("\n📂 Step 6: 保存...")
    random.shuffle(samples)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"✅ 保存到 {OUTPUT_FILE}")
    print(f"   总样本: {len(samples)}")

    # 统计
    domains = [s["output"].split("/")[0] for s in samples if "/" in s["output"]]
    domain_counts = Counter(domains)
    print(f"\n📊 领域分布:")
    for domain, count in domain_counts.most_common():
        print(f"   {domain}: {count}")

    print(f"\n✅ 完成!")


if __name__ == "__main__":
    build_distilled_dataset()
