"""
Distill Data Generator v3 - 智能关键词提取版

核心改进：
1. 真正的每个chunk提取独特关键词
2. 丰富的模板池（每类型20+模板）
3. 避免通用占位符
4. 真正的多样性而非简单替换
"""

import os
import sys
import json
import random
import re
from typing import List, Dict, Any, Tuple
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_json, save_jsonl


# ============================================================
# 停用词
# ============================================================

STOPWORDS = {
    # 基础停用词
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'and', 'or', 'but', 'if', 'then', 'because', 'so', 'that', 'this',
    'these', 'those', 'it', 'its', 'what', 'who', 'how', 'why', 'when',
    'where', 'which', 'their', 'there', 'here', 'all', 'some', 'any',
    'about', 'after', 'before', 'between', 'during', 'without', 'under',
    'over', 'above', 'below', 'against', 'each', 'every', 'only', 'own',
    'same', 'than', 'too', 'very', 'just', 'now', 'also', 'back', 'then',
    
    # 口语/填充
    'okay', 'ok', 'well', 'right', 'yes', 'no', 'yeah', 'gonna', 'wanna',
    'dont', 'doesnt', 'didnt', 'isnt', 'wasnt', 'arent', 'werent',
    'cant', 'wont', 'wouldnt', 'couldnt', 'shouldnt', 'havent', 'hasnt',
    'hadnt', 'theyre', 'youre', 'im', 'hes', 'shes', 'its', 'were',
    'thats', 'whats', 'heres', 'theres', 'whos', 'hows', 'whens', 'wheres',
    
    # 动词
    'said', 'says', 'say', 'saying', 'go', 'going', 'gone', 'went', 'come',
    'coming', 'came', 'get', 'getting', 'got', 'make', 'making', 'made',
    'know', 'knowing', 'knew', 'think', 'thinking', 'thought', 'see',
    'seeing', 'saw', 'look', 'looking', 'looked', 'take', 'taking', 'took',
    'give', 'giving', 'gave', 'tell', 'telling', 'told', 'want', 'wanting',
    'wanted', 'need', 'needing', 'needed', 'use', 'using', 'used', 'like',
    'liking', 'liked', 'call', 'calling', 'called', 'put', 'putting',
    
    # 形容词/副词
    'good', 'bad', 'big', 'small', 'new', 'old', 'first', 'last', 'long',
    'great', 'little', 'own', 'other', 'another', 'same', 'such', 'no',
    'not', 'only', 'very', 'just', 'even', 'still', 'already', 'yet',
    
    # 常见无意义词
    'people', 'thing', 'things', 'something', 'everything', 'nothing',
    'anything', 'way', 'ways', 'point', 'part', 'time', 'times', 'year',
    'years', 'day', 'days', 'man', 'men', 'woman', 'women', 'child', 'children',
    'world', 'life', 'death', 'fact', 'fact', 'reason', 'side', 'end', 'ends',
    'case', 'matter', 'group', 'groups', 'number', 'numbers', 'lot', 'lots',
    
    # 圣经/宗教常用词（高频但无区分度）
    'bible', 'biblical', 'romans', 'genesis', 'psalms', 'corinthians',
    'hebrews', 'revelation', 'timothy', 'thessalonians', 'ephesians',
    'jews', 'jewish', 'christians', 'christian', 'muslims', 'muslim',
    'catholics', 'catholic', 'christianity', 'israel', 'jerusalem',
    
    # 连接词/无意义词
    'however', 'although', 'though', 'while', 'since', 'until', 'unless',
    'therefore', 'thus', 'hence', 'maybe', 'perhaps', 'probably', 'actually',
    'really', '基本', '一定', '这些', '那些', '一些', '各种', '各种各',
}

# 高频无意义词（全局高频）
GLOBAL_HIGH_FREQ = {
    'world', 'people', 'thing', 'things', 'time', 'year', 'years',
    'way', 'right', 'okay', 'first', 'last', 'long', 'great', 'good',
    'different', 'something', 'everything', 'nothing', 'important',
    'history', 'human', 'believe', 'sense', 'together', 'number',
    'actually', 'really', 'thing', 'things', 'going', 'want', 'make',
}


def is_good_keyword(word: str, text: str = None, global_freq: Counter = None) -> bool:
    """判断是否为有效关键词"""
    word_lower = word.lower()
    
    # 长度检查
    if len(word_lower) < 4:
        return False
    
    # 停用词检查
    if word_lower in STOPWORDS:
        return False
    
    # 全局高频检查（降低权重但不一定完全排除）
    if word_lower in GLOBAL_HIGH_FREQ:
        return False
    
    # 纯数字检查
    if word_lower.isdigit():
        return False
    
    # 检查是否在text中作为关键词出现（而非只是出现）
    if text and global_freq:
        text_count = text.lower().count(word_lower)
        global_count = global_freq.get(word_lower, 1)
        # 如果在文本中出现次数高于全局平均，认为是有效关键词
        return text_count >= 2
    
    return True


# ============================================================
# 关键词提取
# ============================================================

def extract_distinctive_keywords(text: str, global_freq: Counter, top_n: int = 10) -> List[str]:
    """提取该chunk特有的、有区分度的关键词"""
    # 清理
    clean = re.sub(r'>>|Okay\?|Okay\.|Okay,|uh|um|um\.|like you know', ' ', text)
    clean = re.sub(r'\s+', ' ', clean)
    
    # 提取所有候选词
    words = re.findall(r'\b[a-zA-Z]{5,}\b', clean.lower())
    
    # 计算每个词在本文中的出现频率
    word_counts = Counter(words)
    
    # 评分：tfidf风格 - 词频 * 逆全局频率
    scored = {}
    for word, count in word_counts.items():
        if not is_good_keyword(word, text, global_freq):
            continue
        # tfidf评分
        idf = 1.0 / (global_freq.get(word, 1) + 1)
        score = count * idf
        scored[word] = score
    
    # 排序返回top_n
    sorted_words = sorted(scored.items(), key=lambda x: -x[1])
    return [w for w, s in sorted_words[:top_n]]


# ============================================================
# 模板池（大幅扩充）
# ============================================================

FACTUAL_TEMPLATES = [
    # 标准what/why/how
    "What is {topic}?",
    "What are the key aspects of {topic}?",
    "What makes {topic} significant?",
    "What is the significance of {topic}?",
    "What happened with {topic}?",
    "What do we know about {topic}?",
    "What is the relationship between {topic} and {topic2}?",
    "What evidence exists for {topic}?",
    "What are the main characteristics of {topic}?",
    "What caused {topic}?",
    "What resulted from {topic}?",
    "How did {topic} occur?",
    "How does {topic} work?",
    "How is {topic} relevant?",
    "How is {topic} connected to history?",
    "How has {topic} evolved?",
    "Why is {topic} important?",
    "Why did {topic} happen?",
    "Why does {topic} matter?",
    "Explain {topic}.",
    "Describe {topic}.",
    "Tell me about {topic}.",
    "What should I know about {topic}?",
    "Can you explain {topic}?",
    "What's the deal with {topic}?",
    # 更多变体
    "What role does {topic} play?",
    "What is {topic} trying to convey?",
    "What is the main idea behind {topic}?",
    "How would you describe {topic}?",
    "What is unique about {topic}?",
]

FUZZY_TEMPLATES = [
    # 口语化问题
    "What's all this about {topic}?",
    "I've heard of {topic}, what's the big idea?",
    "Can someone explain {topic} to me?",
    "What's the fuss regarding {topic}?",
    "What do people mean by {topic}?",
    "Why is {topic} getting so much attention?",
    "What's the story with {topic}?",
    "How would you summarize {topic} in simple terms?",
    "What does {topic} actually mean?",
    "What's the bottom line on {topic}?",
    "Is {topic} really that important?",
    "Why should I care about {topic}?",
    "What exactly is {topic} about?",
    "Can anyone break down {topic} for me?",
    "What's the quick version of {topic}?",
    "How do you explain {topic} to a beginner?",
    "What's a simple way to understand {topic}?",
    "Why does everyone keep mentioning {topic}?",
    "What do I need to know about {topic}?",
    "How does {topic} fit into everything?",
    # 更多模糊表达
    "I'm confused about {topic}, help?",
    "What exactly is going on with {topic}?",
    "Can you dumb down {topic} for me?",
    "What the heck is {topic}?",
    "How does {topic} actually work in practice?",
]

INTENT_TEMPLATES = [
    # 意图理解
    "What point is the lecturer making about {topic}?",
    "What is the lecturer trying to say about {topic}?",
    "Why does the lecturer emphasize {topic}?",
    "What is the lecturer's opinion on {topic}?",
    "What perspective does the lecturer present about {topic}?",
    "Why is the lecturer talking about {topic}?",
    "What is the lecturer getting at with {topic}?",
    "What can we learn from how the lecturer discusses {topic}?",
    "Why bring up {topic} at this point?",
    "What motivates the lecturer to discuss {topic}?",
    "What argument is being made through {topic}?",
    "What's the deeper meaning behind {topic}?",
    "Why focus on {topic} specifically?",
    "What does the lecturer want us to understand about {topic}?",
    "How does the lecturer want us to interpret {topic}?",
    "What reasoning does the lecturer offer about {topic}?",
    "What insight is the lecturer sharing about {topic}?",
    "Why is {topic} central to this discussion?",
    "What conclusion does the lecturer draw about {topic}?",
    "How does {topic} support the lecture's main thesis?",
]

HARD_NEGATIVE_TEMPLATES = [
    # 困难负样本：相似但无关
    "How did {topic} develop in ancient {place}?",
    "What is the history of {topic} in {place}?",
    "Compare {topic} with {topic2} in different contexts.",
    "What do scholars say about {topic} in {place}?",
    "How has {topic} been studied academically?",
    "What are the main academic theories about {topic}?",
    "Describe the cultural significance of {topic}.",
    "What role did {topic} play in earlier civilizations?",
    "How is {topic} taught in modern universities?",
    "What is the current research on {topic}?",
    "Can you compare {topic} and {topic2}?",
    "What are the key differences between {topic} and {topic2}?",
    "How has the understanding of {topic} changed over time?",
    "What evidence supports theories about {topic}?",
    "What controversies surround {topic}?",
    "Who are the major figures associated with {topic}?",
    "What organizations are related to {topic}?",
    "How is {topic} relevant to modern science?",
    "What are common misconceptions about {topic}?",
    "How is {topic} portrayed in popular media?",
]


PLACES = ['China', 'Rome', 'Greece', 'Egypt', 'India', 'Europe', 'America', 'the Middle East', 'Asia', 'Africa']


def build_keyword_pool(chunks: List[Dict]) -> Counter:
    """构建全局关键词频率表"""
    all_words = []
    for chunk in chunks:
        text = chunk['text']
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        all_words.extend(words)
    return Counter(all_words)


# ============================================================
# 生成函数
# ============================================================

def generate_questions(text: str, keywords: List[str], type_: str) -> str:
    """根据类型和关键词生成问题"""
    topic = keywords[0] if keywords else None
    topic2 = keywords[1] if len(keywords) > 1 else None
    
    if not topic:
        return None
    
    templates = {
        'factual': FACTUAL_TEMPLATES,
        'fuzzy': FUZZY_TEMPLATES,
        'intent': INTENT_TEMPLATES,
        'hard_negative': HARD_NEGATIVE_TEMPLATES,
    }
    
    template_pool = templates.get(type_, FACTUAL_TEMPLATES)
    template = random.choice(template_pool)
    
    result = template.format(
        topic=topic,
        topic2=topic2 or 'similar topics',
        place=random.choice(PLACES)
    )
    
    # 清理多余的空格
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def compute_lcs_ratio(query: str, text: str) -> float:
    """计算与原文的重合度"""
    query_words = set(w.lower() for w in query.split())
    text_words = set(w.lower() for w in text.split())
    
    query_content = query_words - STOPWORDS
    text_content = text_words - STOPWORDS
    
    if not query_content:
        return 0.0
    
    intersection = query_content & text_content
    return len(intersection) / len(query_content)


def is_valid_sample(query: str, text: str, threshold: float = 0.55) -> bool:
    """检查生成的问题是否有效"""
    if not query or len(query.split()) < 4:
        return False
    if compute_lcs_ratio(query, text) > threshold:
        return False
    return True


# ============================================================
# 主逻辑
# ============================================================

def distill_chunk(chunk: Dict, global_freq: Counter, max_attempts: int = 15) -> List[Dict]:
    """为单个chunk生成多样化问题"""
    chunk_id = chunk['chunk_id']
    text = chunk['text']
    
    # 提取关键词
    keywords = extract_distinctive_keywords(text, global_freq, top_n=15)
    
    results = []
    types = ['factual'] * 3 + ['fuzzy'] * 3 + ['intent'] * 3 + ['hard_negative'] * 1
    
    for type_ in types:
        for _ in range(max_attempts):
            query = generate_questions(text, keywords, type_)
            if query and is_valid_sample(query, text):
                break
        
        if query:
            label = 'NOT_RELEVANT' if type_ == 'hard_negative' else chunk_id
            results.append({'instruction': query, 'output': label})
    
    return results


def distill_dataset(chunks: List[Dict], verbose: bool = True) -> List[Dict]:
    """蒸馏整个数据集"""
    # 先建全局词频
    print("Building global keyword frequency table...")
    global_freq = build_keyword_pool(chunks)
    
    all_results = []
    for i, chunk in enumerate(chunks):
        results = distill_chunk(chunk, global_freq)
        all_results.extend(results)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks, {len(all_results)} samples...")
    
    return all_results


def compute_stats(samples: List[Dict]) -> Dict:
    """计算统计"""
    instructions = [s['instruction'] for s in samples]
    
    all_words = []
    for inst in instructions:
        all_words.extend(inst.lower().split())
    
    word_counts = Counter(all_words)
    content_words = [w for w in all_words if w not in STOPWORDS]
    
    lengths = [len(inst.split()) for inst in instructions]
    
    return {
        'total': len(samples),
        'unique_instructions': len(set(instructions)),
        'unique_words': len(set(content_words)),
        'word_diversity': len(set(content_words)) / len(content_words) if content_words else 0,
        'avg_length': sum(lengths) / len(lengths) if lengths else 0,
        'not_relevant': sum(1 for s in samples if s['output'] == 'NOT_RELEVANT'),
        'top_words': word_counts.most_common(10),
    }


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/chunks.json")
    parser.add_argument("--output", type=str, default="data/train_distilled.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 60)
    print("Data Distillation Pipeline v3")
    print("=" * 60)
    
    print(f"\n📂 Loading: {args.input}")
    chunks = load_json(args.input)
    print(f"   {len(chunks)} chunks loaded")
    
    print("\n🔬 Distilling...")
    samples = distill_dataset(chunks)
    print(f"   Generated {len(samples)} samples")
    
    # 过滤
    print("\n🧹 Filtering...")
    filtered = [s for s in samples if 4 <= len(s['instruction'].split()) <= 50]
    print(f"   Kept {len(filtered)} after length filter")
    
    random.shuffle(filtered)
    
    print(f"\n💾 Saving: {args.output}")
    save_jsonl(filtered, args.output)
    
    # 统计
    print("\n📊 Statistics:")
    stats = compute_stats(filtered)
    print(f"   Total: {stats['total']}")
    print(f"   Unique instructions: {stats['unique_instructions']}")
    print(f"   Unique content words: {stats['unique_words']}")
    print(f"   Word diversity: {stats['word_diversity']:.2%}")
    print(f"   Avg length: {stats['avg_length']:.1f} words")
    print(f"   NOT_RELEVANT: {stats['not_relevant']}")
    
    print("\n   Top 10 words:")
    for word, count in stats['top_words'][:10]:
        print(f"     {word}: {count}")
    
    # 检查重复
    print("\n🔍 Duplicate check:")
    instructions = [s['instruction'] for s in filtered]
    counter = Counter(instructions)
    duplicates = [(i, c) for i, c in counter.items() if c > 1]
    print(f"   Duplicate instructions: {len(duplicates)}")
    if duplicates[:5]:
        print("   Top duplicates:")
        for inst, count in duplicates[:5]:
            print(f"     [{count}x] {inst[:60]}...")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
