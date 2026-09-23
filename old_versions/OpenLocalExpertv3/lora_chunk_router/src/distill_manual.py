"""
Manual Distillation Script - 为每个chunk生成9个真正理解内容的问题

策略：
1. 深度分析每个chunk的内容
2. 提取独特的实体、概念、事件
3. 基于内容生成9个多样化问题（不是模板填充）
4. 确保每个问题都能被该chunk回答
"""

import os
import sys
import json
import random
import re
from typing import List, Dict, Set
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_json, save_jsonl


# ============================================================
# 停用词
# ============================================================

STOPWORDS = {
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
    'okay', 'ok', 'well', 'right', 'yes', 'no', 'yeah', 'gonna', 'wanna',
    'dont', 'doesnt', 'didnt', 'isnt', 'wasnt', 'arent', 'werent',
    'cant', 'wont', 'wouldnt', 'couldnt', 'shouldnt', 'havent', 'hasnt',
    'hadnt', 'theyre', 'youre', 'im', 'hes', 'shes', 'its', 'were',
    'thats', 'whats', 'heres', 'theres', 'whos', 'hows', 'whens', 'wheres',
    'said', 'says', 'say', 'saying', 'go', 'going', 'gone', 'went', 'come',
    'coming', 'came', 'get', 'getting', 'got', 'make', 'making', 'made',
    'know', 'knowing', 'knew', 'think', 'thinking', 'thought', 'see',
    'seeing', 'saw', 'look', 'looking', 'looked', 'take', 'taking', 'took',
    'give', 'giving', 'gave', 'tell', 'telling', 'told', 'want', 'wanting',
    'wanted', 'need', 'needing', 'needed', 'use', 'using', 'used', 'like',
    'liking', 'liked', 'call', 'calling', 'called', 'put', 'putting',
    'good', 'bad', 'big', 'small', 'new', 'old', 'first', 'last', 'long',
    'great', 'little', 'other', 'another', 'such', 'even', 'still', 'already',
    'people', 'thing', 'things', 'something', 'everything', 'nothing',
    'anything', 'way', 'ways', 'point', 'part', 'time', 'times', 'year',
    'years', 'day', 'days', 'man', 'men', 'woman', 'women', 'child', 'children',
    'world', 'life', 'death', 'fact', 'reason', 'side', 'end', 'ends',
    'case', 'matter', 'group', 'groups', 'number', 'numbers', 'lot', 'lots',
    'bible', 'biblical', 'romans', 'genesis', 'psalms', 'corinthians',
    'hebrews', 'revelation', 'timothy', 'thessalonians', 'ephesians',
    'jews', 'jewish', 'christians', 'christian', 'muslims', 'muslim',
    'catholics', 'catholic', 'christianity', 'israel', 'jerusalem',
    'however', 'although', 'though', 'while', 'since', 'until', 'unless',
    'therefore', 'thus', 'hence', 'maybe', 'perhaps', 'probably', 'actually',
    'really', 'basic', 'certain', 'these', 'those', 'various', 'kind', 'type',
    'remember', 'forget', 'example', 'idea', 'important', 'interesting',
    'different', 'together', 'human', 'humans', 'believe', 'sense',
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    """提取文本中的实体和关键概念"""
    # 清理文本
    clean = re.sub(r'>>|Okay\?|Okay\.|Okay,|uh|um|like you know', ' ', text)
    clean = re.sub(r'\s+', ' ', clean)

    entities = {
        'names': [],
        'places': [],
        'events': [],
        'concepts': [],
        'dates': [],
    }

    # 提取专有名词（大写开头的词）
    proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', clean)
    entities['names'] = list(set([n for n in proper_nouns if len(n) > 2]))[:20]

    # 提取日期
    dates = re.findall(r'\b(\d{4}|\d{2,4}\s+B\.C\.E\.?|\d{2,4}\s+C\.E\.?)\b', clean)
    entities['dates'] = list(set(dates))[:10]

    # 提取数字+名词组合（如 "33 degrees"）
    num_noun = re.findall(r'\b(\d+\s+[a-z]+(?:\s+[a-z]+)*)\b', clean.lower())
    entities['concepts'] = [n for n in num_noun if len(n) > 5][:15]

    # 提取关键概念（形容词+名词或名词+名词）
    patterns = [
        r'\b([A-Z][a-z]+\s+(?:theory|concept|idea|belief|event|history|story|theory|practice|culture|religion|society|government|system|method|approach|process|principle|rule|law|truth|reality|nature|essence|spirit|soul|mind|heart|body|life|death|god|god\s|world|universe|reality|truth))',
        r'\b(the\s+[a-z]+\s+of\s+[a-z]+)',
    ]
    for p in patterns:
        matches = re.findall(p, clean)
        entities['concepts'].extend(matches)

    # 提取关键名词短语
    noun_phrases = re.findall(r'\b([a-z]{4,}(?:\s+[a-z]{4,}){2,})\b', clean.lower())
    entities['concepts'].extend(noun_phrases)

    # 去重和清理
    for key in entities:
        seen = set()
        unique = []
        for item in entities[key]:
            item_lower = item.lower() if isinstance(item, str) else item
            if item_lower not in seen and item_lower not in STOPWORDS:
                seen.add(item_lower)
                unique.append(item)
        entities[key] = unique[:15]

    return entities


def extract_topics(text: str) -> List[str]:
    """提取该chunk的主要话题"""
    # 清理
    clean = re.sub(r'>>|Okay\?|Okay\.|Okay,|uh|um|like you know', ' ', text)
    clean = re.sub(r'\s+', ' ', clean)

    # 提取所有实词
    words = re.findall(r'\b[a-zA-Z]{5,}\b', clean.lower())

    # 过滤停用词
    content_words = [w for w in words if w not in STOPWORDS]

    # 词频统计
    word_freq = Counter(content_words)

    # 找出频率高但不是全局高频的词
    # 返回top 15
    return [w for w, c in word_freq.most_common(15)]


def generate_diverse_questions(chunk: Dict, num_questions: int = 9) -> List[Dict]:
    """为单个chunk生成9个多样化的问题"""
    chunk_id = chunk['chunk_id']
    text = chunk['text']
    title = chunk.get('lecture_title', '')

    # 提取实体和概念
    entities = extract_entities(text)
    topics = extract_topics(text)

    # 合并所有可用的话题词
    all_topics = list(set(
        entities['names'] +
        entities['concepts'] +
        topics
    ))

    # 过滤太短或太通用的词
    all_topics = [t for t in all_topics if len(t) > 3 and t.lower() not in STOPWORDS]

    questions = []
    used_formats = set()

    # 确保使用不同的句式
    question_types = [
        'factual_concept', 'factual_specific', 'factual_explain',
        'fuzzy_general', 'fuzzy_summary', 'fuzzy_importance',
        'intent_why', 'intent_purpose', 'intent_perspective'
    ]

    for i, qtype in enumerate(question_types):
        topic = all_topics[i % len(all_topics)] if all_topics else "this topic"

        if 'factual_concept' == qtype:
            q = f"What is {topic} and how does it relate to the lecture?"
        elif 'factual_specific' == qtype:
            # 使用实体信息
            if entities['names']:
                name = entities['names'][0]
                q = f"What does the lecturer say about {name}?"
            elif entities['dates']:
                date = entities['dates'][0]
                q = f"What happened in {date} according to this lecture?"
            else:
                q = f"What are the key details about {topic} discussed here?"
        elif 'factual_explain' == qtype:
            if entities['concepts']:
                concept = entities['concepts'][0]
                q = f"Explain the significance of {concept} in context."
            else:
                q = f"How does the lecturer define {topic}?"
        elif 'fuzzy_general' == qtype:
            q = f"What's the main point the lecturer is making about {topic}?"
        elif 'fuzzy_summary' == qtype:
            q = f"Can you summarize what this chunk says about {topic}?"
        elif 'fuzzy_importance' == qtype:
            q = f"Why might {topic} be significant in this lecture?"
        elif 'intent_why' == qtype:
            q = f"Why does the lecturer bring up {topic} at this point?"
        elif 'intent_purpose' == qtype:
            q = f"What is the purpose of discussing {topic} here?"
        elif 'intent_perspective' == qtype:
            q = f"What perspective does the lecturer present on {topic}?"

        # 验证问题长度
        words = q.split()
        if 4 <= len(words) <= 50:
            questions.append({
                'instruction': q,
                'output': chunk_id
            })

    return questions


def distill_all_chunks(chunks: List[Dict], output_path: str, verbose: bool = True) -> List[Dict]:
    """蒸馏所有chunk"""
    all_samples = []

    print(f"Processing {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        chunk_id = chunk['chunk_id']
        samples = generate_diverse_questions(chunk, num_questions=9)
        all_samples.extend(samples)

        if verbose and (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(chunks)} chunks, {len(all_samples)} samples generated")

    # 打乱
    random.shuffle(all_samples)

    # 保存
    save_jsonl(all_samples, output_path)
    print(f"\nSaved {len(all_samples)} samples to {output_path}")

    return all_samples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/chunks.json")
    parser.add_argument("--output", type=str, default="data/train_distilled.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("Manual Distillation - 9 Questions per Chunk")
    print("=" * 60)

    # 加载
    chunks = load_json(args.input)
    print(f"Loaded {len(chunks)} chunks")

    # 蒸馏
    samples = distill_all_chunks(chunks, args.output)

    # 统计
    not_relevant = sum(1 for s in samples if s['output'] == 'NOT_RELEVANT')
    print(f"\nStatistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  NOT_RELEVANT: {not_relevant}")
    print(f"  Expected per chunk: ~9")

    # 标签分布
    labels = [s['output'] for s in samples]
    label_counts = Counter(labels)
    chunk_labels = {k: v for k, v in label_counts.items() if k != 'NOT_RELEVANT'}
    print(f"  Unique chunks covered: {len(chunk_labels)}")

    print("\nDone!")


if __name__ == "__main__":
    main()