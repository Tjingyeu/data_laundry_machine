"""
Manual Distillation Final - 基于lecture_title生成问题
直接从lecture_title提取话题，确保问题质量
"""

import os
import sys
import json
import random
import re
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_json, save_jsonl


def extract_topic_from_title(title: str) -> List[str]:
    """从lecture title提取高质量话题"""
    if not title:
        return ["the main topic"]

    # 清理标题
    clean = re.sub(r'#\d+', '', title)  # 移除章节号
    clean = re.sub(r'[^\w\s]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # 分割成词
    words = clean.split()

    # 极严格停用词
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
        'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'what', 'when', 'where', 'who', 'whom', 'whose', 'why',
        'okay', 'ok', 'well', 'right', 'yes', 'no', 'yeah', 'gonna', 'wanna',
        'dont', 'doesnt', 'didnt', 'isnt', 'wasnt', 'arent', 'werent',
        'cant', 'wont', 'wouldnt', 'couldnt', 'shouldnt', 'havent', 'hasnt',
        'hadnt', 'theyre', 'youre', 'im', 'hes', 'shes', 'its', 'were',
        'thats', 'whats', 'heres', 'theres', 'whos', 'hows', 'whens', 'wheres',
        'secret', 'history', 'lecture', 'class', 'part', 'chapter',
        'different', 'another', 'everything', 'something', 'nothing', 'anything',
        'everyone', 'someone', 'anyone', 'nobody', 'person', 'people',
        'way', 'ways', 'thing', 'things', 'point', 'points', 'time', 'times',
        'year', 'years', 'day', 'days', 'man', 'men', 'woman', 'women',
        'world', 'life', 'death', 'fact', 'facts', 'reason', 'reasons',
        'case', 'matter', 'group', 'groups', 'number', 'numbers', 'lot', 'lots',
        'kind', 'type', 'sort', 'part', 'parts', 'stuff',
        'true', 'false', 'real', 'actual', 'possible', 'impossible',
        'good', 'bad', 'big', 'small', 'new', 'old', 'first', 'last', 'long',
        'great', 'little', 'even', 'still', 'already', 'more', 'most', 'less', 'least',
        'important', 'interesting', 'significant', 'relevant', 'useful',
        'seems', 'seemed', 'like', 'really', 'actually', 'basically', 'simply',
        'going', 'coming', 'taking', 'giving', 'making', 'doing', 'getting',
        'having', 'being', 'feeling', 'think', 'thinking', 'wanting', 'needing',
        'trying', 'looking', 'watching', 'seeing', 'hearing', 'saying', 'telling',
        'know', 'knowing', 'knew', 'see', 'seeing', 'saw',
    }

    # 只保留有意义的词（5字符以上，非停用词）
    meaningful = [w for w in words if len(w) >= 4 and w.lower() not in STOPWORDS]

    if not meaningful:
        return ["the main topic"]

    return meaningful[:5]


def extract_topic_from_text(text: str, title_topics: List[str]) -> List[str]:
    """从text中提取与title相关的话题"""
    # 清理
    clean = re.sub(r'>>', ' ', text)
    clean = re.sub(r'\b(Okay|Ok|uh|um|like you know|yeah|right|okay|but|and|the|was|were|has|had)\b', ' ', clean, flags=re.IGNORECASE)
    clean = re.sub(r'[^\w\s]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean)

    words = clean.split()
    word_freq = {}

    # 极严格停用词
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
        'going', 'coming', 'taking', 'giving', 'making', 'doing', 'getting',
        'having', 'being', 'seeming', 'feeling', 'think', 'thinking',
        'wanting', 'needing', 'trying', 'looking', 'watching', 'seeing',
        'hearing', 'saying', 'telling', 'ask', 'asks', 'asked',
        'know', 'knowing', 'knew', 'see', 'seeing', 'saw',
        'people', 'thing', 'things', 'something', 'everything', 'nothing',
        'anything', 'way', 'ways', 'point', 'points', 'part', 'parts',
        'time', 'times', 'year', 'years', 'day', 'days',
        'man', 'men', 'woman', 'women', 'child', 'children', 'person', 'persons',
        'world', 'life', 'death', 'fact', 'facts', 'reason', 'reasons',
        'side', 'end', 'ends', 'case', 'matter', 'group', 'groups',
        'number', 'numbers', 'lot', 'lots', 'kind', 'kinds', 'type', 'types',
        'important', 'interesting', 'significant', 'relevant', 'useful',
        'true', 'false', 'real', 'actual', 'possible', 'impossible',
        'seems', 'seemed', 'like', 'just', 'really', 'actually', 'basically',
        'different', 'another', 'other', 'others', 'others', 'everything',
        'everyone', 'someone', 'anyone', 'nobody', 'class', 'part', 'chapter',
    }

    # 统计词频
    for w in words:
        w_lower = w.lower()
        if len(w_lower) >= 5 and w_lower not in STOPWORDS and not w_lower.isdigit():
            if w_lower not in word_freq:
                word_freq[w_lower] = 0
            word_freq[w_lower] += 1

    # 排序取top
    sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])

    # 从text中提取与title相关的高频词
    text_topics = [w for w, c in sorted_words[:10] if c >= 2]

    # 合并
    all_topics = text_topics[:4]

    return all_topics if all_topics else ["the main topic"]


def build_9_questions(chunk_id: str, title: str, text: str) -> List[Dict]:
    """为chunk构建9个问题"""
    # 从title提取话题
    title_topics = extract_topic_from_title(title)

    # 从text提取补充话题
    text_topics = extract_topic_from_text(text, title_topics)

    # 合并话题
    all_topics = []
    seen = set()
    for t in title_topics + text_topics:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            all_topics.append(t)

    # 确保有足够的话题
    while len(all_topics) < 3:
        all_topics.append(all_topics[0] if all_topics else "the main topic")

    topics = all_topics[:6]

    # 构建9个问题
    questions = []

    # 3 factual
    questions.append(f"What does the lecturer explain about {topics[0]}?")
    questions.append(f"What are the key facts about {topics[1] if len(topics) > 1 else topics[0]} discussed here?")
    questions.append(f"How does the lecturer describe {topics[2] if len(topics) > 2 else topics[0]}?")

    # 3 fuzzy
    questions.append(f"What's the main point about {topics[0]} in this lecture?")
    questions.append(f"Can you summarize what this chunk says about {topics[1] if len(topics) > 1 else topics[0]}?")
    questions.append(f"What should I understand about {topics[2] if len(topics) > 2 else topics[0]} from this?")

    # 3 intent
    questions.append(f"Why does the lecturer emphasize {topics[0]}?")
    questions.append(f"What is the lecturer's purpose in discussing {topics[1] if len(topics) > 1 else topics[0]}?")
    questions.append(f"What perspective does the lecturer present on {topics[2] if len(topics) > 2 else topics[0]}?")

    # 构建样本
    samples = []
    for q in questions:
        # 清理问题中的格式问题
        q = q.replace('%% topics[0]', '').replace('% topics[0]', '')
        q = re.sub(r'\s+', ' ', q).strip()

        # 移除可能的截断词
        bad_endings = ['?', '!', '.', ',', ' Because', ' Yeah', ' Okay', ' But', ' And']
        for ending in bad_endings:
            if q.endswith(ending):
                q = q[:-len(ending)].strip()

        if len(q.split()) >= 4:
            samples.append({
                'instruction': q,
                'output': chunk_id
            })

    return samples


def distill_dataset(chunks: List[Dict], output_path: str, verbose: bool = True) -> List[Dict]:
    """蒸馏整个数据集"""
    all_samples = []

    print(f"Processing {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        chunk_id = chunk['chunk_id']
        title = chunk.get('lecture_title', '')
        text = chunk['text']

        # 生成问题
        samples = build_9_questions(chunk_id, title, text)
        all_samples.extend(samples)

        if verbose and (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(chunks)}, samples: {len(all_samples)}")

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
    print("Manual Distillation Final - Title-based Questions")
    print("=" * 60)

    # 加载
    chunks = load_json(args.input)
    print(f"Loaded {len(chunks)} chunks")

    # 蒸馏
    samples = distill_dataset(chunks, args.output)

    # 统计
    labels = [s['output'] for s in samples]
    unique_chunks = set(labels)
    print(f"\nStatistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Unique chunks: {len(unique_chunks)}")

    print("\nDone!")


if __name__ == "__main__":
    main()