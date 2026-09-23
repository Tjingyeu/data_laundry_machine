"""
Manual Distillation v2 - 改进版
基于chunk的真正内容生成9个有针对性的问题
"""

import os
import sys
import json
import random
import re
from typing import List, Dict, Tuple
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_json, save_jsonl


# 强停用词 - 这些词太通用，不能作为话题
STRONG_STOPWORDS = {
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
    'however', 'although', 'though', 'while', 'since', 'until', 'unless',
    'therefore', 'thus', 'hence', 'maybe', 'perhaps', 'probably', 'actually',
    'really', 'basic', 'certain', 'various', 'kind', 'type', 'remember',
    'forget', 'example', 'idea', 'important', 'interesting', 'different',
    'together', 'human', 'humans', 'believe', 'sense', 'understand',
    'explain', 'describe', 'mean', 'means', 'meaning', 'question', 'answer',
    'problem', 'answer', 'help', 'try', 'trying', 'done', 'doing', 'doesnt',
    'isnt', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt', 'couldnt',
    'wouldnt', 'shouldnt', 'wont', 'dont', 'didnt', 'nothing', 'everything',
    'something', 'anything', 'someone', 'anyone', 'everyone', 'nobody',
    'body', 'anybody', 'person', 'person', 'guy', 'guys', 'everybody',
    'start', 'started', 'starting', 'begin', 'begins', 'beginning', 'began',
    'end', 'ends', 'ending', 'ended', 'continue', 'continues', 'continued',
    'happen', 'happens', 'happened', 'happening', 'become', 'becomes', 'became',
    'because', 'which', 'while', 'although', 'though', 'but', 'however',
    'therefore', 'thus', 'hence', 'so', 'than', 'then', 'when', 'where',
    'why', 'how', 'what', 'who', 'whom', 'whose', 'if', 'unless', 'until',
    'before', 'after', 'during', 'about', 'above', 'below', 'between',
    'into', 'through', 'across', 'over', 'under', 'again', 'further',
    'once', 'here', 'there', 'where', 'when', 'why', 'how', 'ever',
    'now', 'just', 'also', 'very', 'too', 'more', 'most', 'less', 'least',
    'same', 'different', 'own', 'other', 'another', 'such', 'only', 'even',
    'well', 'still', 'already', 'yet', 'almost', 'enough', 'quite', 'rather',
    'actually', 'really', 'probably', 'possibly', 'perhaps', 'maybe',
    'certainly', 'definitely', 'exactly', 'simply', 'only', 'especially',
    'particularly', 'generally', 'usually', 'always', 'never', 'sometimes',
    'often', 'rarely', 'seldom', 'occasionally', 'frequently', 'rarely',
    # 口语/填充词
    'okay', 'ok', 'yeah', 'gonna', 'wanna', 'kinda', 'sorta', 'dunno',
    'em', 'lemme', 'gimme', 'gotta', 'hafta', 'oughta', 'woulda', 'coulda',
    'shoulda', 'aint', 'yall', 'folks', 'anyway', 'whatever', 'anyways',
    'besides', 'instead', 'however', 'though', 'still', 'yet', 'anymore',
    # 学术/lecture常用但无意义的
    'lecture', 'class', 'professor', 'student', 'students', 'teacher',
    'course', 'lesson', 'chapter', 'topic', 'topics', 'subject', 'material',
    'question', 'questions', 'answer', 'answers', 'example', 'examples',
    'point', 'points', 'idea', 'ideas', 'concept', 'concepts', 'theory',
    'theories', 'fact', 'facts', 'evidence', 'argument', 'arguments',
    'discussion', 'discussions', 'explain', 'explains', 'explained', 'explain',
    'understand', 'understood', 'understanding', 'learn', 'learns', 'learned',
    'teach', 'teaches', 'taught', 'study', 'studies', 'studied', 'research',
    # 动词和形容词（太通用）
    'going', 'coming', 'taking', 'giving', 'making', 'doing', 'getting',
    'having', 'being', 'seeming', 'feeling', 'thinking', 'wanting', 'needing',
    'trying', 'looking', 'watching', 'seeing', 'hearing', 'saying', 'telling',
    'ask', 'asks', 'asked', 'question', 'questions', 'wonder', 'wonders',
    'wondered', 'guess', 'guesses', 'suppose', 'supposes', 'assumed',
    'believe', 'believes', 'believed', 'consider', 'considers', 'considered',
    'feel', 'feels', 'felt', 'find', 'finds', 'found', 'realize', 'realizes',
    'realized', 'notice', 'notices', 'noticed', 'remember', 'remembers',
    'remembered', 'forget', 'forgets', 'forgot', 'mean', 'means', 'meant',
    'important', 'interesting', 'significant', 'relevant', 'useful',
    'clear', 'obvious', 'simple', 'complex', 'difficult', 'easy', 'hard',
}


def is_valid_topic(word: str) -> bool:
    """检查是否为有效的话题词"""
    word_lower = word.lower().strip()

    # 太短
    if len(word_lower) < 4:
        return False

    # 停用词
    if word_lower in STRONG_STOPWORDS:
        return False

    # 全是数字
    if word_lower.isdigit():
        return False

    # 包含数字
    if any(c.isdigit() for c in word_lower):
        return False

    # 常见无意义模式
    bad_patterns = ['thing', 'stuff', 'whatever', 'something', 'anything',
                    'nothing', 'everything', 'someone', 'anyone', 'everyone',
                    'gonna', 'wanna', 'kinda', 'sorta', 'dunno', 'lotta']
    if word_lower in bad_patterns:
        return False

    return True


def extract_quality_topics(text: str) -> List[str]:
    """提取高质量的话题词"""
    # 清理
    clean = re.sub(r'>>', ' ', text)
    clean = re.sub(r'\b(Okay|Ok|uh|um|like you know)\b', ' ', clean, flags=re.IGNORECASE)
    clean = re.sub(r'[^\w\s]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean)

    # 提取所有单词
    words = clean.split()

    # 提取连续大写的专有名词
    proper_nouns = []
    caps_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    for match in re.findall(caps_pattern, text):
        if len(match) > 2 and match.lower() not in STRONG_STOPWORDS:
            proper_nouns.append(match)

    # 词频统计
    word_freq = Counter(w.lower() for w in words)

    # 过滤出高频且有效的话题
    topics = []
    for word, freq in word_freq.most_common(100):
        if is_valid_topic(word) and freq >= 2:
            topics.append(word)
        if len(topics) >= 10:
            break

    # 合并专有名词（优先）
    all_topics = proper_nouns[:8] + topics[:5]

    # 去重（不区分大小写）
    seen = set()
    unique = []
    for t in all_topics:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique.append(t)

    return unique[:10]


def build_questions_for_chunk(chunk_id: str, text: str, topics: List[str]) -> List[str]:
    """为chunk构建9个多样化的问题"""
    questions = []

    if not topics:
        # 如果没有有效话题，使用默认
        topics = ["the main topic"]

    # 确保有9个不同的问题
    question_templates = [
        # Factual - 3个
        lambda t: f"What does the lecturer explain about {t}?",
        lambda t: f"What are the key details about {t} discussed in this chunk?",
        lambda t: f"How does the lecturer describe {t}?",

        # Fuzzy - 3个
        lambda t: f"What's the main point about {t} in this lecture?",
        lambda t: f"Can you summarize what this chunk says about {t}?",
        lambda t: f"What should I understand about {t} from this?",

        # Intent/Perspective - 3个
        lambda t: f"Why does the lecturer emphasize {t}?",
        lambda t: f"What is the lecturer's purpose in discussing {t}?",
        lambda t: f"What perspective does the lecturer present on {t}?",
    ]

    # 轮流使用不同的话题和问题模板
    for i, template in enumerate(question_templates):
        topic = topics[i % len(topics)]
        q = template(topic)

        # 清理问题
        q = re.sub(r'\s+', ' ', q).strip()

        # 确保问题长度合理
        words = q.split()
        if 4 <= len(words) <= 45:
            questions.append(q)
        else:
            # 简化问题
            if i % 3 == 0:
                questions.append(f"What is discussed about {topic}?")
            elif i % 3 == 1:
                questions.append(f"What's the point about {topic}?")
            else:
                questions.append(f"Why does the lecturer mention {topic}?")

    return questions[:9]


def distill_dataset(chunks: List[Dict], output_path: str, verbose: bool = True) -> List[Dict]:
    """蒸馏整个数据集"""
    all_samples = []

    print(f"Processing {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        chunk_id = chunk['chunk_id']
        text = chunk['text']

        # 提取高质量话题
        topics = extract_quality_topics(text)

        # 生成问题
        questions = build_questions_for_chunk(chunk_id, text, topics)

        # 构建样本
        for q in questions:
            all_samples.append({
                'instruction': q,
                'output': chunk_id
            })

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
    print("Manual Distillation v2 - 9 Questions per Chunk")
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
    print(f"  Avg questions per chunk: {len(samples) / len(unique_chunks):.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()