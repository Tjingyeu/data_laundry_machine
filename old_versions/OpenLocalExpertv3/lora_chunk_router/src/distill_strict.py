"""
Manual Distillation v3 - 严格过滤版
使用更严格的话题提取和问题生成策略
"""

import os
import sys
import json
import random
import re
from typing import List, Dict
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_json, save_jsonl


# 极严格停用词
VERY_STRICT_STOPWORDS = {
    # 代词/限定词
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
    # 口语
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
    'get', 'gets', 'getting', 'got', 'gotten',
    # 形容词/副词
    'good', 'bad', 'big', 'small', 'new', 'old', 'first', 'last', 'long',
    'great', 'little', 'other', 'another', 'such', 'even', 'still', 'already',
    'more', 'most', 'less', 'least', 'much', 'many', 'few',
    'very', 'really', 'actually', 'basically', 'simply', 'only', 'just',
    'always', 'never', 'sometimes', 'often', 'usually', 'ever', 'already',
    # 名词（太通用）
    'people', 'thing', 'things', 'something', 'everything', 'nothing',
    'anything', 'way', 'ways', 'point', 'points', 'part', 'parts',
    'time', 'times', 'year', 'years', 'day', 'days', 'hour', 'hours',
    'man', 'men', 'woman', 'women', 'child', 'children', 'person', 'persons',
    'world', 'life', 'death', 'fact', 'facts', 'reason', 'reasons',
    'side', 'end', 'ends', 'case', 'matter', 'group', 'groups',
    'number', 'numbers', 'lot', 'lots', 'kind', 'kinds', 'type', 'types',
    'sort', 'sorts', 'bit', 'piece', 'pieces', 'part', 'parts',
    # 学术/lecture词
    'lecture', 'class', 'professor', 'student', 'students', 'teacher',
    'course', 'lesson', 'chapter', 'topic', 'topics', 'subject', 'subjects',
    'material', 'question', 'questions', 'answer', 'answers',
    'example', 'examples', 'idea', 'ideas', 'concept', 'concepts',
    'theory', 'theories', 'argument', 'arguments', 'evidence',
    'discussion', 'discussions', 'explain', 'explains', 'explained',
    'understanding', 'understand', 'understood', 'learn', 'learns', 'learned',
    'teach', 'teaches', 'taught', 'study', 'studies', 'studied',
    # 常见动词
    'going', 'coming', 'taking', 'giving', 'making', 'doing', 'getting',
    'having', 'being', 'seeming', 'feeling', 'feeling', 'think', 'thinking',
    'wanting', 'needing', 'trying', 'looking', 'watching', 'seeing',
    'hearing', 'saying', 'telling', 'ask', 'asks', 'asked',
    'wonder', 'wonders', 'wondered', 'guess', 'guesses', 'suppose', 'supposes',
    'believe', 'believes', 'believed', 'consider', 'considers', 'considered',
    'find', 'finds', 'found', 'realize', 'realizes', 'realized',
    'notice', 'notices', 'noticed', 'remember', 'remembers', 'remembered',
    'forget', 'forgets', 'forgot', 'mean', 'means', 'meant', 'meaning',
    # 形容词
    'important', 'interesting', 'significant', 'relevant', 'useful',
    'clear', 'obvious', 'simple', 'complex', 'difficult', 'easy', 'hard',
    'true', 'false', 'real', 'actual', 'possible', 'impossible',
    'necessary', 'essential', 'special', 'particular', 'general', 'specific',
    'certain', 'sure', 'likely', 'unlikely', 'probably', 'maybe', 'perhaps',
    # 无意义词
    'however', 'although', 'though', 'since', 'until', 'unless', 'therefore',
    'thus', 'hence', 'maybe', 'perhaps', 'probably', 'actually', 'really',
    'basically', 'certainly', 'definitely', 'exactly', 'especially', 'particularly',
    'generally', 'usually', 'frequently', 'occasionally', 'rarely', 'seldom',
    'anyway', 'anyways', 'whatever', 'besides', 'instead', 'anymore',
    # 代词变体
    'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    'one', 'ones', 'someone', 'anyone', 'everyone', 'noone', 'none',
    'something', 'anything', 'everything', 'nothing',
    'somebody', 'anybody', 'everybody', 'nobody',
    # 开头的单词（常见但无意义）
    'let', 'make', 'get', 'got', 'see', 'saw', 'know', 'knew',
    'think', 'thought', 'say', 'said', 'says', 'tell', 'told',
    'ask', 'asks', 'asked', 'want', 'wants', 'wanted',
    'use', 'uses', 'used', 'find', 'finds', 'found',
    'give', 'gives', 'gave', 'take', 'takes', 'took', 'taken',
    'come', 'comes', 'came', 'going', 'goes', 'went',
    'like', 'likes', 'liked', 'need', 'needs', 'needed',
    'look', 'looks', 'looked', 'work', 'works', 'worked',
    'play', 'plays', 'played', 'live', 'lives', 'lived',
    'believe', 'believes', 'believed', 'hold', 'holds', 'held',
    'seem', 'seems', 'seemed', 'help', 'helps', 'helped',
    'show', 'shows', 'showed', 'hear', 'hears', 'heard',
    'play', 'plays', 'played', 'run', 'runs', 'ran',
    'move', 'moves', 'moved', 'like', 'likes', 'liked',
    'try', 'tries', 'tried', 'leave', 'leaves', 'left',
    'call', 'calls', 'called', 'keep', 'keeps', 'kept',
    'begin', 'begins', 'began', 'seem', 'seems', 'seemed',
    'help', 'helps', 'helped', 'show', 'shows', 'showed',
    'happen', 'happens', 'happened', 'write', 'writes', 'wrote',
    'provide', 'provides', 'provided', 'sit', 'sits', 'sat',
    'stand', 'stands', 'stood', 'lose', 'loses', 'lost',
    'pay', 'pays', 'paid', 'meet', 'meets', 'met',
    'include', 'includes', 'included', 'continue', 'continues', 'continued',
    'set', 'sets', 'learn', 'learns', 'learned',
    'change', 'changes', 'changed', 'lead', 'leads', 'led',
    'understand', 'understands', 'understood', 'watch', 'watches', 'watched',
    'follow', 'follows', 'followed', 'stop', 'stops', 'stopped',
    'create', 'creates', 'created', 'speak', 'speaks', 'spoke', 'spoken',
    'read', 'reads', 'allow', 'allows', 'allowed',
    'add', 'adds', 'added', 'spend', 'spends', 'spent',
    'grow', 'grows', 'grew', 'open', 'opens', 'opened',
    'walk', 'walks', 'walked', 'win', 'wins', 'won',
    'offer', 'offers', 'offered', 'remember', 'remembers', 'remembered',
    'love', 'loves', 'loved', 'consider', 'considers', 'considered',
    'appear', 'appears', 'appeared', 'buy', 'buys', 'bought',
    'wait', 'waits', 'waited', 'serve', 'serves', 'served',
    'die', 'dies', 'died', 'send', 'sends', 'sent',
    'expect', 'expects', 'expected', 'build', 'builds', 'built',
    'stay', 'stays', 'stayed', 'fall', 'falls', 'fell', 'fallen',
    'cut', 'cuts', 'reach', 'reaches', 'reached',
    'kill', 'kills', 'killed', 'remain', 'remains', 'remained',
    'suggest', 'suggests', 'suggested', 'raise', 'raises', 'raised',
    'pass', 'passes', 'passed', 'sell', 'sells', 'sold',
    'require', 'requires', 'required', 'report', 'reports', 'reported',
    'decide', 'decides', 'decided', 'pull', 'pulls', 'pulled',
}


def is_valid_topic(word: str) -> bool:
    """严格判断是否为有效话题"""
    word_lower = word.lower().strip()

    # 太短
    if len(word_lower) < 5:
        return False

    # 停用词
    if word_lower in VERY_STRICT_STOPWORDS:
        return False

    # 纯数字
    if word_lower.isdigit():
        return False

    # 包含数字
    if any(c.isdigit() for c in word_lower):
        return False

    # 以常见动词开头
    verb_starts = {'let', 'make', 'get', 'see', 'know', 'think', 'say', 'tell',
                   'ask', 'want', 'use', 'find', 'give', 'take', 'come', 'go',
                   'like', 'need', 'look', 'work', 'play', 'live', 'believe',
                   'hold', 'seem', 'help', 'show', 'happen', 'write', 'speak',
                   'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk',
                   'win', 'offer', 'remember', 'love', 'consider', 'appear',
                   'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build',
                   'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'suggest',
                   'pass', 'sell', 'require', 'report', 'decide', 'pull'}
    if word_lower in verb_starts:
        return False

    # 无意义词
    bad_words = {'thing', 'things', 'something', 'anything', 'everything',
                 'nothing', 'someone', 'anyone', 'everyone', 'nobody',
                 'way', 'ways', 'point', 'time', 'times', 'year', 'years',
                 'lot', 'lots', 'kind', 'type', 'sort', 'part', 'parts',
                 'stuff', 'whatever', 'anyway', 'anyways', 'maybe', 'perhaps'}
    if word_lower in bad_words:
        return False

    return True


def extract_quality_topics(text: str) -> List[str]:
    """提取高质量的话题"""
    # 清理 - 移除口语填充
    clean = re.sub(r'>>', ' ', text)
    clean = re.sub(r'\b(Okay|Ok|uh|um|like you know|you know)\b', ' ', clean, flags=re.IGNORECASE)
    clean = re.sub(r'[^\w\s]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean)

    # 提取大写专有名词（人名、地名、事件名等）
    proper_nouns = []
    caps_pattern = r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b'
    for match in re.findall(caps_pattern, clean):
        if is_valid_topic(match):
            proper_nouns.append(match)

    # 词频统计
    words = clean.split()
    word_freq = Counter(w.lower() for w in words)

    # 选择高频且有效的话题
    topics = []
    for word, freq in word_freq.most_common(150):
        if is_valid_topic(word) and freq >= 2:
            topics.append(word)
        if len(topics) >= 12:
            break

    # 合并：专有名词优先
    all_topics = proper_nouns[:6] + topics[:4]

    # 去重（不区分大小写）
    seen = set()
    unique = []
    for t in all_topics:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique.append(t)

    return unique[:10]


def build_9_questions(chunk_id: str, text: str, topics: List[str]) -> List[Dict]:
    """为chunk构建9个问题"""
    questions = []

    if not topics or len(topics) < 3:
        # 如果话题不足，使用默认
        topics = ["the main topic", "key concepts", "important details"]

    # 使用不同的话题轮流
    t0 = topics[0] if topics else "the topic"
    t1 = topics[1] if len(topics) > 1 else topics[0]
    t2 = topics[2] if len(topics) > 2 else topics[0]

    # 9个不同的问题（3 factual, 3 fuzzy, 3 intent）
    question_data = [
        # Factual - 理解内容
        (f"What does the lecturer explain about {t0}?", "factual"),
        (f"What are the key facts about {t1} discussed here?", "factual"),
        (f"How does the lecturer describe {t2}?", "factual"),

        # Fuzzy - 总结概括
        (f"What's the main point the lecturer makes about {t0}?", "fuzzy"),
        (f"Can you summarize what this chunk says about {t1}?", "fuzzy"),
        (f"What should I understand about {t2} from this?", "fuzzy"),

        # Intent - 目的观点
        (f"Why does the lecturer emphasize {t0}?", "intent"),
        (f"What is the lecturer's purpose in discussing {t1}?", "intent"),
        (f"What perspective does the lecturer present on {t2}?", "intent"),
    ]

    for question, qtype in question_data:
        questions.append({
            'instruction': question,
            'output': chunk_id
        })

    return questions


def distill_dataset(chunks: List[Dict], output_path: str, verbose: bool = True) -> List[Dict]:
    """蒸馏整个数据集"""
    all_samples = []

    print(f"Processing {len(chunks)} chunks, generating 9 questions each...")

    for i, chunk in enumerate(chunks):
        chunk_id = chunk['chunk_id']
        text = chunk['text']

        # 提取高质量话题
        topics = extract_quality_topics(text)

        # 生成问题
        samples = build_9_questions(chunk_id, text, topics)
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
    print("Manual Distillation v3 - Strict Filter")
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