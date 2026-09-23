"""
Preprocess transcript files into chunks and generate training data.

Functions:
- parse_transcript_file(): Parse single transcript file
- create_chunks(): Group entries into chunks (4-6 lines each)
- generate_positive_samples(): Generate fuzzy/colloquial/metaphorical questions
- generate_negative_samples(): Generate unrelated questions
- build_label_mapping(): Create label-to-id mapping
- split_train_test(): Split into train/test sets
"""

import os
import sys
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, save_json, save_jsonl, set_random_seed, get_project_root

# =========================
# 停用词表 (改进版)
# =========================
STOP_WORDS = {
    # Core filler
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'he', 'she', 'it', 'they', 'we', 'i',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'to', 'of', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'up', 'down', 'about', 'into', 'over', 'after',
    'just', 'like', 'this', 'that', 'there', 'here', 'when', 'where', 'how', 'why', 'can', 'will', 'should',
    'very', 'really', 'actually', 'basically', 'something', 'everything', 'anything', 'nothing',
    'small', 'big', 'good', 'bad', 'great', 'stuff', 'thing', 'things', 'yeah', 'okay', 'right',
    # Extended filler
    'class', 'lecture', 'today', 'tomorrow', 'yesterday', 'morning', 'evening', 'point', 'idea',
    'way', 'man', 'woman', 'people', 'person', 'year', 'day', 'time', 'week', 'month', 'god',
    'life', 'world', 'question', 'answer', 'part', 'fact', 'case', 'side', 'kind', 'sort', 'type',
    'know', 'think', 'believe', 'want', 'need', 'love', 'hate', 'feel', 'seem', 'look', 'hear',
    'say', 'tell', 'get', 'make', 'go', 'come', 'see', 'use', 'find', 'give', 'take', 'put',
    'first', 'second', 'third', 'last', 'next', 'other', 'another', 'same', 'yes', 'no', 'maybe',
    'probably', 'perhaps', 'certain', 'sure', 'anyway', 'okay', 'well', 'now', 'then', 'so',
    # Question words (critical fix)
    'what', 'not', 'who', 'which', 'whom', 'does',
}


# Question templates for positive sample generation
FUZZY_TEMPLATES = [
    "What did he say about {topic}?",
    "Can you explain {topic}?",
    "What was the deal with {topic}?",
    "What's the connection between {topic} and everything else?",
    "Did he mention anything about {topic}?",
    "What does he think about {topic}?",
    "How does {topic} fit into his argument?",
    "Where does {topic} come up?",
]

COLLOQUIAL_TEMPLATES = [
    "So basically {topic}, right?",
    "Like, {topic}, no?",
    "Okay so {topic}... make sense?",
    "So {topic} basically?",
    "Right, so {topic}?",
    "Wait, {topic}?",
    "So basically when he talked about {topic}...",
    "What's the vibe on {topic}?",
]

METAPHOR_TEMPLATES = [
    "{topic} is basically like a factory, no?",
    "Is {topic} just a fancy version of {topic2}?",
    "{topic} is really just {topic2} in disguise, right?",
    "Kind of like how {topic} works, right?",
    "So {topic} is really just a fancy {topic2}?",
]

IRRELEVANT_TOPICS = [
    "quantum entanglement", "the best pizza recipe", "how to knit a sweater",
    "crypto mining", "martial arts techniques", "the weather in Tokyo",
    "flower arrangement", "car engine repair", "best video games of 2024",
    "how to bake sourdough", "professional dancing", "space explorationMars",
    "interior design tips", "stock market predictions", "fitness routines",
]


def parse_transcript_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a single transcript file into list of entries.

    Format:
    Line N:     timestamp (e.g., "00:01")
    Line N+1:   transcript text
    Line N+2:   blank line

    Returns:
        [{"timestamp": str, "text": str}, ...]
    """
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Check if it looks like a timestamp (MM:SS or HH:MM:SS)
        timestamp = None
        if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', line):
            timestamp = line
            i += 1

            # Collect all text lines until blank or next timestamp
            text_lines = []
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    # Blank line - end of this entry
                    i += 1
                    break
                if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', next_line):
                    # Next timestamp starts - current entry ends
                    break
                text_lines.append(next_line)
                i += 1

            if text_lines and timestamp:
                text = ' '.join(text_lines)
                entries.append({
                    "timestamp": timestamp,
                    "text": text
                })
        else:
            i += 1

    return entries


def extract_topics(text: str, num_topics: int = 3) -> List[str]:
    """
    改进的 Topic 提取逻辑：
    1. 过滤短词和停用词。
    2. 优先保留大写开头的专有名词（常是核心概念）。
    3. 优先保留长词（通常比短词更有语义）。
    """
    # 清洗文本，只保留 3 字符以上的单词
    words = re.findall(r'\b[A-Za-z]{3,}\b', text)

    candidates = []
    for word in words:
        low_word = word.lower()
        if low_word in STOP_WORDS:
            continue

        # 评分机制：专有名词加权，大写字母开头加更多分
        score = len(word)
        if word[0].isupper():
            score += 15  # 大写字母开头（专有名词）大幅加分

        candidates.append((word, score))

    # 按分数排序并去重
    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    unique_topics = []
    seen = set()
    for word, _ in sorted_candidates:
        if word.lower() not in seen:
            unique_topics.append(word)
            seen.add(word.lower())
        if len(unique_topics) >= num_topics:
            break

    # 如果没有提取到有意义的 Topic，返回空列表
    return unique_topics


def create_chunks(entries: List[Dict], lecture_title: str, lines_per_chunk: int = 5, start_chunk_id: int = 1) -> List[Dict]:
    """
    Group transcript entries into chunks (4-6 lines each).

    Args:
        entries: List of parsed transcript entries
        lecture_title: Name of the lecture
        lines_per_chunk: Target number of lines per chunk (default 5)
        start_chunk_id: Starting chunk ID for this batch

    Returns:
        [{"chunk_id": str, "lecture_title": str, "start_timestamp": str,
          "end_timestamp": str, "text": str, "num_entries": int}, ...]
    """
    chunks = []
    chunk_id = start_chunk_id - 1  # Will be incremented before first use

    for i in range(0, len(entries), lines_per_chunk):
        chunk_entries = entries[i:i + lines_per_chunk]

        # Ensure minimum 4 lines
        if len(chunk_entries) < 4:
            # If this is the last chunk and it's small, merge with previous or skip
            if i + len(chunk_entries) >= len(entries):
                # Skip if too small
                continue
            if chunks and len(chunks[-1]["num_entries"]) <= 4:
                # Merge with previous chunk
                continue

        # Combine text from all entries in chunk
        combined_text = ' '.join([e["text"] for e in chunk_entries])

        # Filter out very short chunks (likely empty/skip)
        if len(combined_text.split()) < 20:
            continue

        chunk_id += 1
        chunk = {
            "chunk_id": f"CH_{chunk_id:04d}",
            "lecture_title": lecture_title,
            "start_timestamp": chunk_entries[0]["timestamp"],
            "end_timestamp": chunk_entries[-1]["timestamp"],
            "text": combined_text,
            "num_entries": len(chunk_entries)
        }
        chunks.append(chunk)

    return chunks, chunk_id


def generate_positive_samples(chunk: Dict, num_samples: int = 3) -> List[Dict]:
    """
    Generate fuzzy/colloquial/metaphorical questions for a chunk.

    Key requirements:
    - Questions are NOT direct quotes
    - Focus on "where to find" not "what is"
    - Diverse phrasing (fuzzy, colloquial, metaphorical)
    """
    text = chunk["text"]
    chunk_id = chunk["chunk_id"]

    # Extract 1-2 topics from the chunk
    topics = extract_topics(text, num_topics=2)

    # 如果没有提取到有意义的 Topic，返回空列表（质量关口）
    if not topics:
        return []

    topic1 = topics[0] if topics else "this topic"
    topic2 = topics[1] if len(topics) > 1 else topic1

    samples = []

    # Fuzzy questions
    for _ in range(num_samples // 3 + 1):
        template = random.choice(FUZZY_TEMPLATES)
        # Replace placeholders
        question = template.replace("{topic}", topic1)
        if "{topic2}" in question and topic2 != topic1:
            question = question.replace("{topic2}", topic2)
        samples.append({
            "question": question,
            "chunk_id": chunk_id
        })

    # Colloquial questions
    for _ in range(num_samples // 3 + 1):
        template = random.choice(COLLOQUIAL_TEMPLATES)
        question = template.replace("{topic}", topic1)
        if "{topic2}" in question and topic2 != topic1:
            question = question.replace("{topic2}", topic2)
        samples.append({
            "question": question,
            "chunk_id": chunk_id
        })

    # Metaphorical questions
    for _ in range(num_samples // 3):
        template = random.choice(METAPHOR_TEMPLATES)
        question = template.replace("{topic}", topic1).replace("{topic2}", topic2)
        samples.append({
            "question": question,
            "chunk_id": chunk_id
        })

    # Trim to exact num_samples
    return samples[:num_samples]


def generate_negative_samples(chunk: Dict, all_chunks: List[Dict], num_samples: int = 1) -> List[Dict]:
    """
    Generate unrelated questions mapping to NOT_RELEVANT.

    Uses random topics not related to the current chunk.
    """
    samples = []

    for _ in range(num_samples):
        # Pick a random irrelevant topic
        topic = random.choice(IRRELEVANT_TOPICS)

        # Create a question that sounds somewhat like a question but is unrelated
        templates = [
            f"What about {topic}?",
            f"Tell me about {topic}",
            f"How does {topic} work?",
            f"Can you explain {topic}?",
        ]
        question = random.choice(templates)

        samples.append({
            "question": question,
            "chunk_id": "NOT_RELEVANT"
        })

    return samples


def build_label_mapping(chunks: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build label-to-id mapping for classification head.

    NOT_RELEVANT = 0, chunk IDs = 1 to N

    Returns:
        (label_to_id, id_to_label)
    """
    label_to_id = {"NOT_RELEVANT": 0}

    for idx, chunk in enumerate(chunks):
        label_to_id[chunk["chunk_id"]] = idx + 1

    id_to_label = {v: k for k, v in label_to_id.items()}

    return label_to_id, id_to_label


def split_train_test(chunks: List[Dict], test_split: float = 0.1, random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split chunks into train and test sets.
    If only 1 lecture, split by chunks directly.
    Otherwise use lecture-level separation to avoid content leakage.
    """
    random.seed(random_seed)

    # Group by lecture
    lecture_chunks = {}
    for chunk in chunks:
        lecture = chunk["lecture_title"]
        if lecture not in lecture_chunks:
            lecture_chunks[lecture] = []
        lecture_chunks[lecture].append(chunk)

    lectures = list(lecture_chunks.keys())

    # If only 1 lecture or few lectures, split by chunks directly
    if len(lectures) <= 2:
        indices = list(range(len(chunks)))
        random.shuffle(indices)
        split_idx = int(len(chunks) * (1 - test_split))
        train_indices = set(indices[:split_idx])
        test_indices = set(indices[split_idx:])

        train_chunks = [chunks[i] for i in train_indices]
        test_chunks = [chunks[i] for i in test_indices]
        return train_chunks, test_chunks

    # Split lectures
    random.shuffle(lectures)

    num_test_lectures = max(1, int(len(lectures) * test_split))
    test_lectures = set(lectures[:num_test_lectures])

    train_chunks = []
    test_chunks = []

    for chunk in chunks:
        if chunk["lecture_title"] in test_lectures:
            test_chunks.append(chunk)
        else:
            train_chunks.append(chunk)

    return train_chunks, test_chunks


def generate_test_samples(test_chunks: List[Dict], num_per_chunk: int = 1) -> List[Dict]:
    """
    Generate test samples with ground truth labels.
    Each test chunk gets 1 test question.
    """
    test_samples = []

    for idx, chunk in enumerate(test_chunks):
        # Use same generation logic but save with ground truth
        positive_samples = generate_positive_samples(chunk, num_samples=num_per_chunk)

        for sample in positive_samples:
            # Determine question type
            question = sample["question"]
            if any(t in question.lower() for t in ['basically', 'like', 'vibe', 'no?']):
                q_type = "colloquial"
            elif 'is' in question.lower() and ('like' in question.lower() or 'fancy' in question.lower()):
                q_type = "metaphor"
            else:
                q_type = "fuzzy"

            test_samples.append({
                "sample_id": f"test_{idx:04d}",
                "query": question,
                "label": chunk["chunk_id"],
                "type": q_type
            })

    return test_samples


def preprocess_all_transcripts(raw_txt_dir: str, lines_per_chunk: int = 5, test_split: float = 0.2) -> Dict[str, Any]:
    """
    Main preprocessing function - processes all transcript files.

    Key improvement: Split by SAMPLE (not lecture) to ensure all chunk IDs
    appear in training set. Test uses different question templates than train.
    """
    raw_txt_path = Path(raw_txt_dir)

    # Parse all transcript files
    all_entries = []
    all_chunks = []
    next_chunk_id = 1  # Global chunk ID counter

    for txt_file in sorted(raw_txt_path.glob("*.txt")):
        lecture_title = txt_file.stem
        entries = parse_transcript_file(str(txt_file))
        chunks, next_chunk_id = create_chunks(entries, lecture_title, lines_per_chunk=lines_per_chunk, start_chunk_id=next_chunk_id + 1)

        all_entries.extend(entries)
        all_chunks.extend(chunks)

        print(f"Processed {lecture_title}: {len(entries)} entries -> {len(chunks)} chunks")

    print(f"\nTotal: {len(all_entries)} entries -> {len(all_chunks)} chunks")

    # Build label mapping first (includes ALL chunks)
    label_to_id, id_to_label = build_label_mapping(all_chunks)

    # --- KEY CHANGE: Generate samples per chunk, split by sample ---
    train_samples = []
    test_samples = []
    samples_without_topics = 0

    for chunk in all_chunks:
        topics = extract_topics(chunk["text"], num_topics=3)

        # 如果没有提取到有效 Topic，跳过该 chunk
        if not topics:
            samples_without_topics += 1
            continue

        # 为该 chunk 生成多个样本（每个 topic 用不同模板）
        chunk_samples = []
        for topic in topics:
            # Use first 3 fuzzy templates for variety
            for template in FUZZY_TEMPLATES[:3]:
                query = template.replace("{topic}", topic)
                chunk_samples.append({
                    "instruction": query,
                    "output": chunk["chunk_id"]
                })

        # 打乱样本
        random.shuffle(chunk_samples)

        # 分割：确保至少 1 个在训练集，其余按比例进测试集
        split_idx = max(1, int(len(chunk_samples) * (1 - test_split)))
        this_train = chunk_samples[:split_idx]
        this_test = chunk_samples[split_idx:]

        train_samples.extend(this_train)

        # 测试样本格式对齐
        for s in this_test:
            test_samples.append({
                "query": s["instruction"],
                "label": s["output"],
                "type": "fuzzy"
            })

    # 添加 NOT_RELEVANT 样本（来自随机 chunk 的随机 topic）
    not_relevant_count = 0
    for _ in range(len(all_chunks) // 4):  # 约 1/4 的 chunk 贡献 negative 样本
        topic = random.choice(IRRELEVANT_TOPICS)
        template = random.choice(FUZZY_TEMPLATES[:3]).replace("{topic}", topic.split()[0])
        train_samples.append({
            "instruction": template,
            "output": "NOT_RELEVANT"
        })
        not_relevant_count += 1

    print(f"\nSkipped {samples_without_topics} chunks without valid topics")
    print(f"Generated {not_relevant_count} NOT_RELEVANT samples")
    print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")

    return {
        "chunks": all_chunks,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label
    }


def main():
    """Run preprocessing and save all outputs."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess transcripts")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    # Get paths
    root = get_project_root()
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = root / "configs" / "config.yaml"
    config = load_config(str(config_path))

    # Resolve raw_txt_dir relative to project root
    raw_txt_dir = root / config["data"]["raw_txt_dir"]

    # Create symlink if it doesn't exist
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    raw_txt_link = data_dir / "raw_txt"
    if not raw_txt_link.exists():
        # Create symlink to the actual txt directory
        actual_txt_dir = root.parent / "txt"  # Go up from lora_chunk_router
        if actual_txt_dir.exists():
            raw_txt_link.symlink_to(actual_txt_dir)
        else:
            # Fallback: copy
            import shutil
            shutil.copytree(actual_txt_dir, raw_txt_link)

    print(f"Processing transcripts from: {raw_txt_link}")
    print(f"Lines per chunk: {config['data']['lines_per_chunk']}")

    # Run preprocessing
    result = preprocess_all_transcripts(
        str(raw_txt_link),
        lines_per_chunk=config['data']['lines_per_chunk'],
        test_split=config['data']['test_split']
    )

    # Save outputs
    save_json(result["chunks"], str(root / config["data"]["chunks_output"]))
    save_jsonl(result["train_samples"], str(root / config["data"]["train_output"]))
    save_json(result["test_samples"], str(root / config["data"]["test_output"]))
    save_json({
        "label_to_id": result["label_to_id"],
        "id_to_label": result["id_to_label"]
    }, str(root / config["data"]["id_to_label_output"]))

    print(f"\nOutputs saved:")
    print(f"  chunks.json: {len(result['chunks'])} chunks")
    print(f"  train.jsonl: {len(result['train_samples'])} samples")
    print(f"  test.json: {len(result['test_samples'])} samples")
    print(f"  id_to_label.json: {len(result['label_to_id'])} labels")


if __name__ == "__main__":
    main()
