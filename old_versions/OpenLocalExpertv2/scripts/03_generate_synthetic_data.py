"""
阶段3: 合成数据生成 (Synthetic Data Generation)
使用模板快速生成训练数据 + Ollama qwen3.5:4b 生成部分样本

由于 qwen3.5:4b 生成速度较慢（~2.5分钟/次），
此脚本使用模板方式快速生成数据，并调用 Ollama 生成少量高质量样本。
"""

import os
import json
import random


def load_chunks(data_path="./data/chunks.json"):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_template_sample(chunk):
    """基于模板快速生成训练样本"""
    chunk_id = chunk["id"]
    filename = chunk["filename"]
    chunk_index = chunk["chunk_index"]
    text = chunk["text"]

    # 提取前100字符作为主题摘要
    summary = text[:100].replace('\n', ' ').strip()

    # 随机选择一个模板
    templates = [
        {
            "instruction": "Route to the correct knowledge chunk.",
            "input": f"Where is the part about {summary}?",
            "output": f"{filename} -> {chunk_index}"
        },
        {
            "instruction": "Route to the correct knowledge chunk.",
            "input": f"Find the chunk that discusses this: {summary}",
            "output": f"{filename} -> {chunk_index}"
        },
        {
            "instruction": "Route to the correct knowledge chunk.",
            "input": f"Which chunk contains information about {summary[:50]}?",
            "output": f"{filename} -> {chunk_index}"
        },
        {
            "instruction": "Route to the correct knowledge chunk.",
            "input": f"Locate the content about {summary[:80]}",
            "output": f"{filename} -> {chunk_index}"
        },
        {
            "instruction": "Route to the correct knowledge chunk.",
            "input": f"Search for: {summary[:60]}",
            "output": f"{filename} -> {chunk_index}"
        },
    ]

    return random.choice(templates)


def generate_eval_sample(chunk):
    """生成评测样本（模糊问句）"""
    chunk_id = chunk["id"]
    filename = chunk["filename"]
    chunk_index = chunk["chunk_index"]
    text = chunk["text"]

    # 提取前80字符作为内容摘要
    summary = text[:80].replace('\n', ' ').strip()

    # 模糊问句模板
    templates = [
        f"关于这个话题在哪里？ {summary[:30]}",
        f"这部分内容在哪个文件？ {summary[:40]}",
        f"哪里有相关的分析？ {summary[:35]}",
        f"能找到这段吗？ {summary[:25]}",
        f"这个主题的相关内容在哪？ {summary[:30]}",
    ]

    return {
        "q": random.choice(templates),
        "target": f"{filename} -> {chunk_index}"
    }


def main():
    chunks = load_chunks()
    print(f"加载了 {len(chunks)} 个chunks")

    # 使用模板生成全部数据
    print("使用模板生成合成数据...")
    train_data = [generate_template_sample(c) for c in chunks]
    eval_data = [generate_eval_sample(c) for c in chunks]

    print(f"  训练数据: {len(train_data)} 条")
    print(f"  评测数据: {len(eval_data)} 条")

    # 保存模板生成的数据
    os.makedirs("./data", exist_ok=True)
    with open("./data/train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open("./data/eval.json", 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"\n模板数据已保存:")
    print(f"  - data/train.json")
    print(f"  - data/eval.json")

    print("\n" + "=" * 60)
    print("注意: 当前使用模板生成数据，质量较低")
    print("如需高质量数据，请单独运行 LLM 生成脚本")
    print("=" * 60)


if __name__ == "__main__":
    main()
