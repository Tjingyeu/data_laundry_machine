"""
阶段1: 数据预处理
将12个字幕文件切分为约500 Token的Chunk，输出chunks.json和chunks_metadata.json
"""

import os
import json
import tiktoken


def load_subtitles(folder_path):
    """加载所有.txt字幕文件"""
    subtitles = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                subtitles.append({
                    "filename": filename,
                    "content": content
                })
    return subtitles


def split_into_chunks(subtitles, chunk_size=500):
    """
    按Token数量切分字幕内容
    chunk_size: 每个chunk约500 Token
    """
    # 使用cl100k_base编码器（GPT-4使用的编码器）
    enc = tiktoken.get_encoding("cl100k_base")

    all_chunks = []
    chunk_id_counter = 0

    for subtitle in subtitles:
        filename = subtitle["filename"]
        content = subtitle["content"]

        # 编码成tokens
        tokens = enc.encode(content)

        # 按chunk_size切分
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = enc.decode(chunk_tokens)

            chunk_id = f"{filename}_{chunk_id_counter}"
            all_chunks.append({
                "id": chunk_id,
                "filename": filename,
                "chunk_index": chunk_id_counter,
                "text": chunk_text,
                "token_count": len(chunk_tokens)
            })
            chunk_id_counter += 1

    return all_chunks


def save_chunks(chunks, output_path):
    """保存chunks到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def save_metadata(chunks, output_path):
    """保存chunks元数据映射表"""
    metadata = {
        "total_chunks": len(chunks),
        "files": {}
    }

    for chunk in chunks:
        filename = chunk["filename"]
        if filename not in metadata["files"]:
            metadata["files"][filename] = {
                "total_chunks": 0,
                "chunk_ids": []
            }
        metadata["files"][filename]["chunk_ids"].append(chunk["id"])
        metadata["files"][filename]["total_chunks"] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    txt_folder = "./txt"
    data_folder = "./data"

    print("正在加载字幕文件...")
    subtitles = load_subtitles(txt_folder)
    print(f"加载了 {len(subtitles)} 个字幕文件")

    print("正在切分chunks (500 Token/chunk)...")
    chunks = split_into_chunks(subtitles, chunk_size=500)
    print(f"共生成 {len(chunks)} 个chunks")

    print("正在保存...")
    save_chunks(chunks, os.path.join(data_folder, "chunks.json"))
    save_metadata(chunks, os.path.join(data_folder, "chunks_metadata.json"))
    print("完成!")
    print(f"  - data/chunks.json: {len(chunks)} 个chunks")
    print(f"  - data/chunks_metadata.json: 元数据")


if __name__ == "__main__":
    main()