# build_index.py
# 向量索引构建

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

CHUNKS_FILE = "chunks.txt"
INDEX_FILE = "index.faiss"
CHUNKS_NPY = "chunks.npy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_chunks(filepath=CHUNKS_FILE):
    """从文件加载chunks"""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # 按 "===" 分隔chunks
    parts = raw.split("===")
    chunks = []
    for part in parts:
        part = part.strip()
        if part:
            # 去掉 "--- Chunk N ---" 这行
            lines = part.split("\n")
            content_lines = [l for l in lines if not l.startswith("--- Chunk")]
            content = " ".join(content_lines).strip()
            if content:
                chunks.append(content)
    return chunks


def build_index(chunks, model_name=EMBEDDING_MODEL):
    """构建FAISS向量索引"""
    print(f"🔄 加载嵌入模型: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"🔄 编码 {len(chunks)} 个chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)

    print(f"🔄 构建FAISS索引...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    return index, model


if __name__ == "__main__":
    # 检查chunks文件
    if not os.path.exists(CHUNKS_FILE):
        print(f"❌ chunks文件不存在: {CHUNKS_FILE}")
        print("请先运行: python pdf_to_chunks.py")
        exit(1)

    # 加载chunks
    chunks = load_chunks()
    print(f"✅ 加载 {len(chunks)} 个chunks")

    # 构建索引
    index, model = build_index(chunks)

    # 保存
    faiss.write_index(index, INDEX_FILE)
    np.save(CHUNKS_NPY, np.array(chunks, dtype=object))
    print(f"✅ 索引已保存:")
    print(f"   - FAISS: {INDEX_FILE}")
    print(f"   - Chunks: {CHUNKS_NPY}")
