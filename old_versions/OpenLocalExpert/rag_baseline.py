# rag_baseline.py
# 系统B：传统RAG（对照组）

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

INDEX_FILE = "index.faiss"
CHUNKS_NPY = "chunks.npy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class RAGBaseline:
    """传统RAG系统：直接向量检索"""

    def __init__(self):
        print("📦 加载RAG基线系统...")
        self.index = faiss.read_index(INDEX_FILE)
        self.chunks = np.load(CHUNKS_NPY, allow_pickle=True)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ 已加载 {len(self.chunks)} 个chunks")

    def retrieve(self, query, top_k=3):
        """检索最相关的chunks"""
        q_emb = self.model.encode([query])
        D, I = self.index.search(q_emb, top_k)
        return [self.chunks[i] for i in I[0]]

    def ask(self, question, top_k=3):
        """提问并返回检索结果"""
        print(f"\n❓ 用户问题: {question}")
        results = self.retrieve(question, top_k)

        print(f"\n📚 RAG检索结果 (Top-{top_k}):")
        for i, r in enumerate(results, 1):
            print(f"\n--- 结果 {i} ---")
            print(r[:300] + "..." if len(r) > 300 else r)

        return results


def main():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_NPY):
        print("❌ 索引文件不存在!")
        print("请先运行:")
        print("  1. python pdf_to_chunks.py")
        print("  2. python build_index.py")
        exit(1)

    rag = RAGBaseline()

    print("\n" + "="*60)
    print("🧪 传统RAG系统 (对照组)")
    print("="*60)
    print("输入问题进行检索，输入 'quit' 退出")

    while True:
        q = input("\n提问: ").strip()
        if q.lower() == 'quit':
            break
        if q:
            rag.ask(q)


if __name__ == "__main__":
    main()
