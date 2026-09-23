# lora_navigation.py
# 系统A：LoRA导航增强RAG（实验组）
# 支持路由式检索

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
import json

INDEX_FILE = "index.faiss"
CHUNKS_NPY = "chunks.npy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LORA_DIR = "expert_lora"  # Expert LoRA目录


class LoRANavigationRAG:
    """LoRA路由增强RAG系统"""

    def __init__(self, use_lora=True):
        print("📦 加载LoRA路由系统...")

        # 加载向量检索组件
        self.index = faiss.read_index(INDEX_FILE)
        self.chunks = np.load(CHUNKS_NPY, allow_pickle=True)
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        self.use_lora = use_lora
        self.model = None
        self.tokenizer = None

        if use_lora:
            self._load_lora_model()

        print(f"✅ 已加载 {len(self.chunks)} 个chunks")

    def _load_lora_model(self):
        """加载MLX LoRA路由模型"""
        if not os.path.exists(LORA_DIR):
            print(f"⚠️ LoRA目录不存在: {LORA_DIR}")
            self.use_lora = False
            return

        try:
            from mlx_lm import load, generate
            from mlx_lm.utils import load_adapters

            print(f"🔄 加载MLX LoRA路由模型...")

            # 加载基础模型
            self.model, self.tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")

            # 加载LoRA适配器
            self.model = load_adapters(self.model, LORA_DIR)

            # 保存generate函数
            self._generate = generate

            print("✅ MLX LoRA路由模型加载成功")
        except Exception as e:
            print(f"❌ LoRA加载失败: {e}")
            self.use_lora = False

    def lora_route(self, question):
        """LoRA路由：将问题路由到正确的Chunk"""
        if self.use_lora and self.model:
            return self._lora_route_to_chunk(question)
        else:
            return self._fallback_route(question)

    def _lora_route_to_chunk(self, question):
        """使用LoRA模型生成Chunk ID"""
        prompt = f"""### Instruction
Route to the correct knowledge chunk.

### Query
{question}

### Answer"""

        try:
            result = self._generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=10
            )

            # 解析Chunk ID
            result = result.strip()
            print(f"   [调试] LoRA原始输出: {repr(result)}")

            # 尝试提取Chunk编号
            # 格式: "domain/intent → Chunk_xxxx" 或 "Chunk_xxxx"
            import re
            match = re.search(r'Chunk_?(\d+)', result)
            if match:
                chunk_id = int(match.group(1))
                if 0 <= chunk_id < len(self.chunks):
                    return chunk_id

            # 格式2: 直接是数字
            match = re.search(r'^(\d+)$', result)
            if match:
                chunk_id = int(match.group(1))
                if 0 <= chunk_id < len(self.chunks):
                    return chunk_id

            # 如果解析失败，返回None
            return None

        except Exception as e:
            print(f"⚠️ LoRA路由失败: {e}")
            return None

    def _fallback_route(self, question):
        """备用路由：使用向量检索"""
        # 简单的向量检索作为fallback
        q_emb = self.embed_model.encode([question])
        D, I = self.index.search(q_emb, 1)
        return I[0][0] if len(I[0]) > 0 else 0

    def retrieve_by_chunk_id(self, chunk_id):
        """通过Chunk ID获取内容"""
        if 0 <= chunk_id < len(self.chunks):
            return self.chunks[chunk_id]
        return None

    def retrieve_by_query(self, query, top_k=3):
        """通过查询词检索"""
        q_emb = self.embed_model.encode([query])
        D, I = self.index.search(q_emb, top_k)
        return [self.chunks[i] for i in I[0]]

    def ask(self, question, top_k=3):
        """提问：LoRA路由 + 检索"""
        print(f"\n❓ 用户问题: {question}")

        # 路由
        chunk_id = self.lora_route(question)

        if chunk_id is not None:
            print(f"\n🧠 LoRA路由: Chunk_{chunk_id:04d}")

            # 获取路由到的chunk
            routed_content = self.retrieve_by_chunk_id(chunk_id)

            if routed_content:
                print(f"\n📚 路由检索结果:")
                print(f"--- Chunk_{chunk_id:04d} ---")
                print(routed_content[:500] + "..." if len(routed_content) > 500 else routed_content)

            # 同时做向量检索对比
            print(f"\n📊 向量检索对比 (Top-3):")
            vector_results = self.retrieve_by_query(question, top_k)
            for i, r in enumerate(vector_results, 1):
                print(f"\n--- Vector-{i} ---")
                print(r[:200] + "..." if len(r) > 200 else r)

            return routed_content, chunk_id
        else:
            # Fallback到向量检索
            print(f"\n⚠️ 路由失败，使用向量检索")
            results = self.retrieve_by_query(question, top_k)
            print(f"\n📚 向量检索结果:")
            for i, r in enumerate(results, 1):
                print(f"\n--- 结果 {i} ---")
                print(r[:300] + "..." if len(r) > 300 else r)
            return results, None


def main():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_NPY):
        print("❌ 索引文件不存在!")
        print("请先运行:")
        print("  1. python pdf_to_chunks.py")
        print("  2. python build_index.py")
        exit(1)

    use_lora = os.path.exists(LORA_DIR)
    mode = "LoRA路由" if use_lora else "向量检索"
    print(f"\n🧠 启动模式: {mode}")

    rag = LoRANavigationRAG(use_lora=use_lora)

    print("\n" + "="*60)
    print(f"🧠 LoRA路由RAG系统 ({mode}模式)")
    print("="*60)

    while True:
        q = input("\n提问: ").strip()
        if q.lower() == 'quit':
            break
        if q:
            rag.ask(q)


if __name__ == "__main__":
    main()
