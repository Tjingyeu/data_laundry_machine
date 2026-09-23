"""
阶段2: 向量索引对照组 (Vector RAG Baseline)
使用 LanceDB + sentence-transformers 建立向量检索基准
"""

import os
import json
import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer


# 加载chunks
def load_chunks(data_path="./data/chunks.json"):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 初始化向量模型和数据库
def init_vector_db(chunks, db_path="./vector_db"):
    print("正在加载向量模型 all-MiniLM-L6-v2...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("正在连接LanceDB...")
    db = lancedb.connect(db_path)

    # 删除旧表（如存在）
    try:
        db.drop_table("subtitles")
    except:
        pass

    # 创建新表
    print("正在生成向量索引...")
    table = db.create_table("subtitles", data=[
        {
            "vector": embed_model.encode(c['text']).tolist(),
            "id": c['id'],
            "filename": c['filename'],
            "text": c['text'],
            "token_count": c['token_count']
        }
        for c in chunks
    ], mode="overwrite")

    print(f"向量索引创建完成，共 {len(chunks)} 条记录")
    return db, table, embed_model


def vector_search(query, table, embed_model, top_k=3):
    """
    向量检索
    返回 top_k 个最相似的 chunk id
    """
    query_vec = embed_model.encode(query)
    results = table.search(query_vec).limit(top_k).to_list()
    return [r['id'] for r in results]


def main():
    # 加载数据
    chunks = load_chunks()
    print(f"加载了 {len(chunks)} 个chunks")

    # 初始化向量数据库
    db, table, embed_model = init_vector_db(chunks)

    # 简单测试
    test_query = "Putin's strategic imagination"
    results = vector_search(test_query, table, embed_model, top_k=3)
    print(f"\n测试查询: '{test_query}'")
    print(f"返回结果: {results}")


if __name__ == "__main__":
    main()