"""
阶段5: 交叉验证引擎 (Evaluation)
对比向量检索和LoRA路由的性能
评测指标: Hit Rate, Latency, Token Saving
"""

import os
import json
import time
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3.5:4b"


def load_eval_data(data_path="./data/eval.json"):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_chunks_for_context(data_path="./data/chunks.json"):
    """加载chunks并建立filename->chunk_index->text的映射"""
    chunks = json.load(open(data_path, 'r', encoding='utf-8'))
    chunk_map = {}
    for c in chunks:
        key = f"{c['filename']} -> {c['chunk_index']}"
        chunk_map[key] = c
    return chunk_map


def call_lora_inference(query):
    """
    调用LoRA adapter进行推理
    返回: (predicted_chunk_id, latency)
    """
    start = time.time()

    # 构建prompt
    prompt = f"""You are a routing expert. Given a query, return the exact chunk ID that contains the answer.

**Query:** {query}

**Response format:**
<filename> -> <chunk_index>"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 128,
            "thinking": False
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json().get("response", "").strip()

        # 解析output格式: filename -> chunk_index
        if "->" in result:
            parts = result.split("->")
            filename = parts[0].strip().strip('"\'')
            chunk_index = parts[1].strip().strip('"\'')
            predicted_id = f"{filename} -> {chunk_index}"
        else:
            predicted_id = result.strip()

        latency = time.time() - start
        return predicted_id, latency

    except Exception as e:
        latency = time.time() - start
        return f"ERROR: {str(e)}", latency


def init_vector_db():
    """初始化向量数据库"""
    import lancedb
    from sentence_transformers import SentenceTransformer

    chunks = json.load(open("./data/chunks.json", 'r', encoding='utf-8'))
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    db = lancedb.connect("./vector_db")

    try:
        db.drop_table("subtitles")
    except:
        pass

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

    return db, table, embed_model


def vector_search(query, table, embed_model, top_k=3):
    """向量检索"""
    query_vec = embed_model.encode(query)
    results = table.search(query_vec).limit(top_k).to_list()
    return [r['id'] for r in results]


def run_experiment():
    """运行对比实验"""
    print("=" * 60)
    print("交叉验证实验")
    print("=" * 60)

    # 加载数据
    eval_data = load_eval_data()
    print(f"加载了 {len(eval_data)} 条评测数据")

    chunk_map = load_chunks_for_context()
    print(f"加载了 {len(chunk_map)} 个chunks")

    # 初始化向量数据库
    print("\n初始化向量数据库...")
    db, table, embed_model = init_vector_db()

    # 运行实验
    results = []
    vector_hits = 0
    lora_hits = 0
    vector_total_latency = 0
    lora_total_latency = 0

    print(f"\n开始评测（共 {len(eval_data)} 条）...")

    for i, item in enumerate(eval_data):
        query = item["q"]
        target = item["target"]

        # --- 向量检索 ---
        start = time.time()
        v_ids = vector_search(query, table, embed_model, top_k=3)
        v_latency = time.time() - start
        v_hit = 1 if target in v_ids else 0

        # --- LoRA 推理 ---
        lora_pred, l_latency = call_lora_inference(query)
        l_hit = 1 if target == lora_pred else 0

        # 记录
        results.append({
            "query": query,
            "target": target,
            "vector_ids": v_ids,
            "vector_hit": v_hit,
            "vector_latency": v_latency,
            "lora_pred": lora_pred,
            "lora_hit": l_hit,
            "lora_latency": l_latency
        })

        vector_hits += v_hit
        lora_hits += l_hit
        vector_total_latency += v_latency
        lora_total_latency += l_latency

        if (i + 1) % 10 == 0:
            print(f"  进度: {i + 1}/{len(eval_data)}")

    # 计算统计
    n = len(eval_data)
    vector_hit_rate = vector_hits / n
    lora_hit_rate = lora_hits / n
    avg_vector_latency = vector_total_latency / n
    avg_lora_latency = lora_total_latency / n

    # Token Saving计算
    vector_context_tokens = 3 * 500
    lora_context_tokens = 1 * 500
    token_saving = (vector_context_tokens - lora_context_tokens) / vector_context_tokens

    # 打印报告
    print("\n" + "=" * 60)
    print("实验结果报告")
    print("=" * 60)
    print(f"评测样本数: {n}")
    print(f"\n【Hit Rate 命中率】")
    print(f"  Vector RAG:  {vector_hit_rate:.2%} ({vector_hits}/{n})")
    print(f"  LoRA Router: {lora_hit_rate:.2%} ({lora_hits}/{n})")
    print(f"\n【Latency 平均延迟】")
    print(f"  Vector RAG:  {avg_vector_latency:.3f}s")
    print(f"  LoRA Router: {avg_lora_latency:.3f}s")
    print(f"\n【Token Saving 上下文压缩率】")
    print(f"  Vector返回: {vector_context_tokens} tokens (Top-3 chunks)")
    print(f"  LoRA返回:   {lora_context_tokens} tokens (1精准chunk)")
    print(f"  节省率:     {token_saving:.1%}")

    # 保存详细结果
    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_samples": n,
                "vector_hit_rate": vector_hit_rate,
                "lora_hit_rate": lora_hit_rate,
                "avg_vector_latency": avg_vector_latency,
                "avg_lora_latency": avg_lora_latency,
                "token_saving": token_saving
            },
            "detailed_results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: outputs/evaluation_results.json")

    return results


if __name__ == "__main__":
    run_experiment()
