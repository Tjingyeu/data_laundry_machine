"""
Evaluation script comparing LoRA router vs Vector search.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, load_json, get_project_root
from infer_lora import load_lora_model, batch_predict
from vector_search import VectorSearchIndex


def compute_accuracy_at_k(predictions: List[List[Dict[str, any]]], ground_truths: List[str], k_values: List[int] = [1, 3]) -> Dict[int, float]:
    results = {}
    for k in k_values:
        hits = 0
        for preds, truth in zip(predictions, ground_truths):
            if not preds:
                continue
            # Check if truth is in top-K predictions
            top_k_preds = [p["id"] for p in preds[:k]]
            if truth in top_k_preds:
                hits += 1
        results[k] = hits / len(ground_truths) if ground_truths else 0.0
    return results

def compute_mrr(predictions: List[List[Dict[str, any]]], ground_truths: List[str]) -> float:
    reciprocal_ranks = []
    for preds, truth in zip(predictions, ground_truths):
        for rank, pred in enumerate(preds, 1):
            if pred["id"] == truth:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def compute_fuzzy_accuracy(predictions: List[List[Dict[str, any]]], ground_truths: List[str], query_types: List[str]) -> float:
    fuzzy_preds, fuzzy_truths = [], []
    for preds, truth, qtype in zip(predictions, ground_truths, query_types):
        if qtype in ["fuzzy", "metaphor", "colloquial"]:
            fuzzy_preds.append(preds)
            fuzzy_truths.append(truth)
    if not fuzzy_preds:
        return 0.0
    hits = 0
    for preds, truth in zip(fuzzy_preds, fuzzy_truths):
        if preds and preds[0]["id"] == truth:
            hits += 1
    return hits / len(fuzzy_truths)

def compute_not_relevant_accuracy(predictions: List[List[Dict[str, any]]], ground_truths: List[str]) -> float:
    not_relevant_indices = [i for i, t in enumerate(ground_truths) if t == "NOT_RELEVANT"]
    if not not_relevant_indices:
        return 0.0
    hits = 0
    for idx in not_relevant_indices:
        preds = predictions[idx]
        if preds and preds[0]["id"] == "NOT_RELEVANT":
            hits += 1
    return hits / len(not_relevant_indices)

def compute_avg_chunks_returned(predictions: List[List[Dict[str, any]]]) -> float:
    if not predictions:
        return 0.0
    total_chunks = sum(len(p) for p in predictions)
    return total_chunks / len(predictions)

def evaluate_lora(test_data: List[Dict], lora_dir: str, base_model_name: str, top_k: int = 5, relevance_threshold: float = 0.3) -> Dict[str, Any]:
    print("Loading LoRA model...")
    model, tokenizer, label_to_id, id_to_label = load_lora_model(lora_dir, base_model_name)

    queries = [item["query"] for item in test_data]
    ground_truths = [item["label"] for item in test_data]
    query_types = [item.get("type", "unknown") for item in test_data]

    print(f"Running inference on {len(queries)} test samples...")
    predictions = batch_predict(
        questions=queries,
        model=model,
        tokenizer=tokenizer,
        id_to_label=id_to_label,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
    )

    metrics = {}
    for k in [1, 3, 5]:
        k = min(k, top_k)
        acc = compute_accuracy_at_k(predictions, ground_truths, [k])[k]
        metrics[f"Top{k}"] = round(acc, 4)

    metrics["MRR"] = round(compute_mrr(predictions, ground_truths), 4)
    metrics["Fuzzy"] = round(compute_fuzzy_accuracy(predictions, ground_truths, query_types), 4)
    metrics["NOT_RELEVANT"] = round(compute_not_relevant_accuracy(predictions, ground_truths), 4)
    metrics["AvgChunks"] = round(compute_avg_chunks_returned(predictions), 2)
    return metrics

def evaluate_vector(test_data: List[Dict], index_dir: str, top_k: int = 5) -> Dict[str, Any]:
    print("Loading Vector search index...")
    index = VectorSearchIndex(index_path=index_dir)
    index.load(index_dir)

    queries = [item["query"] for item in test_data]
    ground_truths = [item["label"] for item in test_data]
    query_types = [item.get("type", "unknown") for item in test_data]

    print(f"Running search on {len(queries)} test samples...")
    predictions = [index.search(query=q, top_k=top_k) for q in queries]

    metrics = {}
    for k in [1, 3, 5]:
        k = min(k, top_k)
        acc = compute_accuracy_at_k(predictions, ground_truths, [k])[k]
        metrics[f"Top{k}"] = round(acc, 4)

    metrics["MRR"] = round(compute_mrr(predictions, ground_truths), 4)
    metrics["Fuzzy"] = round(compute_fuzzy_accuracy(predictions, ground_truths, query_types), 4)
    metrics["NOT_RELEVANT"] = round(compute_not_relevant_accuracy(predictions, ground_truths), 4)
    metrics["AvgChunks"] = round(compute_avg_chunks_returned(predictions), 2)
    return metrics

def print_comparison(lora_metrics: Dict, vector_metrics: Dict) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS: LoRA Router vs Vector Search")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'LoRA':>15} {'Vector':>15} {'Winner':>15}")
    print("-" * 65)

    metrics_order = ["Top1", "Top3", "Top5", "MRR", "Fuzzy", "NOT_RELEVANT", "AvgChunks"]
    for metric in metrics_order:
        lora_val = lora_metrics.get(metric, 0)
        vector_val = vector_metrics.get(metric, 0)

        winner = "Tie"
        if metric == "AvgChunks":
            if lora_val < vector_val: winner = "LoRA"
            elif vector_val < lora_val: winner = "Vector"
        else:
            if lora_val > vector_val: winner = "LoRA"
            elif vector_val > lora_val: winner = "Vector"

        print(f"{metric:<20} {lora_val:>15.4f} {vector_val:>15.4f} {winner:>15}")
    print("-" * 65)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LoRA vs Vector")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    root = get_project_root()
    config_path = Path(args.config) if args.config else root / "configs" / "config.yaml"
    config = load_config(str(config_path))

    test_path = root / config["data"]["test_output"]
    test_data = load_json(str(test_path))
    if isinstance(test_data, dict) and "test_samples" in test_data:
        test_data = test_data["test_samples"]

    print("\n" + "=" * 60 + "\nEvaluating LoRA Router...\n" + "=" * 60)
    lora_metrics = evaluate_lora(
        test_data=test_data,
        lora_dir=str(root / config["inference"]["lora_dir"]),
        base_model_name=config["model"]["base_model"],
        top_k=config["inference"]["top_k"],
        relevance_threshold=config["inference"]["relevance_threshold"],
    )

    print("\n" + "=" * 60 + "\nEvaluating Vector Search...\n" + "=" * 60)
    vector_metrics = evaluate_vector(
        test_data=test_data,
        index_dir=str(root / "models" / "embeddings"),
        top_k=config["inference"]["top_k"],
    )

    print_comparison(lora_metrics, vector_metrics)

if __name__ == "__main__":
    main()