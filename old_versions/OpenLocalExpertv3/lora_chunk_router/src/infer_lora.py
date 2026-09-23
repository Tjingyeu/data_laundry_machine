"""
LoRA Inference using Sequence Classification logits approach.
Returns predictions in format: [{"id": "CH_0123", "score": 0.85}, ...]
"""

import os
import sys
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, load_json, get_project_root

def load_lora_model(
    lora_dir: str,
    base_model_name: str = "Qwen/Qwen2.5-0.5B",
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, Dict[str, int], Dict[int, str]]:
    """Load fine-tuned LoRA Sequence Classification model."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading model from: {lora_dir}")
    print(f"Using device: {device}")

    # Load label mapping
    label_path = os.path.join(lora_dir, "id_to_label.json")
    if os.path.exists(label_path):
        label_data = load_json(label_path)
        label_to_id = {k: int(v) for k, v in label_data["label_to_id"].items()}
        id_to_label = {int(k): v for k, v in label_data["id_to_label"].items()}
    else:
        raise FileNotFoundError(f"Label mapping not found at {label_path}")

    tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Base Model as Sequence Classifier
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.float16,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()

    print(f"Loaded {len(label_to_id)} labels")
    return model, tokenizer, label_to_id, id_to_label


def predict_topk(
    query: str,
    model,
    tokenizer,
    id_to_label: Dict[int, str],
    top_k: int = 5,
    relevance_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Predict top-K chunks with real Softmax confidence scores.
    """
    # No prompt template needed for sequence classification, just the query
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0] # Extract probabilities for batch size 1

    # Get Top-K probabilities and indices
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(id_to_label)))

    results = []
    for prob, idx in zip(top_probs, top_indices):
        score = prob.item()
        label = id_to_label[idx.item()]

        # Stop collecting if we fall below threshold
        if score < relevance_threshold:
            break

        results.append({"id": label, "score": score})

    # If all predictions were below threshold, or model predicted nothing
    if not results:
        return [{"id": "NOT_RELEVANT", "score": float(top_probs[0].item())}]

    return results


def batch_predict(
    questions: List[str],
    model,
    tokenizer,
    id_to_label: Dict[int, str],
    top_k: int = 5,
    relevance_threshold: float = 0.3,
) -> List[List[Dict[str, Any]]]:
    """Batch prediction using a simple loop for evaluation."""
    results = []
    for question in questions:
        preds = predict_topk(
            query=question,
            model=model,
            tokenizer=tokenizer,
            id_to_label=id_to_label,
            top_k=top_k,
            relevance_threshold=relevance_threshold
        )
        results.append(preds)
    return results


def main():
    """CLI for testing."""
    import argparse
    parser = argparse.ArgumentParser(description="LoRA Inference")
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    root = get_project_root()
    config = load_config(str(root / "configs" / "config.yaml"))
    lora_dir = args.lora_dir if args.lora_dir else str(root / config["inference"]["lora_dir"])

    print("LoRA Inference - Enter a question (or 'quit' to exit)")
    model, tokenizer, label_to_id, id_to_label = load_lora_model(lora_dir, config["model"]["base_model"])

    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() == 'quit':
            break

        results = predict_topk(query, model, tokenizer, id_to_label, args.top_k, args.threshold)
        print("Predictions:")
        for res in results:
            print(f"  {res['id']} | {res['score']:.4f}")

if __name__ == "__main__":
    main()