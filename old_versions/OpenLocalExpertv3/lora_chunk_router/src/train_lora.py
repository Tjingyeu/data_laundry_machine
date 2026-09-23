"""
Train LoRA model for chunk classification using Sequence Classification approach.
Uses Qwen2.5-0.5B with PEFT LoRA and sequence classification head.
"""

import os
import sys
import torch
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_project_root, load_json

# =========================
# 配置
# =========================
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 128


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train LoRA model for Classification")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    root = get_project_root()
    config_path = Path(args.config) if args.config else root / "configs" / "config.yaml"
    config = load_config(str(config_path))

    train_path = root / config["data"]["train_output"]
    output_dir = root / config["training"]["output_dir"]
    label_path = root / config["data"]["id_to_label_output"]

    print("=" * 60)
    print("LoRA Chunk Router Training (Sequence Classification)")
    print("=" * 60)

    # 1. Load Label Mappings
    print("\n🏷️ Loading label mappings...")
    label_data = load_json(str(label_path))
    label_to_id = {k: int(v) for k, v in label_data["label_to_id"].items()}
    id_to_label = {int(k): v for k, v in label_data["id_to_label"].items()}
    num_labels = len(label_to_id)
    print(f"   Found {num_labels} classes.")

    # 2. Load Tokenizer
    print("\n🚀 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Process Dataset
    print(f"📚 Loading dataset from: {train_path}")
    dataset = load_dataset("json", data_files=str(train_path))["train"]

    def preprocess_function(examples):
        inputs = examples["instruction"]
        labels = [label_to_id[out] for out in examples["output"]]
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
        model_inputs["labels"] = labels
        return model_inputs

    print("✂️ Tokenizing and mapping labels...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # 4. Load Model for Sequence Classification
    print("\n🧩 Loading model...")
    device_map = "auto" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device_map}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
        device_map=device_map,
        torch_dtype=torch.float32,  # Use float32 for stability
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 5. Apply LoRA
    print("\n🔧 Applying LoRA...")
    lora_config = LoraConfig(
        r=8,  # Smaller rank for stability
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Training Arguments - STABILITY FIXES
    print("\n⚙️ Training config (stability-optimized)...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,  # Smaller batch
        gradient_accumulation_steps=4,  # Effective batch = 16
        learning_rate=1e-4,  # Lower LR for score head
        num_train_epochs=10,  # More epochs
        max_grad_norm=1.0,  # Gradient clipping
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        eval_strategy="no",
        fp16=False,  # No FP16 on MPS
        bf16=False,
        torch_empty_cache_steps=50,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7. Train
    print("\n🏋️ Start training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # 8. Save
    print("\n💾 Saving model...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    import shutil
    shutil.copy(str(label_path), str(output_dir / "id_to_label.json"))

    print("\n✅ Done!")

if __name__ == "__main__":
    main()