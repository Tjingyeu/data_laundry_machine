"""
阶段4: LoRA 训练 (Unsloth)
使用 Unsloth 训练 LoRA adapter
配置: rank=16, lr=2e-4, steps=60
"""

import os
import json
from unsloth import FastLanguageModel
import torch


# 训练配置
MODEL_NAME = "qwen3.5:4b"  # Ollama中的模型名（但Unsloth需要本地模型文件或HuggingFace模型）
LORA_RANK = 16
LORA_ALPHA = 16
LEARNING_RATE = 2e-4
STEPS = 60
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4


def load_train_data(data_path="./data/train.json"):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_train_sample(sample):
    """将样本格式化为Alpaca格式"""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")

    # 构建prompt
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""

    return prompt


def main():
    print("=" * 60)
    print("LoRA 训练脚本")
    print("=" * 60)

    # 加载训练数据
    train_data = load_train_data()
    print(f"加载了 {len(train_data)} 条训练数据")

    # 加载模型
    print(f"\n正在加载模型 {MODEL_NAME}...")
    print("注意: Unsloth需要本地模型或HuggingFace模型")
    print("如果Ollama中有模型，需要先导出为GGUF格式")

    # 检查是否有本地模型
    model_path = "./models/qwen3.5-4b"

    if os.path.exists(model_path):
        print(f"找到本地模型: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        # 尝试从HuggingFace下载
        print(f"未找到本地模型，尝试从HuggingFace下载...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen2.5-4B-Instruct",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

    # 添加LoRA adapter
    print(f"\n添加LoRA adapter (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing=True,
    )

    # 格式化数据
    print("\n格式化训练数据...")
    train_texts = [format_train_sample(s) for s in train_data]

    # 创建数据集
    from unsloth.trainer import ConstantLengthDataset

    def generate_train_ds():
        for text in train_texts:
            yield {"text": text}

    train_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=generate_train_ds(),
        dataset_text_field="text",
        max_seq_length=512,
    )

    # 训练
    print(f"\n开始训练 (steps={STEPS}, lr={LEARNING_RATE})...")

    from unsloth import UnslothTrainer
    from unsloth.trainer import ConstantLengthDataset

    trainer = UnslothTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args={
            "per_device_train_batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "warmup_steps": max(1, STEPS // 10),
            "num_train_epochs": 1,
            "learning_rate": LEARNING_RATE,
            "fp16": not torch.cuda.is_available(),
            "logging_steps": max(1, STEPS // 10),
            "output_dir": "./outputs/expert_lora_adapter",
            "optim": "adamw_8bit",
            "save_steps": STEPS,
            "max_steps": STEPS,
        }
    )

    trainer.train()

    # 保存adapter
    print("\n保存LoRA adapter...")
    model.save_pretrained("./outputs/expert_lora_adapter")
    print("训练完成!")
    print(f"  输出目录: ./outputs/expert_lora_adapter")


if __name__ == "__main__":
    main()