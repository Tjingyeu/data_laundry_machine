# train_lora_m1.py
# M1 Pro 苹果芯片 LoRA 训练脚本
# 优化版：使用mlx（苹果官方加速）或量化模型

import os
import sys
import json
import torch

# 检查设备
def get_device():
    """检测可用设备"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

print(f"🔍 检测设备: {get_device()}")
print(f"💻 Apple Silicon MPS: {torch.backends.mps.is_available()}")
print(f"🔢 PyTorch版本: {torch.__version__}")

# ============================================================
# 方案选择 (根据你的硬件选择)
# ============================================================

TRAINING_MODE = input("""
选择训练方案:
1. MLX方案 (推荐苹果芯片 - 使用mlx库)
2. 量化方案 (使用bitsandbytes 4bit量化)
3. CPU方案 (保守但稳定)

请输入选项 [1-3]: ").strip() or "1"

# ============================================================
# 方案1: MLX (苹果官方加速)
# ============================================================

if TRAINING_MODE == "1":
    print("""
============================================================
🚀 方案1: MLX 苹果芯片加速
============================================================
    """)

    # 检查是否安装mlx
    try:
        import mlx.core as mx
        print(f"✅ MLX已安装，版本: {mx.__version__}")
        HAS_MLX = True
    except ImportError:
        print("❌ MLX未安装")
        print("安装命令: pip install mlx")
        print("\n或者使用方案2/3")
        HAS_MLX = False
        sys.exit(1)

    # 检查训练数据
    if not os.path.exists("train.jsonl"):
        print("❌ 训练数据不存在!")
        print("请先运行: python build_lora_dataset.py")
        sys.exit(1)

    print("✅ 准备就绪")
    print("""
MLX训练说明:
- MLX是苹果官方为Apple Silicon优化的ML框架
- 直接使用Metal GPU加速
- 16GB内存可训练Qwen2.5-3B (建议使用较小模型)

下一步手动执行:
    import mlx.core as mx
    from mlx_lm import generate, load
    # 参考: https://github.com/ml-explore/mlx-examples
    """)

# ============================================================
# 方案2: 量化方案 (bitsandbytes)
# ============================================================

elif TRAINING_MODE == "2":
    print("""
============================================================
🚀 方案2: 4bit量化 + LoRA训练
============================================================
    """)

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import bitsandbytes as bnb

    # 配置
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # 小模型适合16GB
    OUTPUT_DIR = "lora_output"
    TRAINING_STEPS = 100
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 4

    print(f"📦 加载模型: {MODEL_NAME}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 4bit量化配置
    print("🔄 加载4bit量化模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        device_map="auto"
    )

    # 准备训练
    model = prepare_model_for_kbit_training(model)

    # LoRA配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据
    print("📂 加载训练数据...")
    dataset = load_dataset("json", data_files="train.jsonl", split="train")

    def tokenize(examples):
        # 格式化instruction tuning数据
        texts = [
            f"{inst}\n\n{inp}\n\n{out}"
            for inst, inp, out in zip(
                examples["instruction"],
                examples["input"],
                examples["output"]
            )
        ]
        return tokenizer(texts, truncation=True, max_length=256, padding="max_length")

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=2e-4,
        fp16=False,  # MPS不支持fp16
        logging_steps=10,
        save_steps=50,
        max_steps=TRAINING_STEPS,
        report_to="none",
        save_total_limit=2,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 训练
    print(f"🚀 开始训练 ({TRAINING_STEPS} steps)...")
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # 保存
    print(f"💾 保存LoRA到 {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ 训练完成!")

# ============================================================
# 方案3: CPU保守方案
# ============================================================

else:
    print("""
============================================================
🚀 方案3: CPU训练 (保守但稳定)
============================================================
    """)

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    OUTPUT_DIR = "lora_output_cpu"

    print(f"📦 加载模型: {MODEL_NAME} (CPU模式)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据
    dataset = load_dataset("json", data_files="train.jsonl", split="train")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=20,
        max_steps=50,
        fp16=False,
    )

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("🚀 开始训练...")
    trainer.train()

    print(f"💾 保存到 {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)

    print("✅ 完成!")
