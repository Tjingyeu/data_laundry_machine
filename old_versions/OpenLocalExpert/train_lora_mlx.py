# train_lora_mlx.py
# MLX 苹果芯片 LoRA 训练脚本

import os
import sys

# 设置日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mlx.core as mx
from mlx_lm import load, generate, train
from mlx_lm.utils import generate_lora_weights, save_lora

print("""
============================================================
🚀 MLX LoRA 训练 (Apple Silicon)
============================================================
""")

# 检查训练数据
if not os.path.exists("train.jsonl"):
    print("❌ 训练数据不存在!")
    print("请先运行: python build_lora_dataset.py")
    sys.exit(1)

# 加载模型
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"📦 加载模型: {MODEL_NAME}")

model, tokenizer = load(MODEL_NAME)
print(f"✅ 模型加载完成")
print(f"   设备: {mx.default_device()}")

# LoRA配置
LORA_OUTPUT = "lora_output_mlx"
os.makedirs(LORA_OUTPUT, exist_ok=True)

# 训练参数
TRAIN_STEPS = 100  # 先测试100步
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

print(f"""
============================================================
⚙️ 训练配置
============================================================
模型: {MODEL_NAME}
LoRA输出: {LORA_OUTPUT}
训练步数: {TRAIN_STEPS}
批次大小: {BATCH_SIZE}
学习率: {LEARNING_RATE}
============================================================
""")

# 检查mlx_lm.train是否可用
try:
    from mlx_lm import train as mlx_train
    print("✅ mlx_lm.train 可用")
except ImportError as e:
    print(f"❌ mlx_lm.train 不可用: {e}")
    print("""
替代方案: 使用标准MLX训练循环
""")
    mlx_train = None

if mlx_train:
    # 使用mlx_lm.train训练
    print(f"🚀 开始训练 ({TRAIN_STEPS} steps)...")

    # 训练数据格式转换 (需要适配mlx_lm格式)
    # mlx_lm.train需要一个特定的数据集格式

    print("""
训练说明:
mlx_lm.train 需要特定的训练数据格式。

当前train.jsonl格式:
{"instruction": "...", "input": "...", "output": "..."}

mlx_lm.train需要:
{"text": "..."} 格式

建议手动执行:

1. 转换数据格式:
   python -c "
   import json
   with open('train.jsonl') as f:
       lines = f.readlines()
   with open('train_mlx.jsonl', 'w') as out:
       for line in lines:
           d = json.loads(line)
           text = f\"{d['instruction']}\\n\\n{d['input']}\\n\\n{d['output']}\"
           out.write(json.dumps({'text': text}) + '\\n')
   "

2. 运行训练:
   mlx_lm.lora --model Qwen/Qwen2.5-0.5B-Instruct \\
               --train ./train_mlx.jsonl \\
               --steps 100 \\
               --save_dir ./lora_output_mlx
""")
else:
    print("需要使用mlx_lm CLI工具进行训练")
    print(f"""
命令:
mlx_lm.lora --model Qwen/Qwen2.5-0.5B-Instruct \\
            --train ./train.jsonl \\
            --steps {TRAIN_STEPS} \\
            --save_dir ./{LORA_OUTPUT}
""")
