# Data Laundry Machine

AI 驱动的数据清洗 Agent + LoRA 训练数据准备管道，同一仓库双子系统。

## 项目功能

本项目由两套独立但可组合的子系统构成：

1. **Data Cleaning Agent** —— 让大模型（OpenAI / Anthropic）以 tool-calling 的方式自主分析数据样本、决定清洗动作、迭代执行至收敛或 AI 主动收手。
2. **LoRA Pipeline** —— 把一堆原始长文本切成有重叠的 chunk，本地小模型先"脱水"成骨架，云端 API 再从中抽因果链 / 概念 / 前置知识，最后跨 chunk 语义去重焊接。`chunk_id` 全程贯穿，是数据血缘的身份证。

两者通过同一套 `BaseAIProvider` / `PromptManager` / 配置系统共享底层能力。

## 仓库结构

```
data_laundry_machine/
├── config.yaml              AI provider / agent / 日志配置
├── requirements.txt
├── src/
│   ├── main.py              入口
│   ├── cli/commands.py      Click CLI
│   ├── agent/               数据清洗 Agent 核心
│   │   ├── base_agent.py            BaseAgent 抽象 + CleaningResult / AIMessage
│   │   ├── data_cleaning_agent.py   AI tool-calling 循环
│   │   ├── agent_executor.py        注册/调度/批量执行清洗操作
│   │   └── cleaning_operation.py    CleaningOperation / OperationStatus
│   ├── operations/          清洗操作实现
│   │   ├── missing_value_handler.py drop / mean / median / mode / constant / interpolate / ffill / bfill
│   │   ├── deduplication.py         去重（按列 + keep 策略）
│   │   ├── type_converter.py        int/float/string/datetime/boolean/category
│   │   └── base_operation.py
│   ├── data_source/         CSV / JSON 适配器
│   ├── ai_providers/        OpenAI / Anthropic 抽象与实现
│   ├── prompts/             Jinja2 模板管理（system / analysis_request / operation_result）
│   ├── workflow/            CleaningWorkflow + IterationController
│   ├── config/settings.py   Settings 加载 / 环境变量解析
│   └── lora_pipeline/       LoRA 训练数据准备管道
│       ├── lora_pipeline.py         主管道：chunking → dehydration → extraction → aggregation
│       ├── smart_chunker.py         智能切片（tiktoken + 滑窗）
│       ├── dehydration.py           本地模型文本脱水
│       ├── logic_extraction.py      云端 API 提取因果链
│       ├── logic_aggregator.py      跨 chunk 逻辑焊接 + 语义去重
│       └── prompts/                 各阶段 prompt 模板
├── txt/                     原始文本输入（默认放 txt 文件）
├── output_test/             输出样例
│   ├── dehydrated/          脱水后的 chunk
│   ├── logic_extraction.jsonl     提取的逻辑链
│   └── logic_extraction_manifest.json
└── tests/                   测试
```

## 安装

```bash
git clone <repo-url> data_laundry_machine
cd data_laundry_machine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

环境变量（至少设一个 provider 的 key）：

```bash
export OPENAI_API_KEY="sk-..."
# 或
export ANTHROPIC_API_KEY="ant-..."
```

## 快速开始

### 初始化一个新项目（生成 `data/raw/`、`data/cleaned/`、默认 `config.yaml`）

```bash
python -m src.main init
```

### 数据质量分析

```bash
python -m src.main analyze data/raw/input.csv
python -m src.main analyze data/raw/input.csv --format json
```

### AI 自动清洗

```bash
python -m src.main clean data/raw/input.csv data/cleaned/output.csv \
  --provider openai \
  --model gpt-4o \
  --max-iterations 5 \
  --sample-size 100 \
  --verbose
```

`--provider` 支持 `openai` / `anthropic`，`--api-key` 可显式传入，留空时从环境变量读。

### 列出可用 Provider

```bash
python -m src.main providers
```

## Data Cleaning Agent 工作流

```
CSV / JSON
   │
   ▼
Adapter.read()        ── 读取为 DataFrame
   │
   ▼
Agent._sample_data()  ── 按 sample_size 随机采样
   │
   ▼
┌─── 迭代循环（最多 max_iterations 轮）──────────────────┐
│                                                         │
│  1. 构建 prompt：schema + 样本 + 最近 6 条对话历史       │
│  2. 调用 AI，带 tool 列表：                             │
│       - handle_missing_values                          │
│       - deduplicate                                    │
│       - convert_type                                   │
│       - finish_cleaning                                │
│  3. 解析 tool_calls → CleaningOperation                 │
│  4. AgentExecutor 顺序执行，更新样本                    │
│  5. 把执行结果作为 user 消息回灌到对话历史              │
│  6. 收敛判断：quality_score ≥ convergence_threshold    │
│     或 AI 调用 finish_cleaning → 退出                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
   │
   ▼
输出 CSV / JSON + 质量报告 + 操作日志
```

注册自定义操作：

```python
from src.agent import DataCleaningAgent
agent = DataCleaningAgent(ai_provider=..., prompt_manager=..., config=...)
agent.register_operation("my_op", MyCustomOperation)
```

## LoRA Pipeline 工作流

四阶段主管道（`src/lora_pipeline/lora_pipeline.py` 的 `LoRAPipeline.run()`）：

| 阶段 | 模块 | 干什么 | 默认配置 |
|------|------|--------|----------|
| 1. 智能切片 | `smart_chunker.py` | tiktoken 滑窗切片，找最近句子终止符 | chunk 30k 字符，overlap 50% |
| 2. 文本脱水 | `dehydration.py` | 本地小模型压缩为骨架文本 | `qcwind/qwen2.5-7B-instruct-Q4_K_M` |
| 3. 因果链提取 | `logic_extraction.py` | 云端 API 抽 concepts / prerequisites / impact，缺前置则标 `[待补全]` | — |
| 4. 逻辑聚合 | `logic_aggregator.py` | sentence-transformers 语义去重 + 跨 chunk 焊接 | 相似度阈值 0.9 |

`chunk_id` 贯穿四阶段，必要时可反向追溯回原始文本。

直接调用：

```python
from src.lora_pipeline import LoRAPipeline, PipelineConfig

pipeline = LoRAPipeline(
    ai_provider=my_ai_provider,  # 不传则走 mock 路径
    config=PipelineConfig(
        output_dir="./output",
        dehydration_enabled=True,
        extraction_enabled=True,
        similarity_threshold=0.9,
    ),
)

result = pipeline.run(
    input_path="./txt",
    is_directory=True,
    pattern="*.txt",
)
```

输出文件：

- `lora_chunking.jsonl` —— 切片结果
- `lora_dehydration.jsonl` —— 脱水结果
- `lora_extraction.jsonl` —— 因果链提取
- `lora_aggregation.jsonl` —— 最终聚合

## 配置

`config.yaml` 顶层结构：

```yaml
ai:
  provider: "openai"          # openai / anthropic
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o"
  temperature: 0.1
  max_tokens: 4096

agent:
  max_iterations: 5
  convergence_threshold: 0.95
  sample_size: 100

data:
  input_path: "data/raw/input.csv"
  output_path: "data/cleaned/output.csv"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

`${VAR}` 形式的值会自动从环境变量解析。

LoRA 管道参数集中在 `lora_pipeline.PipelineConfig`（`src/lora_pipeline/lora_pipeline.py`）：

```python
@dataclass
class PipelineConfig:
    context_window: int = 32768
    chunk_ratio: float = 0.30
    overlap_ratio: float = 0.50
    dehydration_enabled: bool = True
    extraction_enabled: bool = True
    similarity_threshold: float = 0.9
    output_dir: str = "./output"
    verbose: bool = False
```

## 已验证的运行记录

`output_test/logic_extraction_manifest.json` 记录了一次真实 LoRA 管道跑通：

```
processor: logic_extraction
model: MiniMax-M2.7
api_url: https://api.minimaxi.com/anthropic/v1/messages
parallel_workers: 4
stats: { total: 389, processed: 389, failed: 0, total_concepts_extracted: 2559 }
```

说明：仓库使用的 anthropic provider 可以指向第三方 Anthropic 兼容端点（如 `api.minimaxi.com`），在 `AnthropicProvider.__init__` 之后自行注入 `client.base_url` 即可（base class 没暴露该参数，需要在调用方设置）。

## 当前已知问题 / 局限

- **Dehydration 串行**：`_run_dehydration` 用 `tqdm` 串行处理 chunk，大语料下耗时可观；可改为 `ProcessPoolExecutor` 或 `asyncio.gather`。
- **Dehydration 模型未在 requirements 锁定**：默认 `qcwind/qwen2.5-7B-instruct-Q4_K_M` 通过 llama.cpp 之类调用，需要本机有推理服务。
- **Anthropic provider 的 base_url 不可配**：想用第三方 Anthropic 兼容端点需要在外部改 `provider.client.base_url`。
- **清洗输出 vs 采样数据**：`clean` 命令保存的是 `agent.data_sample`（采样过的），不是 `full_data`；如要保存全量需要扩展。
- **tests/ 为空**：核心逻辑未覆盖单元测试。
- **没有 `.gitignore`**：建议至少忽略 `.venv/`、`__pycache__/`、`*.pyc`、`.env`、`output/`。

## 路线图

- 把 Dehydration 改为并发
- 给 AnthropicProvider 加 `base_url` 注入
- 加单元测试（至少覆盖 operations 和 chunker）
- 输出侧区分 `data_sample` vs `full_data`
- 写一个一键 `run_pipeline.py` 脚本（封装 CLI 之外的 LoRA 调用）

## 许可

待定。
