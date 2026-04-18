"""
Program 3: 逻辑概念提取器 (Logic Extraction Module)
从脱水后的文本中提取因果逻辑链和知识概念单元。

特点：
1. 调用云端大模型 API（OpenAI / Claude 等兼容接口）
2. 并发处理：ThreadPoolExecutor 提升云端 I/O 效率
3. 指数退避重试：处理 API 超时和 Rate Limit
4. 每条记录携带原始 chunk_id，输出为 .jsonl
"""

import json
import time
import httpx
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================
# 配置类
# ============================================================

@dataclass
class LogicExtractionConfig:
    """逻辑提取配置"""
    # API Provider: "openai" | "anthropic"
    # openai:      POST <base_url>/v1/chat/completions
    # anthropic:   POST <base_url>/v1/messages  (MiniMax / Claude 兼容)
    provider: str = "openai"

    # API 基础地址（不包含具体路径）
    base_url: str = "https://api.openai.com"

    # API Key
    api_key: str = ""

    # 模型名称
    model: str = "gpt-4o"

    # 生成参数
    temperature: float = 0.3

    # Anthropic API 专用：最大输出 token 数
    max_tokens: int = 4096

    # API 请求超时（秒）
    timeout: int = 120

    # 最大重试次数
    retry_count: int = 20

    # 重试初始间隔（秒）
    retry_delay: float = 10.0

    # 重试最大间隔（秒）
    max_retry_delay: float = 120.0

    # 并行工作线程数
    parallel_workers: int = 4

    # System Prompt 文件路径（可选）
    system_prompt_file: Optional[str] = None

    @property
    def api_url(self) -> str:
        """根据 provider 推导具体端点路径"""
        if self.provider == "anthropic":
            return f"{self.base_url.rstrip('/')}/v1/messages"
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    @property
    def headers(self) -> Dict[str, str]:
        """构建请求头"""
        headers = {"Content-Type": "application/json"}
        if self.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


# ============================================================
# 概念单元数据类
# ============================================================

@dataclass
class ConceptUnit:
    """单个逻辑概念单元"""
    concept: str                      # 概念名称
    definition: str                   # 深度解析，保留原始讲稿风格
    prerequisites: List[str] = field(default_factory=list)   # 前置逻辑概念
    impact: str = ""                   # 概念导致的后果或延伸逻辑

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept,
            "definition": self.definition,
            "prerequisites": self.prerequisites,
            "impact": self.impact,
        }


# ============================================================
# 云端大模型 API 客户端（支持指数退避重试）
# ============================================================

class CloudModelClient:
    """
    云端大模型 API 客户端
    支持 OpenAI / Claude 等兼容 /v1/chat/completions 接口
    内置指数退避重试机制
    """

    def __init__(self, config: LogicExtractionConfig):
        self.config = config
        self.client = httpx.Client(timeout=config.timeout)

    def _load_system_prompt(self) -> str:
        """加载 System Prompt（从文件或内置默认）"""
        if self.config.system_prompt_file:
            prompt_path = Path(self.config.system_prompt_file)
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return f.read()
        return DEFAULT_EXTRACTION_PROMPT

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        发送对话请求到云端模型，带指数退避重试

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
                      Anthropic 模式下 role 可为 "user" | "assistant"
            model: 模型名称（可选）
            temperature: 温度参数（可选）

        Returns:
            模型的回复内容（文本）
        """
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature

        retry_delay = self.config.retry_delay

        for attempt in range(self.config.retry_count):
            try:
                if self.config.provider == "anthropic":
                    payload = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": self.config.max_tokens,
                        "temperature": temperature,
                    }
                else:
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                    }

                response = self.client.post(
                    self.config.api_url,
                    json=payload,
                    headers=self.config.headers,
                )

                if response.status_code == 429:
                    # Rate Limit
                    logger.warning(
                        f"[Attempt {attempt + 1}] Rate limited. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    if attempt < self.config.retry_count - 1:
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, self.config.max_retry_delay)
                    continue

                if response.status_code >= 500:
                    # Server-side error
                    logger.warning(
                        f"[Attempt {attempt + 1}] Server error {response.status_code}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    if attempt < self.config.retry_count - 1:
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, self.config.max_retry_delay)
                    continue

                response.raise_for_status()
                result = response.json()

                # 解析不同 provider 的响应格式
                if self.config.provider == "anthropic":
                    # Anthropic: {"content": [{"type": "text", "text": "..."}]}
                    # MiniMax 可能同时返回 thinking + text，取 text 类型
                    for block in result["content"]:
                        if block.get("type") == "text":
                            return block["text"]
                    # 兜底：返回 thinking 内容的前 500 字符
                    for block in result["content"]:
                        if block.get("type") == "thinking":
                            return "[thinking]\n" + block.get("thinking", "")[:500]
                    raise ValueError(f"No text block in Anthropic response: {result}")
                else:
                    # OpenAI: {"choices": [{"message": {"content": "..."}}]}
                    return result["choices"][0]["message"]["content"]

            except httpx.TimeoutException:
                logger.warning(
                    f"[Attempt {attempt + 1}] Request timed out. "
                    f"Retrying in {retry_delay:.1f}s..."
                )
                if attempt < self.config.retry_count - 1:
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.config.max_retry_delay)
                else:
                    raise

            except httpx.HTTPError as e:
                logger.warning(f"[Attempt {attempt + 1}] HTTP error: {e}")
                if attempt < self.config.retry_count - 1:
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.config.max_retry_delay)
                else:
                    raise

        return ""

    def chat_with_fresh_context(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        使用全新上下文发送消息

        Args:
            user_message: 用户消息
            system_prompt: System Prompt（可选）

        Returns:
            模型回复文本
        """
        messages = []

        if system_prompt:
            if self.config.provider == "anthropic":
                # Anthropic 不支持单独的 system role，把 system 拆入 messages
                messages.append({"role": "user", "content": f"[System Prompt]\n{system_prompt}"})
            else:
                messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_message})

        return self.chat(messages)


# ============================================================
# 逻辑提取器
# ============================================================

class LogicExtractor:
    """
    逻辑概念提取器
    读取脱水后的 JSON 文件，调用云端大模型提取因果逻辑链，
    输出为 .jsonl 格式的概念单元列表
    """

    def __init__(
        self,
        config: Optional[LogicExtractionConfig] = None,
        model_client: Optional[CloudModelClient] = None,
    ):
        self.config = config or LogicExtractionConfig()
        self.model_client = model_client or CloudModelClient(self.config)

        # 加载 System Prompt
        self.system_prompt = self.model_client._load_system_prompt()

    def build_user_message(self, chunk_data: Dict[str, Any]) -> str:
        """构建用户消息，把 chunk_id 和文本组装进去"""
        chunk_id = chunk_data.get("chunk_id", "")
        text = chunk_data.get("dehydrated_text", chunk_data.get("text", ""))
        return f"""### CHUNK ID: {chunk_id}

### TEXT TO ANALYZE:
{text}"""

    def parse_concept_units(self, raw_response: str) -> List[ConceptUnit]:
        """
        解析模型返回的文本，从中提取概念单元列表
        支持两种格式：
        1. JSON 数组格式 [...]
        2. Markdown 代码块格式 ```json ... ```
        """
        # 尝试提取 JSON 数组
        text = raw_response.strip()

        # 去掉 markdown 代码块（支持 ```json ... ``` 或 ``` ... ```）
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])  # 去掉第一行 ```json
            text = text.rsplit("```", 1)[0]  # 去掉最后一个 ```

        # 用 json.JSONDecoder.raw_decode 自动找到第一个完整 JSON 对象（跳过前导非JSON内容）
        # 即使模型在 JSON 前后加了说明文字，此方法也能正确提取
        try:
            parsed, end_idx = json.JSONDecoder().raw_decode(text)
            if isinstance(parsed, list):
                return self._parse_concept_list(parsed)
        except json.JSONDecodeError:
            pass

        # 降级：返回一条"未解析"的占位记录
        logger.warning(
            f"Failed to parse concept units from response (first 200 chars): "
            f"{text[:200]!r}"
        )
        return [
            ConceptUnit(
                concept="[PARSE_FAILED]",
                definition=text[:500],
                prerequisites=[],
                impact="",
            )
        ]

    def _parse_concept_list(self, parsed: List[Any]) -> List[ConceptUnit]:
        """将解析出的列表转换为 ConceptUnit 对象列表"""
        units = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            unit = ConceptUnit(
                concept=str(item.get("concept", "")),
                definition=str(item.get("definition", "")),
                prerequisites=item.get("prerequisites", []),
                impact=str(item.get("impact", "")),
            )
            units.append(unit)
        return units

    def extract_from_chunk(self, chunk_data: Dict[str, Any]) -> List[ConceptUnit]:
        """
        从单个 chunk 数据中提取概念单元

        Args:
            chunk_data: 包含 chunk_id 和 dehydrated_text 的字典

        Returns:
            ConceptUnit 列表
        """
        user_message = self.build_user_message(chunk_data)

        response = self.model_client.chat_with_fresh_context(
            user_message=user_message,
            system_prompt=self.system_prompt,
        )

        return self.parse_concept_units(response)

    def extract_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        从单个脱水后的 JSON 文件中提取概念单元

        Args:
            file_path: dehydrated_chunk_*.json 文件路径

        Returns:
            可直接写入 jsonl 的字典列表，每条记录携带 chunk_id
        """
        with open(file_path, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)

        chunk_id = chunk_data.get("chunk_id", file_path.stem)

        try:
            concept_units = self.extract_from_chunk(chunk_data)
        except Exception as e:
            logger.error(f"Failed to extract from {file_path}: {e}")
            concept_units = [
                ConceptUnit(
                    concept="[EXTRACTION_FAILED]",
                    definition=str(e),
                    prerequisites=[],
                    impact="",
                )
            ]

        results = []
        for unit in concept_units:
            record = {"chunk_id": chunk_id, **unit.to_dict()}
            results.append(record)

        return results

    def extract_from_directory(
        self,
        input_dir: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        批量处理目录中的所有脱水文件

        Args:
            input_dir: 包含 dehydrated_chunk_*.json 的目录
            output_path: 输出 .jsonl 文件路径
            progress_callback: 进度回调函数

        Returns:
            处理统计
        """
        input_path = Path(input_dir)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 查找所有脱水后的 chunk 文件
        chunk_files = list(input_path.glob("dehydrated_chunk_*.json"))

        if not chunk_files:
            logger.warning(f"No dehydrated_chunk_*.json files found in {input_dir}")
            return {
                "total": 0,
                "processed": 0,
                "failed": 0,
                "errors": [],
            }

        stats = {
            "total": len(chunk_files),
            "processed": 0,
            "failed": 0,
            "errors": [],
            "total_concepts_extracted": 0,
        }

        # 写入 jsonl 文件
        with open(output_file, "w", encoding="utf-8") as out_f:
            workers = self.config.parallel_workers

            def process_one(chunk_file: Path) -> List[Dict[str, Any]]:
                """处理单个文件，每个 worker 使用独立的 config 和 client"""
                worker_config = LogicExtractionConfig(
                    provider=self.config.provider,
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    retry_count=self.config.retry_count,
                    retry_delay=self.config.retry_delay,
                    max_retry_delay=self.config.max_retry_delay,
                    system_prompt_file=self.config.system_prompt_file,
                )
                worker_client = CloudModelClient(worker_config)
                worker_extractor = LogicExtractor(
                    config=worker_config,
                    model_client=worker_client,
                )
                return worker_extractor.extract_from_file(chunk_file)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_one, f): f for f in chunk_files
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Logic Extraction",
                ):
                    chunk_file = futures[future]
                    try:
                        records = future.result()

                        for record in records:
                            out_f.write(
                                json.dumps(
                                    record, ensure_ascii=False
                                ) + "\n"
                            )
                            stats["total_concepts_extracted"] += 1

                        stats["processed"] += 1

                        if progress_callback:
                            progress_callback(stats["processed"], stats["total"])

                    except Exception as e:
                        logger.error(f"Failed to process {chunk_file}: {e}")
                        stats["failed"] += 1
                        stats["errors"].append({
                            "file": str(chunk_file),
                            "error": str(e),
                        })

        # 保存 manifest
        manifest = {
            "processor": "logic_extraction",
            "model": self.config.model,
            "api_url": self.config.api_url,
            "parallel_workers": self.config.parallel_workers,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "output_file": str(output_file),
        }

        manifest_path = output_file.parent / "logic_extraction_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return stats


# ============================================================
# 内置 System Prompt — 逻辑概念提取
# ============================================================

DEFAULT_EXTRACTION_PROMPT = """# Role
你是"因果逻辑链提取专家"，工作于一个高精度机器学习数据管线。你的任务是将文本拆解为一系列"逻辑概念单元（Concept Units）"，而非摘要。

# 核心原则
- **不要摘要，要提取逻辑链**：你的目标不是压缩或概括原文，而是识别并结构化文本中蕴含的因果逻辑、知识概念和推理链条。
- **保持语言一致性**：输出语言必须与输入文本语言完全一致（输入是中文就输出中文，是英文就输出英文）。
- **保留讲稿风格**：definition 字段应保留原始讲述风格，包括类比、举例、感叹等表达方式，不要改成干巴巴的学术定义。

# 每个 Concept Unit 的结构

每个概念单元（Concept Unit）是一个 JSON 对象，必须包含以下 4 个字段：

1. **concept**（概念名称）
   - 用一句话精炼命名该逻辑单元，通常是一个结论、原理或关键判断
   - 示例（中文文本）："梯度下降通过沿误差曲面最陡方向迭代更新参数"
   - 示例（英文文本）："Gradient descent updates parameters by iterating along the steepest error descent direction"

2. **definition**（深度解析）
   - 详细阐述该概念的含义、原理或结论
   - **必须保留原始讲稿的叙述风格**（类比、举例、个人感悟等）
   - 可以是多句话，字数不限
   - 示例："梯度下降就像蒙眼下山的人每一步都往坡度最陡的方向走，虽然不一定能到达谷底（全局最优），但通常能到达一个不错的低点（局部最优）"

3. **prerequisites**（前置逻辑）
   - 学习该概念前需要理解的前置逻辑概念列表
   - 由模型推理得出，是概念之间的逻辑依赖关系
   - 示例：["导数的几何意义", "误差函数的定义", "学习率的作用"]
   - 如果没有前置概念，返回空列表 []

4. **impact**（后果或延伸逻辑）
   - **重要规则**：如果文本中明确或隐含地表达"A导致了B"或"A引出了B"，则必须将B记录在 impact 中
   - 记录该概念带来的后果、引发的下一个逻辑、或延伸出的问题
   - 示例："参数更新可能越过最优解导致震荡；过小的学习率会使收敛速度极慢；batch大小影响梯度估计的稳定性"

# 任务定义
仔细阅读输入文本，识别其中的：
- 核心论点（Claim）
- 因果链条（Cause → Effect）
- 条件判断（If A, then B）
- 概念定义（Definition）
- 对比与类比（Contrast / Analogy）
- 转折与例外（Exception / Limitation）

将每个独立的逻辑单元格式化为一个 Concept Unit。

# 严格约束
- **不要摘要**：每一行输出都必须是一个结构化的逻辑单元，而非原文摘要
- **不要泛化**：每个 concept 字段应该是具体的、基于原文的判断，不写空泛的概念名
- **不要编造**：prerequisites 和 impact 必须来自原文推理，不得凭空添加
- **输出格式**：必须输出一个 JSON 数组，不要输出 Markdown 代码块包裹的 JSON，不要写任何解释性文字

# 示例

输入（中文）：
"梯度下降是机器学习中最常用的优化算法。它的核心思想是沿着误差函数梯度的反方向逐步调整参数，每一步的大小由学习率决定。如果学习率设置过大，参数更新会越过最优解导致发散；如果学习率过小，收敛速度会变得很慢。除了标准梯度下降，还有带动量的梯度下降，它利用历史梯度信息来加速收敛。"

期望输出（JSON 数组）：
```json
[
  {
    "concept": "梯度下降通过沿误差梯度的反方向迭代更新参数",
    "definition": "梯度下降的核心思想是沿着误差函数梯度的反方向逐步调整参数，每一步的大小由学习率决定。这就像蒙眼下山的人每一步都往最陡的方向走。",
    "prerequisites": ["误差函数的定义", "梯度的几何意义"],
    "impact": "标准梯度下降在遇到平坦区域时收敛变慢；在高曲率区域可能震荡。"
  },
  {
    "concept": "学习率过大导致参数更新越过最优解而发散",
    "definition": "如果学习率设置过大，参数在每次更新时迈的步子太大，会直接越过误差函数的最低点，导致无法收敛。",
    "prerequisites": ["学习率的作用", "梯度下降的基本原理"],
    "impact": "参数在最优解附近来回跳动，无法稳定收敛，需要降低学习率或使用学习率调度策略。"
  },
  {
    "concept": "学习率过小使收敛速度极慢",
    "definition": "学习率过小时，每次参数更新幅度很小，虽然不会越过最优解，但到达最优解需要的迭代次数会大幅增加。",
    "prerequisites": ["学习率的作用"],
    "impact": "训练时间大幅增加，计算成本上升。"
  },
  {
    "concept": "带动量的梯度下降利用历史梯度信息加速收敛",
    "definition": "动量方法在更新参数时不仅考虑当前梯度，还累积历史梯度的指数加权平均，从而在相关方向上加速，在振荡方向上抑制。",
    "prerequisites": ["标准梯度下降原理", "指数加权平均的概念"],
    "impact": "加速收敛，减少震荡，但引入了动量超参数需要调优。"
  }
]
```

开始提取，直接输出 JSON 数组，不要包含任何其他文字。"""


# ============================================================
# 便捷函数
# ============================================================

def extract_logic_chunks(
    chunks_dir: str,
    output_file: str,
    model: str = "gpt-4o",
    api_key: str = "",
    provider: str = "openai",
    base_url: str = "https://api.openai.com",
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """便捷函数：批量处理脱水后的 Chunk 目录"""
    config = LogicExtractionConfig(
        model=model,
        api_key=api_key,
        provider=provider,
        base_url=base_url,
        max_tokens=max_tokens,
    )

    extractor = LogicExtractor(config)
    return extractor.extract_from_directory(
        input_dir=Path(chunks_dir),
        output_path=Path(output_file),
    )


def extract_single_logic(
    chunk_file: str,
    output_file: str,
    model: str = "gpt-4o",
    api_key: str = "",
    provider: str = "openai",
    base_url: str = "https://api.openai.com",
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """便捷函数：处理单个脱水后的 Chunk 文件"""
    config = LogicExtractionConfig(
        model=model,
        api_key=api_key,
        provider=provider,
        base_url=base_url,
        max_tokens=max_tokens,
    )

    extractor = LogicExtractor(config)
    records = extractor.extract_from_file(Path(chunk_file))

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "processed": 1,
        "concepts_extracted": len(records),
        "output_file": str(output_path),
    }


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Logic Extraction — 从脱水文本中提取因果逻辑链和概念单元"
    )
    parser.add_argument(
        "input",
        help="脱水后的 Chunk 文件或目录（dehydrated_chunk_*.json）",
    )
    parser.add_argument(
        "output",
        help="输出 .jsonl 文件路径",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic"],
        help="API Provider（默认：openai）",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.openai.com",
        help="API 基础地址（默认：https://api.openai.com）",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API Key",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="模型名称（默认：gpt-4o）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="生成温度（默认：0.3）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Anthropic API 最大输出 token 数（默认：4096）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="API 请求超时秒数（默认：120）",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=20,
        help="最大重试次数（默认：20）",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=4,
        help="并行工作线程数（默认：4）",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System Prompt 文件路径（默认使用内置 Prompt）",
    )

    args = parser.parse_args()

    config = LogicExtractionConfig(
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retry_count=args.retry_count,
        parallel_workers=args.parallel_workers,
        system_prompt_file=args.system_prompt,
    )

    extractor = LogicExtractor(config)
    input_path = Path(args.input)

    if input_path.is_dir():
        stats = extractor.extract_from_directory(input_path, Path(args.output))
    else:
        records = extractor.extract_from_file(input_path)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats = {
            "processed": 1,
            "concepts_extracted": len(records),
        }

    print(f"\n{'='*50}")
    print("逻辑提取完成！")
    print(f"{'='*50}")
    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print(f"处理数量: {stats.get('processed', 0)}")
    print(f"失败数量: {stats.get('failed', 0)}")
    print(f"提取概念数: {stats.get('total_concepts_extracted', stats.get('concepts_extracted', 0))}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
