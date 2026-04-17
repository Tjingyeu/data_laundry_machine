"""
Program 2: 文本脱水处理器 (Dehydration Processor)
使用本地小模型对每个 Chunk 进行脱水处理。

特点：
1. 每个 Chunk 单独处理，使用新的 session/clean context
2. 支持多种本地模型 API（Ollama、llama.cpp、FastChat 等）
3. 自动检测模型上下文窗口，确保 Chunk 不超过 30%
4. chunk_id 全程传递
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
import os

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================
# 配置类
# ============================================================

@dataclass
class DehydrationConfig:
    """脱水配置"""
    # 模型 API 地址
    api_url: str = "http://localhost:11434/api/chat"  # Ollama 默认

    # 模型名称（需要是本地已下载的模型）
    model_name: str = "qcwind/qwen2.5-7B-instruct-Q4_K_M"

    # 模型上下文窗口大小
    model_context_window: int = 32768  # 32k 上下文

    # Chunk 最大占用上下文比例
    chunk_context_ratio: float = 0.30

    # 生成参数
    temperature: float = 0.2

    # API 请求超时（秒）
    timeout: int = 300

    # 最大重试次数
    retry_count: int = 3

    # 重试间隔（秒）
    retry_delay: float = 2.0

    # 并行处理数（建议 1，因为每个模型实例需要独立）
    parallel_workers: int = 1

    # System Prompt 文件路径（可选）
    system_prompt_file: Optional[str] = None

    # Stage 1 Prompt 文件路径（纯文本生成）
    stage1_prompt_file: Optional[str] = None

    # Stage 2 Prompt 文件路径（JSON 结构生成）
    stage2_prompt_file: Optional[str] = None

    @property
    def max_chunk_tokens(self) -> int:
        """Chunk 允许的最大 Token 数"""
        return int(self.model_context_window * self.chunk_context_ratio)  # 30% of 128k = 38.4k tokens


# ============================================================
# 本地模型 API 客户端
# ============================================================

class LocalModelClient:
    """
    本地模型客户端
    支持 Ollama、llama.cpp、FastChat 等兼容 API
    """

    def __init__(self, config: DehydrationConfig):
        self.config = config
        self.client = httpx.Client(timeout=config.timeout)

    def _load_system_prompt(self) -> str:
        """加载 System Prompt"""
        if self.config.system_prompt_file:
            prompt_path = Path(self.config.system_prompt_file)
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return f.read()

        # 默认 System Prompt
        return DEFAULT_DEHYDRATION_PROMPT

    def _load_stage1_prompt(self) -> str:
        """加载 Stage 1 Prompt"""
        if self.config.stage1_prompt_file:
            prompt_path = Path(self.config.stage1_prompt_file)
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return f.read()
        return DEFAULT_STAGE1_PROMPT

    def _load_stage2_prompt(self) -> str:
        """加载 Stage 2 Prompt（已废弃，仅保留接口兼容）"""
        if self.config.stage2_prompt_file:
            prompt_path = Path(self.config.stage2_prompt_file)
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return f.read()
        return ""  # Stage 2 不再使用

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None
    ) -> str:
        """
        发送对话请求到本地模型

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            model: 模型名称（可选）

        Returns:
            模型的回复内容
        """
        model = model or self.config.model_name

        # 检测消息总长度是否超限
        total_chars = sum(len(m["content"]) for m in messages)
        estimated_tokens = total_chars // 2  # 粗略估算

        if estimated_tokens > self.config.max_chunk_tokens:
            logger.warning(
                f"Input too long ({estimated_tokens} tokens), "
                f"max recommended: {self.config.max_chunk_tokens}"
            )

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": self.config.temperature,
            "options": {
                "num_ctx": self.config.model_context_window
            },
            "keep_alive": 300  # 保持模型加载5分钟，避免频繁重新加载影响速度
        }

        for attempt in range(self.config.retry_count):
            try:
                response = self.client.post(
                    self.config.api_url,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                return result.get("message", {}).get("content", "")

            except httpx.HTTPError as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise

        return ""

    def chat_with_fresh_context(
        self,
        user_message: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        使用全新的上下文发送消息
        每次调用都会清空历史，相当于新 session

        Args:
            user_message: 用户消息
            system_prompt: System Prompt（可选）

        Returns:
            模型回复
        """
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": user_message
        })

        return self.chat(messages)


# ============================================================
# 脱水处理器
# ============================================================

class DehydrationProcessor:
    """
    脱水处理器
    逐个处理 Chunk 文件，每次使用新 session
    """

    def __init__(
        self,
        config: Optional[DehydrationConfig] = None,
        model_client: Optional[LocalModelClient] = None
    ):
        self.config = config or DehydrationConfig()
        self.model_client = model_client or LocalModelClient(self.config)

        # 加载 System Prompt
        self.system_prompt = self.model_client._load_system_prompt()
        self.stage1_prompt = self.model_client._load_stage1_prompt()
        self.stage2_prompt = self.model_client._load_stage2_prompt()

    def validate_stage1_output(self, text: str) -> str:
        """校验 Stage 1 输出（如果明显是 JSON 才拒绝）"""
        # 仅当文本明显像 JSON 对象时才拒绝
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            raise ValueError(f"Stage1 output looks like JSON: {text[:100]}")
        return text

    def run_stage1(
        self,
        raw_text: str,
        is_first: bool = False,
        is_last: bool = False
    ) -> str:
        """Stage 1: 生成脱水后的纯文本（带边缘裁剪上下文）"""
        edge_warning = ""
        if not is_first:
            edge_warning += "The BEGINNING of this text is cut from a larger file. Remove any broken words at the start. "
        if not is_last:
            edge_warning += "The END of this text is cut. Remove any broken sentences at the very end."

        user_content = f"### CONTEXT ###\n{edge_warning}\n\n### TEXT TO CLEAN ###\n{raw_text}"

        messages = [
            {"role": "system", "content": self.stage1_prompt},
            {"role": "user", "content": user_content}
        ]
        response = self.model_client.chat(messages)

        # 暴力去除 Markdown 代码块
        cleaned = response.replace("```json", "").replace("```", "").strip()

        return cleaned


    def load_chunk(self, chunk_file: Path) -> Dict[str, Any]:
        """加载单个 Chunk 文件"""
        with open(chunk_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_dehydrated(
        self,
        chunk_data: Dict[str, Any],
        output_file: Path
    ) -> None:
        """保存脱水后的结果"""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    def process_chunk(
        self,
        chunk_file: Path,
        output_dir: Path,
        original_text: Optional[str] = None,
        is_first: bool = False,
        is_last: bool = False
    ) -> Dict[str, Any]:
        """
        处理单个 Chunk - 纯 Stage 1 生成

        Args:
            chunk_file: Chunk 文件路径
            output_dir: 输出目录
            original_text: 可选的原始文本
            is_first: 是否是第一个 chunk（影响边缘裁剪上下文）
            is_last: 是否是最后一个 chunk（影响边缘裁剪上下文）

        Returns:
            处理结果
        """
        chunk_data = self.load_chunk(chunk_file)
        chunk_id = chunk_data.get("chunk_id", "")
        text = original_text or chunk_data.get("text", "")

        logger.debug(f"Processing chunk {chunk_id}: {len(text)} chars")

        original_length = len(text)

        try:
            # Stage 1: 生成纯文本
            clean_text = self.run_stage1(text, is_first=is_first, is_last=is_last)
            clean_text = self.validate_stage1_output(clean_text)
        except ValueError as e:
            logger.warning(f"Stage1 validation failed for {chunk_id}, using original: {e}")
            clean_text = text

        # Python 直接生成安全、确定性的 JSON
        dehydrated_length = len(clean_text)

        result = {
            "chunk_id": chunk_id,
            "dehydrated_text": clean_text,
            "stats": {
                "original_length": original_length,
                "dehydrated_length": dehydrated_length,
                "compression_ratio": round(dehydrated_length / original_length, 4) if original_length > 0 else 1.0
            }
        }

        # 保存结果
        output_file = output_dir / f"dehydrated_{chunk_file.stem}.json"
        self.save_dehydrated(result, output_file)

        result["output_file"] = str(output_file)

        return result


    def process_directory(
        self,
        chunks_dir: str,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        批量处理目录中的所有 Chunk 文件

        Args:
            chunks_dir: Chunk 文件所在目录
            output_dir: 输出目录
            progress_callback: 进度回调函数

        Returns:
            处理统计
        """
        chunks_path = Path(chunks_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 查找所有 chunk 文件
        chunk_files = list(chunks_path.glob("chunk_*.json"))

        if not chunk_files:
            logger.warning(f"No chunk files found in {chunks_dir}")
            return {
                "total": 0,
                "processed": 0,
                "failed": 0,
                "errors": []
            }

        stats = {
            "total": len(chunk_files),
            "processed": 0,
            "failed": 0,
            "errors": [],
            "total_chars_input": 0,
            "total_chars_output": 0,
            "total_compression_ratio": 0.0
        }

        # 并行处理
        workers = self.config.parallel_workers

        def process_one(chunk_file: Path) -> Dict[str, Any]:
            """处理单个 chunk，每个 worker 使用独立的 config 和 client"""
            worker_config = DehydrationConfig(
                api_url=self.config.api_url,
                model_name=self.config.model_name,
                model_context_window=self.config.model_context_window,
                chunk_context_ratio=self.config.chunk_context_ratio,
                temperature=self.config.temperature,
                timeout=self.config.timeout,
                retry_count=self.config.retry_count,
                retry_delay=self.config.retry_delay,
                system_prompt_file=self.config.system_prompt_file,
                stage1_prompt_file=self.config.stage1_prompt_file,
                stage2_prompt_file=self.config.stage2_prompt_file
            )
            worker_processor = DehydrationProcessor(config=worker_config)
            return worker_processor.process_chunk(chunk_file, output_path)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, f): f for f in chunk_files}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Dehydration"):
                chunk_file = futures[future]
                try:
                    result = future.result()

                    if "error" in result:
                        stats["failed"] += 1
                        stats["errors"].append({
                            "chunk_id": result.get("chunk_id", ""),
                            "error": result["error"]
                        })
                    else:
                        stats["processed"] += 1
                        stats["total_chars_input"] += result["stats"]["original_length"]
                        stats["total_chars_output"] += result["stats"]["dehydrated_length"]
                        stats["total_compression_ratio"] += result["stats"]["compression_ratio"]

                    if progress_callback:
                        progress_callback(stats["processed"], stats["total"])

                except Exception as e:
                    logger.error(f"Failed to process {chunk_file}: {e}")
                    stats["failed"] += 1
                    stats["errors"].append({
                        "file": str(chunk_file),
                        "error": str(e)
                    })

        # 计算平均压缩比
        if stats["processed"] > 0:
            stats["avg_compression_ratio"] = (
                stats["total_chars_output"] / stats["total_chars_input"]
                if stats["total_chars_input"] > 0 else 1.0
            )

        # 保存 manifest
        manifest = {
            "processor": "dehydration",
            "model": self.config.model_name,
            "model_context_window": self.config.model_context_window,
            "parallel_workers": workers,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "files": [
                f.name for f in output_path.glob("dehydrated_*.json")
            ]
        }

        manifest_path = output_path / "dehydration_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        stats["manifest_file"] = str(manifest_path)

        return stats

    def process_single(
        self,
        chunk_file: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """处理单个 Chunk 文件"""
        chunk_path = Path(chunk_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        result = self.process_chunk(chunk_path, output_path)

        # 保存 manifest
        manifest = {
            "processor": "dehydration",
            "model": self.config.model_name,
            "timestamp": datetime.now().isoformat(),
            "input_file": str(chunk_path),
            "result": result
        }

        manifest_path = output_path / "single_dehydration_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return result


# ============================================================
# 默认 System Prompt（针对 GLM4-9B 优化）
# ============================================================

DEFAULT_DEHYDRATION_PROMPT = """You are a Data Refinement Specialist in a high-precision machine learning pipeline. Your task is to clean noisy speech-to-text (ASR) transcripts and output structured JSON, while preserving 100% of the original logic, details, and length.



CORE TASKS:

1. Verbal Dehydration: Remove all filler words (e.g., "um," "ah," "uh," "you know," "like," "so") and redundant stutters.

2. ASR Correction: Fix phonetic typos using technical context.

3. Edge Pruning: Identify and remove incomplete words or broken phrases ONLY at the absolute beginning and end of the chunk. Ensure the text starts and ends with a complete semantic unit.

4. Information Preservation (CRITICAL): Keep every technical detail, analogy, example, and causal link. DO NOT summarize. DO NOT omit details.



STRICT CONSTRAINTS:

- No Summarization: The target compression ratio is between 0.7 and 0.95. If you reduce the text length significantly (e.g., a ratio below 0.60), you have failed the task. Rewrite oral speech into direct, written statements without dropping data.

- No Conversational Filler: Do not output "Here is the JSON", "Sure", or any markdown outside the JSON block.

- JSON Integrity: Ensure the `dehydrated_text` value is properly escaped for JSON. Use `\n` for newlines and escape double quotes as `\"`.



OUTPUT FORMAT:

You must output ONLY a valid JSON object matching this exact schema:

{

  "chunk_id": "id_from_user_input",

  "dehydrated_text": "The fully cleaned, unsummarized narrative text...",

  "corrections": ["list of major technical typos fixed, e.g., 'go dot' -> 'Godot'"],

  "stats": {

    "original_length": 0,

    "dehydrated_length": 0,

    "compression_ratio": 0.00

  }

}"""


# ============================================================
# 默认 Stage 1 Prompt（纯文本生成）
# ============================================================

DEFAULT_STAGE1_PROMPT = """# Role
You are a "High-Precision Data Refinement Specialist" in a machine learning pipeline. Your mission is to transform noisy, raw Speech-to-Text (ASR) transcriptions into clean, written-style narrative data.

# Core Task: Dehydration, Correction & Repair
1. **Verbal Dehydration**:
   - Strip all filler words: e.g., "um," "ah," "uh," "you know," "like," "actually," "so to speak," "then basically."
   - Eliminate redundant repetitions: e.g., change "we, we need, we need to check" to "we need to check."
2. **ASR Error Correction**:
   - Fix homophone errors: Use context to correct misidentified words (e.g., changing "site" to "cite" or "weather" to "whether" based on technical context).
   - Logical Punctuation: Break long, rambling run-on sentences into clear, concise statements.
3. **Edge Pruning**:
   - Identify and discard "fragments" at the start and end of the text. If a sentence or phrase is cut off or incomplete due to slicing, remove it entirely. Ensure the output begins and ends with complete semantic units.
4. **100% Logic Preservation**:
   - **NO Summarization**: Retain every technical detail, metaphor, example, causal link, and original argument.
   - **NO Content Reduction**: Your goal is "purification," not "compression." It is better to keep a slightly longer detailed explanation than to omit a single information point.

# Strict Constraints
- **NO Third-Person Description**: Do not use phrases like "The speaker says..." or "This text describes..." Restore the content as a direct statement or in the original first-person perspective.
- **NO Meta-Talk**: Do NOT respond with "Sure," "Here is the cleaned text," or any introductory/concluding remarks.
- **NO Hallucinations**: Do not add any external knowledge or information not present in the input.
- **Output Format**: Output ONLY the refined plain text. Do NOT include Markdown headers, code blocks (```), or any JSON structures.

# Workflow Guideline
- Step 1: Scan for ASR errors and fillers.
- Step 2: Evaluate the integrity of the beginning and end; prune fragmented phrases.
- Step 3: Reconstruct into a professional written flow without altering any causal logic.
- Step 4: Final Check: Verify that no original information points were lost during the process.

Start directly with the first cleaned sentence:"""

# ============================================================
# 便捷函数
# ============================================================

def dehydrate_chunks(
    chunks_dir: str,
    output_dir: str,
    model_name: str = "qwen2.5:3b",
    api_url: str = "http://localhost:11434/api/chat"
) -> Dict[str, Any]:
    """便捷函数：批量处理 Chunk 目录"""
    config = DehydrationConfig(
        model_name=model_name,
        api_url=api_url
    )

    processor = DehydrationProcessor(config)
    return processor.process_directory(chunks_dir, output_dir)


def dehydrate_single_chunk(
    chunk_file: str,
    output_dir: str,
    model_name: str = "qwen2.5:3b",
    api_url: str = "http://localhost:11434/api/chat"
) -> Dict[str, Any]:
    """便捷函数：处理单个 Chunk"""
    config = DehydrationConfig(
        model_name=model_name,
        api_url=api_url
    )

    processor = DehydrationProcessor(config)
    return processor.process_single(chunk_file, output_dir)


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dehydration Processor - 文本脱水"
    )
    parser.add_argument(
        "input",
        help="Chunk 文件或目录"
    )
    parser.add_argument(
        "output",
        help="输出目录"
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:3b",
        help="本地模型名称"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:11434/api/chat",
        help="模型 API 地址"
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=32768,
        help="模型上下文窗口大小"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System Prompt 文件路径（默认使用内置 Prompt）"
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="并行处理的工作进程数（建议1，因为Ollama主要支持串行）"
    )
    parser.add_argument(
        "--stage1-prompt",
        type=str,
        default=None,
        help="Stage 1 Prompt 文件路径（纯文本生成）"
    )
    parser.add_argument(
        "--stage2-prompt",
        type=str,
        default=None,
        help="Stage 2 Prompt 文件路径（JSON 结构生成）"
    )

    args = parser.parse_args()

    config = DehydrationConfig(
        model_name=args.model,
        api_url=args.api_url,
        model_context_window=args.context_window,
        system_prompt_file=args.system_prompt,
        parallel_workers=args.parallel_workers,
        stage1_prompt_file=args.stage1_prompt,
        stage2_prompt_file=args.stage2_prompt
    )

    processor = DehydrationProcessor(config)

    input_path = Path(args.input)

    if input_path.is_dir():
        stats = processor.process_directory(args.input, args.output)
    else:
        result = processor.process_single(args.input, args.output)
        stats = {"single": result}

    print(f"\n{'='*50}")
    print("脱水完成！")
    print(f"{'='*50}")
    print(f"处理数量: {stats.get('processed', 1)}")
    print(f"失败数量: {stats.get('failed', 0)}")
    if "avg_compression_ratio" in stats:
        print(f"平均压缩比: {stats['avg_compression_ratio']:.2%}")
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
