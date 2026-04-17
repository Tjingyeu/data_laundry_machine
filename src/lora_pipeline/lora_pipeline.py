"""
LoRA Data Preparation Pipeline - 主管道
整合四个程序，协调数据流和 chunk_id 传承。

Program 1: Smart Chunker - 智能切片
Program 2: Dehydration - 文本脱水（本地 AI）
Program 3: Logic Extraction - 因果链提取（云端 API）
Program 4: Logic Aggregator - 逻辑焊接与去重

chunk_id 贯穿整个流程，如同身份证号一样穿透所有步骤。
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum
import logging

import pandas as pd
from tqdm import tqdm

from .smart_chunker import SmartChunker, ChunkerConfig, Chunk
from .logic_aggregator import LogicAggregator, Concept, aggregate_logic_chains
from .dehydration import DehydrationProcessor, DehydrationConfig

logger = logging.getLogger(__name__)


# ============================================================
# 配置类
# ============================================================

@dataclass
class DehydrationConfig:
    """程序 2 配置"""
    enabled: bool = True
    batch_size: int = 10
    retry_count: int = 3
    retry_delay: float = 1.0


@dataclass
class ExtractionConfig:
    """程序 3 配置"""
    enabled: bool = True
    batch_size: int = 5
    retry_count: int = 3
    retry_delay: float = 2.0


@dataclass
class PipelineConfig:
    """主管道配置"""
    # 切片配置
    context_window: int = 32768
    chunk_ratio: float = 0.30
    overlap_ratio: float = 0.50

    # Dehydration 配置
    dehydration_enabled: bool = True

    # Extraction 配置
    extraction_enabled: bool = True

    # Aggregation 配置
    similarity_threshold: float = 0.9

    # 通用配置
    output_dir: str = "./output"
    verbose: bool = False


# ============================================================
# Pipeline 状态
# ============================================================

class PipelineStage(Enum):
    """管道阶段"""
    INITIAL = "initial"
    CHUNKING = "chunking"
    DEHYDRATION = "dehydration"
    EXTRACTION = "extraction"
    AGGREGATION = "aggregation"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineState:
    """管道状态"""
    stage: PipelineStage = PipelineStage.INITIAL
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    files_processed: int = 0
    chunks_created: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "files_processed": self.files_processed,
            "chunks_created": self.chunks_created,
            "errors": self.errors,
            "metadata": self.metadata
        }


# ============================================================
# LoRA Pipeline 主类
# ============================================================

class LoRAPipeline:
    """
    LoRA 数据准备管道
    按顺序执行：切片 → 脱水 → 提取 → 聚合
    chunk_id 全程传递
    """

    def __init__(
        self,
        ai_provider=None,
        config: Optional[PipelineConfig] = None
    ):
        self.config = config or PipelineConfig()
        self.ai_provider = ai_provider

        self.state = PipelineState()
        self.chunker = SmartChunker(
            ChunkerConfig(
                target_chunk_size=30000,
                overlap_ratio=self.config.overlap_ratio
            )
        )

        # 中间结果文件
        self._chunks_file = None
        self._dehydrated_file = None
        self._extracted_file = None
        self._final_file = None

    def _get_output_path(self, stage: PipelineStage, suffix: str = ".jsonl") -> Path:
        """获取输出路径"""
        base_dir = Path(self.config.output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = f"lora_{stage.value}{suffix}"
        return base_dir / filename

    def run(
        self,
        input_path: str,
        is_directory: bool = True,
        pattern: str = "*.txt",
        preprocess_callback: Optional[Callable] = None,
        postprocess_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        执行完整的 LoRA 数据准备管道

        Args:
            input_path: 输入文件或目录
            is_directory: 是否为目录
            pattern: 文件匹配模式（目录模式）
            preprocess_callback: 预处理回调
            postprocess_callback: 后处理回调

        Returns:
            管道执行结果
        """
        import asyncio

        self.state.start_time = datetime.now()
        self.state.stage = PipelineStage.CHUNKING

        results = {
            "config": {
                "target_chunk_size": 30000,
                "overlap_ratio": self.config.overlap_ratio
            },
            "stages": {}
        }

        try:
            # ============================================================
            # Stage 1: 智能切片
            # ============================================================
            logger.info("Stage 1: Smart Chunking...")
            self.state.stage = PipelineStage.CHUNKING

            chunks_result = self._run_chunking(input_path, is_directory, pattern)
            results["stages"]["chunking"] = chunks_result
            self.state.chunks_created = chunks_result.get("total_chunks", 0)

            if preprocess_callback:
                preprocess_callback(chunks_result)

            # ============================================================
            # Stage 2: 文本脱水（可选）
            # ============================================================
            if self.config.dehydration_enabled:
                logger.info("Stage 2: Dehydration...")
                self.state.stage = PipelineStage.DEHYDRATION

                dehydration_result = self._run_dehydration()
                results["stages"]["dehydration"] = dehydration_result

                if postprocess_callback:
                    postprocess_callback(dehydration_result, "dehydration")
            else:
                # 跳过脱水阶段，直接复制 chunks 文件
                self._dehydrated_file = self._chunks_file

            # ============================================================
            # Stage 3: 因果链提取（可选）
            # ============================================================
            if self.config.extraction_enabled:
                logger.info("Stage 3: Logic Extraction...")
                self.state.stage = PipelineStage.EXTRACTION

                extraction_result = self._run_extraction()
                results["stages"]["extraction"] = extraction_result

                if postprocess_callback:
                    postprocess_callback(extraction_result, "extraction")
            else:
                # 创建空的提取结果
                self._extracted_file = self._get_output_path(
                    PipelineStage.EXTRACTION
                )
                with open(self._extracted_file, "w") as f:
                    pass

            # ============================================================
            # Stage 4: 逻辑聚合
            # ============================================================
            logger.info("Stage 4: Logic Aggregation...")
            self.state.stage = PipelineStage.AGGREGATION

            aggregation_result = self._run_aggregation()
            results["stages"]["aggregation"] = aggregation_result

            # ============================================================
            # 完成
            # ============================================================
            self.state.stage = PipelineStage.COMPLETED
            self.state.end_time = datetime.now()

            results["final_output"] = str(self._final_file)
            results["total_duration_seconds"] = (
                self.state.end_time - self.state.start_time
            ).total_seconds()

            logger.info(f"Pipeline completed in {results['total_duration_seconds']:.2f}s")

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.state.stage = PipelineStage.FAILED
            self.state.errors.append(str(e))
            results["error"] = str(e)
            raise

    def _run_chunking(
        self,
        input_path: str,
        is_directory: bool,
        pattern: str
    ) -> Dict[str, Any]:
        """执行切片"""
        self._chunks_file = self._get_output_path(PipelineStage.CHUNKING)

        if is_directory:
            stats = self.chunker.chunk_directory_files(
                input_directory=input_path,
                output_file=str(self._chunks_file),
                pattern=pattern
            )
        else:
            stats = self.chunker.chunk_text_file(
                input_file=input_path,
                output_file=str(self._chunks_file)
            )

        self.state.files_processed = stats["total_files"]

        return {
            "method": "smart_chunker",
            "config": {
                "context_window": self.config.context_window,
                "chunk_ratio": self.config.chunk_ratio,
                "overlap_ratio": self.config.overlap_ratio
            },
            **stats
        }

    def _run_dehydration(self) -> Dict[str, Any]:
        """执行脱水处理"""
        self._dehydrated_file = self._get_output_path(PipelineStage.DEHYDRATION)

        chunks = self._load_chunks(self._chunks_file)

        # 实例化脱水处理器
        dehydrator = DehydrationProcessor(config=DehydrationConfig(
            model_name="qcwind/qwen2.5-7B-instruct-Q4_K_M",
            temperature=0.1
        ))

        stats = {
            "total_chunks": len(chunks),
            "processed": 0,
            "errors": []
        }

        # 创建临时目录存储 chunk 文件
        import tempfile
        import shutil
        temp_dir = Path(tempfile.mkdtemp())
        try:
            chunk_files = []
            for i, chunk in enumerate(chunks):
                chunk_file = temp_dir / f"chunk_{i}.json"
                with open(chunk_file, "w", encoding="utf-8") as f:
                    json.dump(chunk, f, ensure_ascii=False)
                chunk_files.append((chunk_file, i == 0, i == len(chunks) - 1))

            # 批量处理
            results = []
            for chunk_file, is_first, is_last in tqdm(chunk_files, desc="Dehydration"):
                try:
                    result = dehydrator.process_chunk(
                        chunk_file,
                        temp_dir,
                        is_first=is_first,
                        is_last=is_last
                    )
                    results.append(result)
                    stats["processed"] += 1
                except Exception as e:
                    logger.error(f"Dehydration error: {e}")
                    stats["errors"].append({"error": str(e)})

            # 写入最终文件
            with open(self._dehydrated_file, "w", encoding="utf-8") as out_f:
                for result in results:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return stats

    def _run_extraction(self) -> Dict[str, Any]:
        """执行因果链提取（同步版本）"""
        import asyncio

        self._extracted_file = self._get_output_path(PipelineStage.EXTRACTION)

        chunks = self._load_chunks(self._dehydrated_file)

        stats = {
            "total_chunks": len(chunks),
            "concepts_extracted": 0,
            "extraction_errors": []
        }

        # 如果没有 AI provider，生成模拟数据
        if not self.ai_provider:
            logger.warning("No AI provider configured, generating mock extraction results")

            with open(self._extracted_file, "w", encoding="utf-8") as out_f:
                for chunk in tqdm(chunks, desc="Extraction (mock)"):
                    mock_result = self._mock_extraction(chunk)
                    out_f.write(json.dumps(mock_result, ensure_ascii=False) + "\n")
                    stats["concepts_extracted"] += len(mock_result.get("concepts", []))

            return stats

        # 真实 API 调用 - 在新 event loop 中执行
        async def run_extraction_async():
            results = []
            for chunk in tqdm(chunks, desc="Logic Extraction"):
                try:
                    result = await self._extract_logic_chain(chunk)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Extraction error: {e}")
                    stats["extraction_errors"].append({
                        "chunk_id": chunk.get("chunk_id"),
                        "error": str(e)
                    })
            return results

        loop = asyncio.new_event_loop()
        try:
            extraction_results = loop.run_until_complete(run_extraction_async())
        finally:
            loop.close()

        with open(self._extracted_file, "w", encoding="utf-8") as out_f:
            for result in extraction_results:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                stats["concepts_extracted"] += len(result.get("concepts", []))

        return stats

    async def _extract_logic_chain(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """调用 AI API 提取因果链"""
        from ..ai_providers.base_provider import AIMessage, MessageRole

        chunk_id = chunk.get("chunk_id", "")

        # 构建提取请求
        prompt = f"""
从以下脱水文本中提取因果逻辑链。

文本：
{chunk.get('dehydrated_text', '')}

请输出 JSON 格式，包含：
- chunk_id: 必须原样传递
- concepts: 概念列表，每个包含 concept, definition, prerequisites, impact
- 特别标记 [待补全] 如果有缺失的前置知识
"""

        messages = [
            AIMessage(role=MessageRole.USER, content=prompt)
        ]

        # 调用 API
        response = await self.ai_provider.chat(messages)

        # 解析响应
        # 假设 API 返回的是 JSON 格式
        try:
            result = json.loads(response.content)
            result["chunk_id"] = chunk_id  # 确保 chunk_id 传递
        except json.JSONDecodeError:
            # 如果不是 JSON，创建简单的结构
            result = {
                "chunk_id": chunk_id,
                "concepts": []
            }

        return result

    def _mock_extraction(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟的提取结果（用于测试）"""
        import uuid

        chunk_id = chunk.get("chunk_id", "")

        # 简单地从文本中提取一些"概念"
        text = chunk.get("dehydrated_text", "")

        # 提取前 100 个字符作为"概念"
        concept_name = text[:20].strip() if text else "Unknown"

        return {
            "chunk_id": chunk_id,
            "extraction_timestamp": datetime.now().isoformat(),
            "concepts": [
                {
                    "concept_id": str(uuid.uuid4()),
                    "concept": concept_name,
                    "definition": text[:200] if text else "",
                    "prerequisites": [],
                    "impact": "[待补全] 需要进一步分析",
                    "evidence": {
                        "source_sentences": [text[:100]] if text else [],
                        "location": "chunk_0"
                    },
                    "logical_coherence": {
                        "cause_valid": True,
                        "effect_valid": True,
                        "link_strength": "medium",
                        "reasoning": "Mock extraction"
                    },
                    "flags": {
                        "has_unresolved_prerequisites": True,
                        "requires_external_knowledge": False,
                        "cross_chunk_reference": None
                    }
                }
            ],
            "relationships": [],
            "unresolved_gaps": [
                {
                    "gap_concept": "相关背景",
                    "gap_description": "[待补全] 需要补充背景知识",
                    "marked_as": "[待补全] 相关背景知识"
                }
            ],
            "quality_score": 0.5
        }

    def _run_aggregation(self) -> Dict[str, Any]:
        """执行逻辑聚合"""
        if not Path(self._extracted_file).exists():
            logger.warning("No extraction file found, skipping aggregation")
            return {"skipped": True}

        self._final_file = self._get_output_path(PipelineStage.AGGREGATION)

        try:
            stats = aggregate_logic_chains(
                input_file=str(self._extracted_file),
                output_file=str(self._final_file),
                similarity_threshold=self.config.similarity_threshold
            )

            return stats

        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return {
                "error": str(e),
                "original_concepts": 0,
                "after_welding": 0
            }

    def _load_chunks(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载 chunks 文件"""
        chunks = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunks.append(json.loads(line))

        return chunks

    def get_state(self) -> PipelineState:
        """获取当前状态"""
        return self.state

    def cleanup(self) -> None:
        """清理中间文件"""
        files_to_keep = [self._final_file]

        for f in [self._chunks_file, self._dehydrated_file, self._extracted_file]:
            if f and Path(f).exists() and f not in files_to_keep:
                try:
                    Path(f).unlink()
                    logger.debug(f"Cleaned up {f}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {f}: {e}")


# ============================================================
# 便捷函数
# ============================================================

def run_lora_pipeline(
    input_path: str,
    output_dir: str = "./output",
    is_directory: bool = True,
    pattern: str = "*.txt",
    context_window: int = 32768,
    similarity_threshold: float = 0.9,
    enable_dehydration: bool = True,
    enable_extraction: bool = True,
    ai_provider=None
) -> Dict[str, Any]:
    """
    便捷函数：运行完整的 LoRA 管道
    """
    config = PipelineConfig(
        context_window=context_window,
        similarity_threshold=similarity_threshold,
        dehydration_enabled=enable_dehydration,
        extraction_enabled=enable_extraction,
        output_dir=output_dir
    )

    pipeline = LoRAPipeline(
        ai_provider=ai_provider,
        config=config
    )

    return pipeline.run(
        input_path=input_path,
        is_directory=is_directory,
        pattern=pattern
    )


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LoRA Data Preparation Pipeline"
    )
    parser.add_argument(
        "input",
        help="输入文件或目录"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="输出目录"
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="文件匹配模式"
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=32768,
        help="API 上下文窗口大小"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.9,
        help="相似度阈值"
    )
    parser.add_argument(
        "--skip-dehydration",
        action="store_true",
        help="跳过脱水阶段"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="跳过提取阶段"
    )

    args = parser.parse_args()

    results = run_lora_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        pattern=args.pattern,
        context_window=args.context_window,
        similarity_threshold=args.similarity_threshold,
        enable_dehydration=not args.skip_dehydration,
        enable_extraction=not args.skip_extraction
    )

    print(f"\n{'='*50}")
    print("LoRA Pipeline 完成！")
    print(f"{'='*50}")
    print(f"总耗时: {results.get('total_duration_seconds', 0):.2f}s")
    print(f"最终输出: {results.get('final_output')}")


if __name__ == "__main__":
    main()
