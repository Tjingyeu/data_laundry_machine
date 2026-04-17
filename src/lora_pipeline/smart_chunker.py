"""
Program 1: Smart Chunker - 智能切片
基于 tiktoken 的动态滑动窗口切片，支持批量处理文件夹中的文件。

功能：
1. 检测目标 API 的上下文窗口（默认按 32k Token 计算）
2. 取窗口的 30% 字符量作为切片长度（限制在 10k-50k 字）
3. 设置 50% 的滑动重叠
4. 切片优化：寻找最近的句子终止符
5. 支持单个文件输出（Program 2 本地模型处理用）
"""

import os
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import tiktoken
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================
# 配置类
# ============================================================

@dataclass
class ChunkerConfig:
    """切片器配置"""
    # 切片目标长度（字符）- 固定 4k
    target_chunk_size: int = 4000

    # 重叠比例
    overlap_ratio: float = 0.50

    # 句子终止符
    sentence_delimiters: List[str] = field(default_factory=lambda: [
        '。', '！', '？', '.\n', '!\n', '?\n',
        '...', '…', '\n\n', '；'
    ])

    # 编码模型（用于 tiktoken）
    encoding_model: str = "cl100k_base"  # GPT-4 / Claude 用的编码

    # 输出模式
    output_mode: str = "individual_files"  # individual_files | jsonl

    @property
    def overlap_size(self) -> int:
        """重叠大小（字符）"""
        return int(self.target_chunk_size * self.overlap_ratio)

    @property
    def overlap_size(self) -> int:
        """重叠大小（字符）"""
        return int(self.target_chunk_size * self.overlap_ratio)


# ============================================================
# Chunk 数据结构
# ============================================================

@dataclass
class Chunk:
    """切片数据"""
    chunk_id: str  # 唯一标识，贯穿整个流程
    source_file: str  # 来源文件
    source_type: str  # txt 或 srt
    text: str  # 切片文本
    char_count: int  # 字符数
    token_count: int  # Token 数
    start_pos: int  # 在原文中的起始位置
    end_pos: int  # 在原文中的结束位置
    chunk_index: int  # 在该文件中的切片序号
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "source_type": self.source_type,
            "text": self.text,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata
        }

    def to_file_format(self) -> Dict[str, Any]:
        """转换为单独文件存储格式"""
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "source_type": self.source_type,
            "text": self.text,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata
        }

    @property
    def parent_id(self) -> str:
        """父级 ID（用于追踪）"""
        return f"{Path(self.source_file).stem}_{self.chunk_index}"

    def get_output_filename(self) -> str:
        """生成输出文件名"""
        source_name = Path(self.source_file).stem
        return f"chunk_{source_name}_{self.chunk_index:04d}_{self.chunk_id[:8]}.json"


# ============================================================
# Smart Chunker 主类
# ============================================================

class SmartChunker:
    """
    智能切片器
    使用 tiktoken 精确计算 Token 数量，
    实现动态滑动窗口切片，并优化断句。
    """

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()
        self.encoding = tiktoken.get_encoding(self.config.encoding_model)

    def count_tokens(self, text: str) -> int:
        """使用 tiktoken 精确计算 Token 数量"""
        return len(self.encoding.encode(text))

    def count_tokens_safe(self, text: str) -> int:
        """安全计算 Token（处理特殊字符）"""
        try:
            return self.count_tokens(text)
        except Exception:
            return int(len(text) / 2.5)

    def find_sentence_boundary(
        self,
        text: str,
        target_pos: int
    ) -> int:
        """
        寻找最近的句子终止符
        从 target_pos 向前和向后搜索，找到最近的完整句子边界
        """
        delimiters = self.config.sentence_delimiters

        # 向前搜索
        backward_pos = target_pos
        for _ in range(1000):
            if backward_pos <= 0:
                break
            found = False
            for d in delimiters:
                if text[max(0, backward_pos-len(d)):backward_pos] == d:
                    backward_pos += len(d)
                    found = True
                    break
            if found:
                break
            backward_pos -= 1

        # 向后搜索
        forward_pos = target_pos
        for _ in range(1000):
            if forward_pos >= len(text):
                break
            for d in delimiters:
                if text[forward_pos:forward_pos+len(d)] == d:
                    forward_pos += len(d)
                    return forward_pos
            forward_pos += 1

        return target_pos

    def chunk_text(
        self,
        text: str,
        source_file: str,
        source_type: str = "txt"
    ) -> List[Chunk]:
        """对单个文本进行切片"""
        chunks = []
        chunk_size = self.config.target_chunk_size
        overlap_size = self.config.overlap_size

        if len(text) <= chunk_size:
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                source_file=source_file,
                source_type=source_type,
                text=text,
                char_count=len(text),
                token_count=self.count_tokens_safe(text),
                start_pos=0,
                end_pos=len(text),
                chunk_index=0,
                metadata={"chunking_method": "single"}
            )
            chunks.append(chunk)
            return chunks

        # 滑动窗口切片
        position = 0
        chunk_index = 0

        while position < len(text):
            window_end = min(position + chunk_size, len(text))

            if window_end < len(text):
                boundary = self.find_sentence_boundary(text, window_end)
                min_boundary = position + int(chunk_size * 0.5)
                if boundary < min_boundary:
                    boundary = window_end
            else:
                boundary = window_end

            chunk_text = text[position:boundary]

            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                source_file=source_file,
                source_type=source_type,
                text=chunk_text,
                char_count=len(chunk_text),
                token_count=self.count_tokens_safe(chunk_text),
                start_pos=position,
                end_pos=boundary,
                chunk_index=chunk_index,
                metadata={
                    "chunking_method": "sliding_window",
                    "overlap_with_previous": chunk_index > 0
                }
            )
            chunks.append(chunk)

            position = boundary - overlap_size
            if position <= chunks[-1].end_pos:
                position = boundary
            chunk_index += 1

        # 添加父子关系
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.metadata["previous_chunk_id"] = chunks[i-1].chunk_id
            if i < len(chunks) - 1:
                chunk.metadata["next_chunk_id"] = chunks[i+1].chunk_id

        return chunks

    def chunk_file(
        self,
        file_path: str,
        encoding: str = "utf-8"
    ) -> List[Chunk]:
        """对单个文件进行切片"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        source_type = "srt" if suffix == ".srt" else "txt"

        with open(path, "r", encoding=encoding) as f:
            text = f.read()

        return self.chunk_text(text, file_path, source_type)

    def save_chunk_to_file(
        self,
        chunk: Chunk,
        output_dir: Path
    ) -> Path:
        """将单个 Chunk 保存为单独文件"""
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = chunk.get_output_filename()
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chunk.to_file_format(), f, ensure_ascii=False, indent=2)

        return filepath

    def chunk_directory(
        self,
        directory: str,
        pattern: str = "*.txt",
        recursive: bool = True,
        encoding: str = "utf-8"
    ) -> Iterator[Tuple[str, List[Chunk]]]:
        """批量处理目录中的文件"""
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = list(dir_path.rglob(pattern)) if recursive else list(dir_path.glob(pattern))
        srt_files = list(dir_path.rglob("*.srt")) if recursive else list(dir_path.glob("*.srt"))
        files = list(set(files) | set(srt_files))

        logger.info(f"Found {len(files)} files to process in {directory}")

        for file_path in files:
            try:
                chunks = self.chunk_file(str(file_path), encoding)
                yield str(file_path), chunks
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

    def process_and_save(
        self,
        input_path: str,
        output_dir: str,
        is_directory: bool = True,
        pattern: str = "*.txt",
        encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """
        处理文件/目录并保存结果

        Args:
            input_path: 输入文件或目录路径
            output_dir: 输出目录路径（每个 chunk 保存为单独文件）
            is_directory: input_path 是否为目录
            pattern: 文件匹配模式
            encoding: 文件编码

        Returns:
            处理统计信息
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_files": 0,
            "total_chunks": 0,
            "total_chars": 0,
            "total_tokens": 0,
            "files_processed": [],
            "files_failed": [],
            "chunks": []  # 保存每个 chunk 的信息
        }

        if is_directory:
            for file_path, chunks in tqdm(
                self.chunk_directory(input_path, pattern=pattern, encoding=encoding),
                desc="Processing files"
            ):
                stats["total_files"] += 1
                stats["files_processed"].append(file_path)

                for chunk in chunks:
                    filepath = self.save_chunk_to_file(chunk, output_path)
                    stats["chunks"].append({
                        "chunk_id": chunk.chunk_id,
                        "filename": filepath.name,
                        "source_file": chunk.source_file,
                        "char_count": chunk.char_count,
                        "token_count": chunk.token_count,
                        "chunk_index": chunk.chunk_index
                    })
                    stats["total_chunks"] += 1
                    stats["total_chars"] += chunk.char_count
                    stats["total_tokens"] += chunk.token_count
        else:
            chunks = self.chunk_file(input_path, encoding)
            stats["total_files"] = 1
            stats["files_processed"].append(input_path)

            for chunk in chunks:
                filepath = self.save_chunk_to_file(chunk, output_path)
                stats["chunks"].append({
                    "chunk_id": chunk.chunk_id,
                    "filename": filepath.name,
                    "source_file": chunk.source_file,
                    "char_count": chunk.char_count,
                    "token_count": chunk.token_count,
                    "chunk_index": chunk.chunk_index
                })
                stats["total_chunks"] += 1
                stats["total_chars"] += chunk.char_count
                stats["total_tokens"] += chunk.token_count

        # 保存索引文件
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_chunks": stats["total_chunks"],
                "total_files": stats["total_files"],
                "total_chars": stats["total_chars"],
                "total_tokens": stats["total_tokens"],
                "chunks": stats["chunks"]
            }, f, ensure_ascii=False, indent=2)

        stats["manifest_file"] = str(manifest_path)

        logger.info(f"Processing complete: {stats['total_chunks']} chunks saved to {output_dir}")

        return stats


# ============================================================
# 便捷函数
# ============================================================

def chunk_text_file(
    input_file: str,
    output_dir: str,
    config: Optional[ChunkerConfig] = None
) -> Dict[str, Any]:
    """便捷函数：处理单个文本文件"""
    chunker = SmartChunker(config)
    return chunker.process_and_save(
        input_path=input_file,
        output_dir=output_dir,
        is_directory=False
    )


def chunk_directory_files(
    input_directory: str,
    output_dir: str,
    pattern: str = "*.txt",
    config: Optional[ChunkerConfig] = None
) -> Dict[str, Any]:
    """便捷函数：批量处理目录"""
    chunker = SmartChunker(config)
    return chunker.process_and_save(
        input_path=input_directory,
        output_dir=output_dir,
        is_directory=True,
        pattern=pattern
    )


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Chunker - 智能文本切片器"
    )
    parser.add_argument(
        "input",
        help="输入文件或目录路径"
    )
    parser.add_argument(
        "output_dir",
        help="输出目录路径（每个 chunk 保存为单独文件）"
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="文件匹配模式（目录模式时）"
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=32768,
        help="API 上下文窗口大小（Token）"
    )
    parser.add_argument(
        "--chunk-ratio",
        type=float,
        default=0.30,
        help="切片长度占窗口的比例"
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="文件编码"
    )

    args = parser.parse_args()

    config = ChunkerConfig(
        target_chunk_size=4000,
        overlap_ratio=0.50,
        output_mode="individual_files"
    )

    is_dir = os.path.isdir(args.input)

    chunker = SmartChunker(config)

    if is_dir:
        stats = chunk_directory_files(
            input_directory=args.input,
            output_dir=args.output_dir,
            pattern=args.pattern,
            config=config
        )
    else:
        stats = chunk_text_file(
            input_file=args.input,
            output_dir=args.output_dir,
            config=config
        )

    print(f"\n{'='*50}")
    print("切片完成！")
    print(f"{'='*50}")
    print(f"处理文件数: {stats['total_files']}")
    print(f"生成切片数: {stats['total_chunks']}")
    print(f"总字符数: {stats['total_chars']:,}")
    print(f"总 Token 数: {stats['total_tokens']:,}")
    print(f"输出目录: {args.output_dir}")
    print(f"索引文件: {stats['manifest_file']}")


if __name__ == "__main__":
    main()
