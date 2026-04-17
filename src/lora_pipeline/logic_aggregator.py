"""
Program 4: 逻辑焊接与去重 (Logic Aggregator)
结合向量搜索和 API，实现概念融合和 [待补全] 标签的自动替换。
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ============================================================
# 依赖检查和可选导入
# ============================================================

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Concept:
    """概念数据结构"""
    concept_id: str
    chunk_id: str  # 原始 chunk_id，贯穿整个流程
    concept: str
    definition: str
    prerequisites: List[str]
    impact: str
    embedding: Optional[Any] = None  # 向量表示
    original_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_extraction(cls, data: Dict[str, Any]) -> "Concept":
        """从 Program 3 的提取结果创建 Concept"""
        return cls(
            concept_id=data.get("concept_id", str(uuid.uuid4())),
            chunk_id=data.get("chunk_id", ""),
            concept=data.get("concept", ""),
            definition=data.get("definition", ""),
            prerequisites=data.get("prerequisites", []),
            impact=data.get("impact", ""),
            original_data=data
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "concept_id": self.concept_id,
            "chunk_id": self.chunk_id,
            "concept": self.concept,
            "definition": self.definition,
            "prerequisites": self.prerequisites,
            "impact": self.impact,
            "original_data": self.original_data
        }

    def get_text_representation(self) -> str:
        """获取用于向量化的文本表示"""
        parts = [self.concept, self.definition, self.impact]
        if self.prerequisites:
            parts.append(" ".join(self.prerequisites))
        return " | ".join(parts)


@dataclass
class FusedConcept:
    """融合后的概念"""
    final_id: str
    original_chunk_ids: List[str]
    concept: str
    definition: str
    prerequisites: List[str]
    impact: str
    welding_source: Dict[str, str]
    resolved_gaps: List[Dict[str, str]]
    unresolved_gaps: List[Dict[str, str]]
    quality_score: float
    confidence: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_id": self.final_id,
            "original_chunk_ids": self.original_chunk_ids,
            "concept": self.concept,
            "definition": self.definition,
            "prerequisites": self.prerequisites,
            "impact": self.impact,
            "welding_source": self.welding_source,
            "resolved_gaps": self.resolved_gaps,
            "unresolved_gaps": self.unresolved_gaps,
            "quality_score": self.quality_score,
            "confidence": self.confidence
        }


# ============================================================
# 工具函数
# ============================================================

def extract_pending_placeholders(text: str) -> List[str]:
    """从文本中提取所有 [待补全] 标记"""
    pattern = r'\[待补全\]\s*([^\]]+)'
    matches = re.findall(pattern, text)
    return [f"[待补全] {m.strip()}" for m in matches]


def has_pending_placeholder(text: str) -> bool:
    """检查文本是否包含 [待补全] 标记"""
    return "[待补全]" in text


def clean_pending_placeholder(gap: str) -> str:
    """清理 [待补全] 标记，获取实际内容"""
    if "[待补全]" in gap:
        return gap.replace("[待补全]", "").strip()
    return gap.strip()


# ============================================================
# 向量相似度服务
# ============================================================

class VectorSimilarityService:
    """向量相似度服务"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required for vector similarity. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self.embeddings = {}

    def encode(self, texts: List[str], show_progress: bool = False) -> List[Any]:
        """编码文本为向量"""
        return self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        embeddings = self.encode([text1, text2])
        from numpy.linalg import norm

        e1, e2 = embeddings[0], embeddings[1]
        cos_sim = np.dot(e1, e2) / (norm(e1) * norm(e2))
        return float(cos_sim)

    def find_similar_pairs(
        self,
        concepts: List[Concept],
        threshold: float = 0.9
    ) -> List[Tuple[Concept, Concept, float]]:
        """找到所有相似度超过阈值概念对"""
        if len(concepts) < 2:
            return []

        texts = [c.get_text_representation() for c in concepts]
        embeddings = self.encode(texts)

        pairs = []
        n = len(concepts)

        from numpy.linalg import norm

        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = embeddings[i], embeddings[j]
                cos_sim = np.dot(e1, e2) / (norm(e1) * norm(e2))

                if cos_sim >= threshold:
                    pairs.append((concepts[i], concepts[j], float(cos_sim)))

        return pairs


# ============================================================
# 逻辑焊接器
# ============================================================

class LogicWelder:
    """
    逻辑焊接器
    负责融合相似概念，修复 [待补全] 标记
    """

    def __init__(
        self,
        ai_provider=None,  # 可选，用于调用 API 进行融合
        similarity_threshold: float = 0.9
    ):
        self.similarity_threshold = similarity_threshold
        self.ai_provider = ai_provider
        self.vector_service = None

        if HAS_SENTENCE_TRANSFORMERS:
            self.vector_service = VectorSimilarityService()

    def fuse_concepts(
        self,
        concept_a: Concept,
        concept_b: Concept,
        fusion_prompt: Optional[str] = None
    ) -> FusedConcept:
        """
        融合两个相似概念

        Args:
            concept_a: 概念 A
            concept_b: 概念 B
            fusion_prompt: 可选的融合 prompt

        Returns:
            FusedConcept
        """
        # 默认融合逻辑（如果不使用 API）
        fused_def = self._merge_definitions(
            concept_a.definition,
            concept_b.definition
        )

        fused_impact = self._merge_impacts(
            concept_a.impact,
            concept_b.impact
        )

        fused_prereqs = self._merge_prerequisites(
            concept_a.prerequisites,
            concept_b.prerequisites
        )

        # 解析并尝试修复 [待补全]
        resolved_gaps = []
        unresolved_gaps = []

        for gap in fused_prereqs:
            if has_pending_placeholder(gap):
                # 尝试修复
                resolution = self._try_resolve_gap(gap, concept_a, concept_b)
                if resolution:
                    resolved_gaps.append({
                        "original_gap": gap,
                        "resolved_with": resolution,
                        "resolution_method": "cross_chunk_lookup"
                    })
                    fused_prereqs = [p for p in fused_prereqs if p != gap]
                    fused_prereqs.append(resolution)
                else:
                    unresolved_gaps.append({
                        "gap": gap,
                        "reason": "无法在相关概念中找到对应定义"
                    })

        return FusedConcept(
            final_id=str(uuid.uuid4()),
            original_chunk_ids=[concept_a.chunk_id, concept_b.chunk_id],
            concept=concept_a.concept,  # 保留原始概念名
            definition=fused_def,
            prerequisites=fused_prereqs,
            impact=fused_impact,
            welding_source={
                "concept_a": concept_a.concept,
                "concept_b": concept_b.concept,
                "fusion_method": "semantic_merge"
            },
            resolved_gaps=resolved_gaps,
            unresolved_gaps=unresolved_gaps,
            quality_score=0.8,
            confidence="medium"
        )

    def _merge_definitions(self, def_a: str, def_b: str) -> str:
        """合并两个定义"""
        if def_a == def_b:
            return def_a

        # 简单策略：按长度合并，保留更完整的
        if len(def_a) >= len(def_b):
            main, supplementary = def_a, def_b
        else:
            main, supplementary = def_b, def_a

        # 如果较短定义被较长定义包含，直接返回长的
        if main in supplementary or supplementary in main:
            return main

        # 否则拼接（去除重复句子）
        sentences_a = set(def_a.split('。'))
        sentences_b = set(def_b.split('。'))

        unique_sentences = sentences_a.union(sentences_b)
        merged = '。'.join(s for s in unique_sentences if s.strip())

        # 确保以句号结尾
        if merged and not merged.endswith('。'):
            merged += '。'

        return merged

    def _merge_impacts(self, impact_a: str, impact_b: str) -> str:
        """合并两个影响描述"""
        if impact_a == impact_b:
            return impact_a

        if not impact_a:
            return impact_b
        if not impact_b:
            return impact_a

        # 合并不同的影响
        if impact_b not in impact_a:
            return f"{impact_a} {impact_b}"

        return impact_a

    def _merge_prerequisites(
        self,
        prereqs_a: List[str],
        prereqs_b: List[str]
    ) -> List[str]:
        """合并前置知识"""
        result = []

        for prereq in prereqs_a + prereqs_b:
            if prereq not in result:
                result.append(prereq)

        return result

    def _try_resolve_gap(
        self,
        gap: str,
        concept_a: Concept,
        concept_b: Concept
    ) -> Optional[str]:
        """
        尝试修复 [待补全] 标记
        在相关概念的定义中查找可以填补的内容
        """
        gap_content = clean_pending_placeholder(gap)

        # 在两个概念的 definition 和 impact 中搜索
        search_targets = [
            concept_a.definition,
            concept_a.impact,
            concept_b.definition,
            concept_b.impact
        ]

        for target in search_targets:
            # 简单的关键词匹配
            gap_keywords = set(gap_content.lower().split())

            # 检查 gap 关键词是否在 target 中
            target_lower = target.lower()
            matches = [w for w in gap_keywords if w in target_lower]

            if len(matches) >= len(gap_keywords) * 0.5:  # 50% 匹配
                # 提取包含关键词的句子作为替代
                sentences = target.split('。')
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(w in sentence_lower for w in gap_keywords):
                        return sentence.strip()

        return None

    def deduplicate_concepts(
        self,
        concepts: List[Concept]
    ) -> Tuple[List[Concept], List[Dict[str, Any]]]:
        """
        去重概念
        返回：(去重后的概念, 合并记录)
        """
        if len(concepts) < 2:
            return concepts, []

        if not self.vector_service:
            return concepts, []

        duplicate_groups = []
        used_indices = set()
        unique_concepts = []

        # 计算相似度
        texts = [c.get_text_representation() for c in concepts]
        embeddings = self.vector_service.encode(texts)

        from numpy.linalg import norm
        import numpy as np

        for i in range(len(concepts)):
            if i in used_indices:
                continue

            current_group = [i]

            for j in range(i + 1, len(concepts)):
                if j in used_indices:
                    continue

                e1, e2 = embeddings[i], embeddings[j]
                cos_sim = np.dot(e1, e2) / (norm(e1) * norm(e2))

                if cos_sim >= self.similarity_threshold:
                    current_group.append(j)
                    used_indices.add(j)

            # 选择最佳代表（定义最长的）
            if len(current_group) > 1:
                best_idx = max(current_group, key=lambda x: len(concepts[x].definition))
                current_group.remove(best_idx)
                used_indices.update(current_group)

                duplicate_groups.append({
                    "representative_id": concepts[best_idx].concept_id,
                    "merged_ids": [concepts[idx].concept_id for idx in current_group],
                    "reason": f"Merged {len(current_group)} duplicate concepts"
                })

            unique_concepts.append(concepts[best_idx])

        return unique_concepts, duplicate_groups


# ============================================================
# 逻辑聚合器（主类）
# ============================================================

class LogicAggregator:
    """
    逻辑聚合器 - Program 4 主类
    协调向量搜索、焊接、去重和 gap 修复
    """

    def __init__(
        self,
        ai_provider=None,
        similarity_threshold: float = 0.9,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.similarity_threshold = similarity_threshold
        self.ai_provider = ai_provider
        self.vector_service = None
        self.welder = LogicWelder(
            ai_provider=ai_provider,
            similarity_threshold=similarity_threshold
        )

        if HAS_SENTENCE_TRANSFORMERS:
            self.vector_service = VectorSimilarityService(model_name)

    def load_extractions(self, input_file: str) -> List[Concept]:
        """加载 Program 3 的提取结果"""
        concepts = []

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                concepts_data = data.get("concepts", [])

                for concept_data in concepts_data:
                    concept = Concept.from_extraction(concept_data)
                    concept.chunk_id = data.get("chunk_id", concept.chunk_id)
                    concepts.append(concept)

        logger.info(f"Loaded {len(concepts)} concepts from {input_file}")
        return concepts

    def aggregate(
        self,
        concepts: List[Concept],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        执行完整的聚合流程

        Returns:
            聚合结果字典
        """
        original_count = len(concepts)

        # Step 1: 去重
        if self.vector_service:
            logger.info("Step 1: Deduplicating concepts...")
            concepts, duplicate_groups = self.welder.deduplicate_concepts(concepts)
            logger.info(f"  After deduplication: {len(concepts)} concepts")
        else:
            duplicate_groups = []

        # Step 2: 相似概念焊接
        fused_concepts = []
        if self.vector_service and len(concepts) >= 2:
            logger.info("Step 2: Welding similar concepts...")

            texts = [c.get_text_representation() for c in concepts]
            embeddings = self.vector_service.encode(texts, show_progress=show_progress)

            import numpy as np
            from numpy.linalg import norm

            used = set()

            if show_progress:
                concepts_iter = tqdm(range(len(concepts)), desc="Welding concepts")
            else:
                concepts_iter = range(len(concepts))

            for i in concepts_iter:
                if i in used:
                    continue

                current_group = [i]

                for j in range(i + 1, len(concepts)):
                    if j in used:
                        continue

                    e1, e2 = embeddings[i], embeddings[j]
                    cos_sim = np.dot(e1, e2) / (norm(e1) * norm(e2))

                    if cos_sim >= self.similarity_threshold:
                        current_group.append(j)
                        used.add(j)

                # 融合同组概念
                if len(current_group) > 1:
                    fused = concepts[current_group[0]]
                    for idx in current_group[1:]:
                        fused = self.welder.fuse_concepts(fused, concepts[idx])
                    fused_concepts.append(fused)
                    used.add(i)
                else:
                    # 单个概念直接保留
                    fused_concepts.append(concepts[i])
                    used.add(i)

        else:
            fused_concepts = [
                self._concept_to_fused(c) for c in concepts
            ]

        # Step 3: 统计
        all_resolved_gaps = sum(len(f.resolved_gaps) for f in fused_concepts)
        all_unresolved_gaps = sum(len(f.unresolved_gaps) for f in fused_concepts)

        statistics = {
            "original_concepts": original_count,
            "after_deduplication": len(concepts),
            "after_welding": len(fused_concepts),
            "resolved_gaps": all_resolved_gaps,
            "unresolved_gaps": all_unresolved_gaps,
            "deduplication_rate": (original_count - len(concepts)) / original_count if original_count > 0 else 0,
            "welding_rate": len(fused_concepts) / original_count if original_count > 0 else 0,
            "gap_resolution_rate": all_resolved_gaps / (all_resolved_gaps + all_unresolved_gaps) if (all_resolved_gaps + all_unresolved_gaps) > 0 else 0
        }

        return {
            "aggregated_concepts": fused_concepts,
            "duplicate_groups": duplicate_groups,
            "final_statistics": statistics
        }

    def _concept_to_fused(self, concept: Concept) -> FusedConcept:
        """将单个 Concept 转换为 FusedConcept"""
        resolved_gaps = []
        unresolved_gaps = []
        clean_prereqs = []

        for prereq in concept.prerequisites:
            if has_pending_placeholder(prereq):
                unresolved_gaps.append({
                    "gap": prereq,
                    "reason": "无相关概念可填补"
                })
            else:
                clean_prereqs.append(prereq)

        return FusedConcept(
            final_id=concept.concept_id,
            original_chunk_ids=[concept.chunk_id],
            concept=concept.concept,
            definition=concept.definition,
            prerequisites=clean_prereqs,
            impact=concept.impact,
            welding_source={
                "concept_a": concept.concept,
                "concept_b": None,
                "fusion_method": "single_concept"
            },
            resolved_gaps=resolved_gaps,
            unresolved_gaps=unresolved_gaps,
            quality_score=0.5,
            confidence="low"
        )

    def save_output(
        self,
        result: Dict[str, Any],
        output_file: str
    ) -> None:
        """保存最终输出为 .jsonl 格式"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for fused in result["aggregated_concepts"]:
                f.write(json.dumps(fused.to_dict(), ensure_ascii=False) + "\n")

        # 同时保存统计信息
        stats_path = output_path.with_suffix(".stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(result["final_statistics"], f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(result['aggregated_concepts'])} concepts to {output_file}")


# ============================================================
# 便捷函数
# ============================================================

def aggregate_logic_chains(
    input_file: str,
    output_file: str,
    similarity_threshold: float = 0.9
) -> Dict[str, Any]:
    """
    便捷函数：执行完整的逻辑聚合

    Args:
        input_file: Program 3 的输出文件
        output_file: 最终 .jsonl 输出路径
        similarity_threshold: 相似度阈值（默认 0.9）

    Returns:
        统计信息
    """
    aggregator = LogicAggregator(similarity_threshold=similarity_threshold)
    concepts = aggregator.load_extractions(input_file)
    result = aggregator.aggregate(concepts)
    aggregator.save_output(result, output_file)
    return result["final_statistics"]


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Logic Aggregator - 逻辑焊接与去重"
    )
    parser.add_argument(
        "input",
        help="Program 3 的提取结果文件（.jsonl）"
    )
    parser.add_argument(
        "output",
        help="输出文件路径（.jsonl）"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="相似度阈值（默认 0.9）"
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformer 模型名"
    )

    args = parser.parse_args()

    stats = aggregate_logic_chains(
        input_file=args.input,
        output_file=args.output,
        similarity_threshold=args.threshold
    )

    print(f"\n{'='*50}")
    print("逻辑聚合完成！")
    print(f"{'='*50}")
    print(f"原始概念数: {stats['original_concepts']}")
    print(f"去重后: {stats['after_deduplication']}")
    print(f"焊接后: {stats['after_welding']}")
    print(f"已修复 [待补全]: {stats['resolved_gaps']}")
    print(f"未修复 [待补全]: {stats['unresolved_gaps']}")
    print(f"去重率: {stats['deduplication_rate']:.2%}")
    print(f"Gap 修复率: {stats['gap_resolution_rate']:.2%}")


if __name__ == "__main__":
    main()
