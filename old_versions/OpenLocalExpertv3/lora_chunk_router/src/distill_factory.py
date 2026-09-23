"""
Contextual Distillation Factory - 高质量 QA 对数据工厂

功能：
- 异步批处理 LLM 调用（支持 OpenAI 兼容 API / 本地模型）
- 每个 Chunk 生成 4 种维度的问题：事实、模糊、推论、困难负样本
- Hard Negative 植入：修复 NOT_RELEVANT 检测率
- LCS 字面重合度过滤
- JSON 健壮性校验

使用方式：
    python -m src.distill_factory --mode distill --batch_size 4
    python -m src.distill_factory --mode filter
    python -m src.distill_factory --mode preview --num_samples 10
"""

import os
import sys
import json
import asyncio
import argparse
import re
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_json, save_json, save_jsonl, get_project_root, load_config


# ============================================================
# 核心 Prompt 模板 - 高价值数据合成
# ============================================================

DIVERSITY_SYSTEM_PROMPT = """You are an expert at generating diverse, high-quality training data for a RAG (Retrieval-Augmented Generation) system. Your task is to create challenging question-answer pairs that test deep semantic understanding.

CRITICAL REQUIREMENTS:
1. Questions must NOT be direct quotes or near-quotes from the text
2. Questions should require semantic understanding, not keyword matching
3. Each question should have a clearly relevant answer within the provided chunk
4. Generate exactly 4 questions per chunk, each from a DIFFERENT dimension:
   - [FACTUAL] Specific questions about facts/claims in the text
   - [FUZZY] Colloquial/vague questions that require semantic matching
   - [INFERENTIAL] Questions requiring reasoning about the content
   - [HARD_NEGATIVE] A question that SOUNDS related but is actually answerable by the chunk yet tests rejection capability

OUTPUT FORMAT - Return ONLY valid JSON:
{
  "questions": [
    {"type": "factual", "question": "...", "answer_hint": "..."},
    {"type": "fuzzy", "question": "...", "answer_hint": "..."},
    {"type": "inferenial", "question": "...", "answer_hint": "..."},
    {"type": "hard_negative", "question": "...", "answer_hint": "..."}
  ]
}

Key for HARD_NEGATIVE:
- Take a keyword from the text
- Craft a question that sounds relevant but is actually answerable by this specific chunk
- The question should test if the model can correctly match despite surface-level similarity
- Example: If text mentions "Freemasons", a hard_negative could ask about "secret societies in medieval Europe" (sounds related but different scope)

IMPORTANT: Return ONLY JSON. No markdown, no explanation, no preamble."""



HARD_NEGATIVE_SYSTEM_PROMPT = """You are an expert at creating hard negative samples for RAG training.

Given a text chunk, create a question that:
1. Uses keywords or concepts from the chunk
2. Sounds like it should be answered by the chunk
3. But is actually about a DIFFERENT specific topic that the chunk does NOT address

The question should be answerable in the same DOMAIN but about a DIFFERENT SPECIFIC subject.

Example:
- Chunk mentions: "The Freemasons are a secret society..."
- Hard Negative: "How did the Illuminati influence the French Revolution?" (sounds similar, different topic)

Return ONLY valid JSON:
{
  "hard_negative": {
    "question": "...",
    "why_related": "...",
    "why_different": "..."
  }
}"""



@dataclass
class DistillConfig:
    """蒸馏配置"""
    # API 配置
    api_base: str = "http://localhost:11434/v1"  # 默认本地 Ollama
    api_key: str = "not-needed"
    model_name: str = "qwen2.5:7b-instruct"
    max_tokens: int = 1024
    temperature: float = 0.8
    timeout: int = 120

    # 批处理配置
    batch_size: int = 4
    max_concurrent: int = 4
    retry_attempts: int = 3
    retry_delay: float = 2.0

    # 数据配置
    chunks_path: str = ""
    output_path: str = ""
    num_factual: int = 1
    num_fuzzy: int = 1
    num_inferential: int = 1
    num_hard_negative: int = 1

    # 过滤配置
    lcs_threshold: float = 0.70  # LCS ratio threshold
    min_question_len: int = 10
    max_question_len: int = 200

    # 平衡配置
    not_relevant_ratio: float = 0.18  # 15-20% NOT_RELEVANT samples


class AsyncLLMClient:
    """
    异步 LLM 客户端
    - 优先使用 subprocess 调用本地 ollama CLI（最稳定）
    - 也支持 OpenAI 兼容 API（通过 aiohttp）
    """

    def __init__(self, config: DistillConfig):
        self.config = config
        self._use_ollama_cli = True  # 默认用 CLI

    def _call_ollama_cli(self, prompt: str, system: str = "") -> str:
        """
        使用 subprocess 调用 ollama CLI（同步，最稳定）
        """
        import subprocess
        import json

        # 构建 messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        cmd = [
            "ollama", "run", self.config.model_name,
            "--verbose",  # 显示更多输出
        ]

        # 用 --format json 如果支持，否则解析原始输出
        # ollama CLI 支持 --format 参数
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        try:
            result = subprocess.run(
                ["ollama", "generate", "--json", "-p", json.dumps(payload)],
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                raise Exception(f"ollama error: {result.stderr}")
        except FileNotFoundError:
            # ollama CLI 不存在，fallback 到 HTTP
            self._use_ollama_cli = False
            raise Exception("ollama CLI not found")

    async def _call_ollama_http(self, prompt: str, system: str = "") -> str:
        """使用 aiohttp HTTP 调用"""
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.config.model_name,
                "messages": [],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
            if system:
                payload["messages"].append({"role": "system", "content": system})
            payload["messages"].append({"role": "user", "content": prompt})

            for attempt in range(self.config.retry_attempts):
                try:
                    async with session.post(
                        f"{self.config.api_base}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            message = data["choices"][0]["message"]
                            content = message.get("content", "")
                            # Handle Qwen's reasoning models
                            if not content and message.get("reasoning"):
                                content = message["reasoning"]
                            return content
                        elif resp.status == 429:
                            await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                            continue
                        else:
                            error_text = await resp.text()
                            raise Exception(f"API error {resp.status}: {error_text}")
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
            raise Exception("Max retries exceeded")

    async def generate(self, prompt: str, system: str = "") -> str:
        """单次生成调用"""
        if self._use_ollama_cli:
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._call_ollama_cli, prompt, system)
            except Exception as e:
                if "ollama CLI not found" in str(e):
                    pass  # Fall through to HTTP
                else:
                    raise

        # Fallback to HTTP
        return await self._call_ollama_http(prompt, system)

    async def batch_generate(self, prompts: List[Tuple[str, str]]) -> List[str]:
        """批量生成"""
        sem = asyncio.Semaphore(self.config.max_concurrent)

        async def generate_one(prompt_sys: Tuple[str, str]) -> str:
            async with sem:
                prompt, system = prompt_sys
                return await self.generate(prompt, system)

        tasks = [generate_one(ps) for ps in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self):
        """关闭（CLI 模式不需要）"""
        pass


def compute_lcs_ratio(text1: str, text2: str) -> float:
    """
    计算两个文本的最长公共子序列（LCS）与 text1 的长度比值
    用于检测字面重合度
    """
    if not text1 or not text2:
        return 0.0

    # 简单实现：分词后求 LCS
    words1 = text1.lower().split()
    words2 = text2.lower().split()

    m, n = len(words1), len(words2)
    if m == 0 or n == 0:
        return 0.0

    # 空间优化的 LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    lcs_length = prev[n]
    return lcs_length / m


def extract_keywords_from_text(text: str, num_keywords: int = 5) -> List[str]:
    """
    从文本中提取关键词（用于 Hard Negative 生成）
    排除停用词，优先选择名词和专有名词
    """
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'he', 'she', 'it', 'they', 'we', 'i',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'to', 'of', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'up', 'down', 'about', 'into', 'over',
        'after', 'just', 'like', 'this', 'that', 'there', 'here', 'when', 'where', 'how', 'why',
        'can', 'will', 'should', 'what', 'who', 'which', 'not', 'all', 'some', 'any', 'each',
        'very', 'really', 'actually', 'basically', 'okay', 'right', 'so', 'well', 'now', 'then',
        'because', 'would', 'could', 'may', 'might', 'must', 'shall', 'will', 'going', 'want',
        'know', 'think', 'believe', 'see', 'make', 'get', 'come', 'say', 'tell', 'give', 'take',
    }

    # 提取 3+ 字符的单词
    words = re.findall(r'\b[A-Za-z]{3,}\b', text)
    candidates = []

    for word in words:
        low = word.lower()
        if low in STOPWORDS:
            continue
        # 专有名词加权
        score = 2 if word[0].isupper() else 1
        candidates.append((word, score))

    # 按分数排序，取 top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    unique = []
    seen = set()
    for word, _ in candidates:
        if word.lower() not in seen:
            unique.append(word)
            seen.add(word.lower())
        if len(unique) >= num_keywords:
            break

    return unique


def parse_llm_json_response(response: str) -> Optional[Dict]:
    """
    解析 LLM 返回的 JSON，包含健壮性处理
    """
    # 去掉可能的 markdown 代码块
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    # 去掉开头的废话（如 "Here is the JSON..."）
    text = re.sub(r"^(?:Here(?:'s| is))?\s*(?:the)?\s*JSON[:\s]*", "", text, flags=re.IGNORECASE)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试修复常见问题
        text = text.replace("'", '"')  # 单引号变双引号
        text = text.replace(",}", "}")  # 尾部多余逗号
        text = text.replace(",]", "]")  # 尾部多余逗号
        try:
            return json.loads(text)
        except:
            return None


def build_distill_prompt(chunk: Dict) -> str:
    """为单个 Chunk 构建蒸馏 prompt"""
    chunk_id = chunk["chunk_id"]
    text = chunk["text"][:1500]  # 截断太长的文本

    prompt = f"""Given this transcript chunk (ID: {chunk_id}):

---
{text}
---

Generate exactly 4 diverse questions that test semantic understanding:

1. FACTUAL: A specific question about facts/claims in the text
2. FUZZY: A colloquial/vague question that requires semantic matching  
3. INFERENTIAL: A question requiring reasoning about the content
4. HARD_NEGATIVE: A question that uses keywords from the text but is about a DIFFERENT specific topic (sounds related but different scope)

Return ONLY valid JSON with "questions" array containing 4 objects with keys: type, question, answer_hint"""

    return prompt


def build_hard_negative_prompt(chunk: Dict) -> str:
    """为 Hard Negative 专门构建 prompt"""
    keywords = extract_keywords_from_text(chunk["text"], num_keywords=8)
    text = chunk["text"][:1000]

    prompt = f"""Text chunk: {chunk_id}
---
{text}
---

Available keywords from text: {', '.join(keywords[:5])}

Create a HARD NEGATIVE question:
- Must use at least one keyword from the text
- Must sound like it could be answered by this chunk
- But must actually be about a DIFFERENT specific topic

Example:
- Text: "The Freemasons are a secret society founded in..."
- Hard Negative: "How did the Illuminati influence the French Revolution?" (different topic, same domain)

Return ONLY valid JSON:
{{"hard_negative": {{"question": "...", "why_related": "...", "why_different": "..."}}}}"""

    return prompt


class DistillFactory:
    """蒸馏工厂 - 完整的数据生成流水线"""

    def __init__(self, config: DistillConfig):
        self.config = config
        self.client = AsyncLLMClient(config)
        self.stats = {
            "total_chunks": 0,
            "successful": 0,
            "failed": 0,
            "filtered_lcs": 0,
            "filtered_json": 0,
            "types_generated": defaultdict(int),
        }
        self.results: List[Dict] = []

    async def process_chunk(self, chunk: Dict) -> List[Dict]:
        """
        处理单个 Chunk，生成 4 种维度的 QA 对
        """
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]

        try:
            # 构建 prompt
            prompt = build_distill_prompt(chunk)

            # 调用 LLM
            response = await self.client.generate(
                prompt=prompt,
                system=DIVERSITY_SYSTEM_PROMPT
            )

            # 解析 JSON
            parsed = parse_llm_json_response(response)
            if not parsed or "questions" not in parsed:
                print(f"  [WARN] {chunk_id}: Failed to parse JSON response")
                self.stats["failed"] += 1
                return []

            questions = parsed["questions"]
            samples = []

            for q in questions:
                q_type = q.get("type", "unknown").lower()
                question = q.get("question", "").strip()

                # 长度过滤
                if len(question) < self.config.min_question_len:
                    continue
                if len(question) > self.config.max_question_len:
                    question = question[:self.config.max_question_len]

                # LCS 字面重合度过滤（针对 factual 类型）
                if q_type == "factual":
                    lcs_ratio = compute_lcs_ratio(question, text)
                    if lcs_ratio > self.config.lcs_threshold:
                        print(f"  [FILTER] {chunk_id} ({q_type}): LCS={lcs_ratio:.2f} too high")
                        self.stats["filtered_lcs"] += 1
                        continue

                # 标签决定
                if q_type == "hard_negative":
                    label = "NOT_RELEVANT"
                else:
                    label = chunk_id

                samples.append({
                    "instruction": question,
                    "output": label,
                    "chunk_id": chunk_id,
                    "type": q_type,
                    "answer_hint": q.get("answer_hint", ""),
                })

                self.stats["types_generated"][q_type] += 1

            if samples:
                self.stats["successful"] += 1
            else:
                self.stats["failed"] += 1

            return samples

        except Exception as e:
            print(f"  [ERROR] {chunk_id}: {str(e)[:100]}")
            self.stats["failed"] += 1
            return []

    async def process_batch(self, chunks: List[Dict]) -> List[List[Dict]]:
        """
        批量处理多个 Chunk
        """
        # 构建批量 prompts
        prompts = [(build_distill_prompt(c), DIVERSITY_SYSTEM_PROMPT) for c in chunks]

        # 批量调用
        responses = await self.client.batch_generate(prompts)

        all_samples = []
        for chunk, response in zip(chunks, responses):
            if isinstance(response, Exception):
                print(f"  [ERROR] {chunk['chunk_id']}: {str(response)[:80]}")
                self.stats["failed"] += 1
                all_samples.append([])
                continue

            parsed = parse_llm_json_response(response)
            if not parsed or "questions" not in parsed:
                self.stats["failed"] += 1
                all_samples.append([])
                continue

            samples = []
            for q in parsed["questions"]:
                q_type = q.get("type", "unknown").lower()
                question = q.get("question", "").strip()

                if len(question) < self.config.min_question_len:
                    continue

                # LCS 过滤
                if q_type == "factual":
                    lcs_ratio = compute_lcs_ratio(question, chunk["text"])
                    if lcs_ratio > self.config.lcs_threshold:
                        self.stats["filtered_lcs"] += 1
                        continue

                label = "NOT_RELEVANT" if q_type == "hard_negative" else chunk["chunk_id"]

                samples.append({
                    "instruction": question,
                    "output": label,
                    "chunk_id": chunk["chunk_id"],
                    "type": q_type,
                })
                self.stats["types_generated"][q_type] += 1

            if samples:
                self.stats["successful"] += 1
            all_samples.append(samples)

        return all_samples

    async def run_distillation(self, chunks: List[Dict], progress_interval: int = 50) -> List[Dict]:
        """
        运行完整蒸馏流程
        """
        print(f"\n{'='*60}")
        print(f"Starting Distillation: {len(chunks)} chunks")
        print(f"Batch size: {self.config.batch_size}, Concurrency: {self.config.max_concurrent}")
        print(f"Model: {self.config.model_name}")
        print(f"{'='*60}\n")

        self.stats["total_chunks"] = len(chunks)
        self.results = []
        all_samples = []

        # 分批处理
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]

            batch_samples = await self.process_batch(batch)
            for samples in batch_samples:
                all_samples.extend(samples)

            # 进度报告
            processed = min(i + self.config.batch_size, len(chunks))
            if processed % progress_interval == 0 or processed == len(chunks):
                print(f"  Progress: {processed}/{len(chunks)} chunks, "
                      f"Generated: {len(all_samples)} samples, "
                      f"Failed: {self.stats['failed']}")

            # 避免过快
            await asyncio.sleep(0.1)

        self.results = all_samples

        # 打印统计
        print(f"\n{'='*60}")
        print("Distillation Complete!")
        print(f"{'='*60}")
        print(f"  Total chunks processed: {self.stats['total_chunks']}")
        print(f"  Successful: {self.stats['successful']}")
        print(f"  Failed: {self.stats['failed']}")
        print(f"  Filtered (LCS): {self.stats['filtered_lcs']}")
        print(f"  Total samples generated: {len(all_samples)}")
        print(f"\n  By type:")
        for qtype, count in sorted(self.stats["types_generated"].items()):
            print(f"    {qtype}: {count}")

        return all_samples

    async def run_hard_negative_enrichment(self, chunks: List[Dict]) -> List[Dict]:
        """
        专门为 NOT_RELEVANT 标签生成高质量困难负样本
        这是修复 NOT_RELEVANT 检测率的关键
        """
        print(f"\n{'='*60}")
        print("Generating Hard Negatives for NOT_RELEVANT...")
        print(f"{'='*60}\n")

        hard_negatives = []

        # 对每个 chunk 生成 2-3 个 hard negatives
        for i, chunk in enumerate(chunks):
            keywords = extract_keywords_from_text(chunk["text"], num_keywords=5)
            text_snippet = chunk["text"][:800]

            prompt = f"""Text from lecture chunk {chunk['chunk_id']}:
---
{text_snippet}
---

Keywords from this chunk: {', '.join(keywords)}

Generate 2 hard negative questions that:
1. Use 1-2 keywords from the text
2. Sound like they could be about the same topic
3. But are actually about a DIFFERENT specific subject (answerable elsewhere)

Example:
- Chunk mentions "Freemasons" → Hard negative: "When were the Knights Templar officially dissolved?"
- Chunk mentions "moon landing" → Hard negative: "Who was the first human to orbit Earth?"

Return ONLY valid JSON:
{{"hard_negatives": [
  {{"question": "...", "keyword_used": "...", "actual_topic": "..."}},
  {{"question": "...", "keyword_used": "...", "actual_topic": "..."}}
]}}"""

            try:
                response = await self.client.generate(prompt, system=HARD_NEGATIVE_SYSTEM_PROMPT)
                parsed = parse_llm_json_response(response)

                if parsed and "hard_negatives" in parsed:
                    for hn in parsed["hard_negatives"]:
                        question = hn.get("question", "").strip()
                        if len(question) >= 10:
                            hard_negatives.append({
                                "instruction": question,
                                "output": "NOT_RELEVANT",
                                "chunk_id": chunk["chunk_id"],
                                "type": "hard_negative",
                                "keyword_used": hn.get("keyword_used", ""),
                                "actual_topic": hn.get("actual_topic", ""),
                            })
            except Exception as e:
                print(f"  [WARN] Hard negative failed for {chunk['chunk_id']}: {str(e)[:60]}")

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(chunks)} chunks, HN collected: {len(hard_negatives)}")

        print(f"\n  Hard negatives generated: {len(hard_negatives)}")
        return hard_negatives


class DistillFilter:
    """数据清洗与校验"""

    def __init__(self, config: DistillConfig):
        self.config = config
        self.stats = {
            "total": 0,
            "kept": 0,
            "removed_lcs": 0,
            "removed_json": 0,
            "removed_length": 0,
            "removed_dupe": 0,
            "balance_not_relevant": 0,
        }

    def compute_lcs_ratio(self, text1: str, text2: str) -> float:
        return compute_lcs_ratio(text1, text2)

    def check_label_balance(self, samples: List[Dict], target_not_relevant_ratio: float = 0.18) -> List[Dict]:
        """
        检查并平衡标签分布
        确保 NOT_RELEVANT 占总数据集的 15-20%
        """
        total = len(samples)
        not_relevant_count = sum(1 for s in samples if s["output"] == "NOT_RELEVANT")

        current_ratio = not_relevant_count / total if total > 0 else 0
        print(f"\n  Label balance: NOT_RELEVANT={current_ratio:.1%} (target: {target_not_relevant_ratio:.0%})")

        # 如果 NOT_RELEVANT 不足，需要补充
        if current_ratio < target_not_relevant_ratio:
            target_count = int(total * target_not_relevant_ratio)
            deficit = target_count - not_relevant_count

            if deficit > 0:
                # 从现有 hard_negative 中补充（如果之前没有标记的话）
                # 这里简单处理：复制一些 hard_negative 作为 NOT_RELEVANT
                existing_hn = [s for s in samples if s.get("type") == "hard_negative"]
                if existing_hn:
                    extra = random.sample(existing_hn, min(deficit, len(existing_hn)))
                    # 修改标签
                    for s in extra:
                        s["output"] = "NOT_RELEVANT"
                    samples.extend(extra)
                    self.stats["balance_not_relevant"] = len(extra)
                    print(f"  Added {len(extra)} NOT_RELEVANT samples from hard_negatives")

        return samples

    def filter_samples(self, samples: List[Dict], chunks_map: Dict[str, Dict]) -> List[Dict]:
        """
        完整过滤流水线
        """
        print(f"\n{'='*60}")
        print("Running Data Filtering Pipeline")
        print(f"{'='*60}")

        self.stats["total"] = len(samples)
        filtered = []
        seen_questions = set()

        for sample in samples:
            question = sample["instruction"]

            # 1. JSON 健壮性（已经在生成阶段处理）
            if not question or len(question.strip()) < 5:
                self.stats["removed_length"] += 1
                continue

            # 2. 长度过滤
            if len(question) < self.config.min_question_len or len(question) > self.config.max_question_len:
                self.stats["removed_length"] += 1
                continue

            # 3. 重复问题过滤
            q_hash = hashlib.md5(question.lower().encode()).hexdigest()
            if q_hash in seen_questions:
                self.stats["removed_dupe"] += 1
                continue
            seen_questions.add(q_hash)

            # 4. LCS 字面重合度（针对 factual 类型）
            if sample.get("type") == "factual" and sample["chunk_id"] in chunks_map:
                text = chunks_map[sample["chunk_id"]]["text"]
                lcs_ratio = self.compute_lcs_ratio(question, text)
                if lcs_ratio > self.config.lcs_threshold:
                    self.stats["removed_lcs"] += 1
                    continue

            # 5. 标签平衡
            # (在单独步骤中处理)

            filtered.append(sample)
            self.stats["kept"] += 1

        # 标签平衡
        filtered = self.check_label_balance(filtered, self.config.not_relevant_ratio)

        # 统计
        print(f"\n  Filtering Results:")
        print(f"    Total input: {self.stats['total']}")
        print(f"    Kept: {self.stats['kept']}")
        print(f"    Removed (LCS): {self.stats['removed_lcs']}")
        print(f"    Removed (length): {self.stats['removed_length']}")
        print(f"    Removed (duplicate): {self.stats['removed_dupe']}")
        print(f"    Added (balance): {self.stats['balance_not_relevant']}")

        # 按类型统计
        type_counts = defaultdict(int)
        for s in filtered:
            type_counts[s.get("type", "unknown")] += 1
        print(f"\n  Final type distribution:")
        for t, c in sorted(type_counts.items()):
            print(f"    {t}: {c}")

        # 标签分布
        label_counts = defaultdict(int)
        for s in filtered:
            label_counts[s["output"]] += 1
        not_rel = label_counts.get("NOT_RELEVANT", 0)
        print(f"\n  NOT_RELEVANT: {not_rel}/{len(filtered)} = {not_rel/len(filtered):.1%}")

        return filtered


async def run_distill_mode(args):
    """蒸馏模式 - 生成 train_distilled.jsonl"""
    root = get_project_root()

    # 加载配置
    config = DistillConfig()

    # 加载 chunks
    chunks_path = args.chunks or str(root / "experiments" / "chunks.json")
    chunks = load_json(chunks_path)
    if isinstance(chunks, dict):
        chunks = chunks.get("chunks", chunks.get("data", []))

    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # 如果有测试模式限制
    if args.num_test > 0:
        chunks = chunks[:args.num_test]
        print(f"TEST MODE: Only processing first {args.num_test} chunks")

    # API 配置
    if args.api_base:
        config.api_base = args.api_base
    if args.model:
        config.model_name = args.model
    config.batch_size = args.batch_size
    config.max_concurrent = args.concurrency

    # 输出路径
    output_path = args.output or str(root / "data" / "train_distilled.jsonl")

    # 创建工厂并运行
    factory = DistillFactory(config)

    # 主要蒸馏
    samples = await factory.run_distillation(chunks)

    # 额外生成 Hard Negatives
    hn_samples = await factory.run_hard_negative_enrichment(chunks[:min(100, len(chunks))])
    samples.extend(hn_samples)

    # 过滤
    if samples:
        # 构建 chunks map 用于 LCS 检查
        chunks_map = {c["chunk_id"]: c for c in chunks}
        filter_pipeline = DistillFilter(config)
        samples = filter_pipeline.filter_samples(samples, chunks_map)

    # 打乱并保存
    random.shuffle(samples)
    save_jsonl(samples, output_path)
    print(f"\n✅ Saved {len(samples)} samples to {output_path}")

    # 关闭客户端
    await factory.client.close()


async def run_filter_mode(args):
    """过滤模式 - 对已生成的蒸馏数据进行清洗"""
    root = get_project_root()

    input_path = args.input or str(root / "data" / "train_distilled_raw.jsonl")
    output_path = args.output or str(root / "data" / "train_distilled.jsonl")

    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            try:
                samples.append(json.loads(line.strip()))
            except:
                continue

    print(f"Loaded {len(samples)} samples from {input_path}")

    # 加载 chunks 用于 LCS 检查
    chunks_path = root / "experiments" / "chunks.json"
    chunks = load_json(str(chunks_path))
    if isinstance(chunks, dict):
        chunks = chunks.get("chunks", chunks.get("data", []))
    chunks_map = {c["chunk_id"]: c for c in chunks}

    # 过滤
    config = DistillConfig()
    filter_pipeline = DistillFilter(config)
    filtered = filter_pipeline.filter_samples(samples, chunks_map)

    # 打乱并保存
    random.shuffle(filtered)
    save_jsonl(filtered, output_path)
    print(f"\n✅ Saved {len(filtered)} filtered samples to {output_path}")


async def run_preview_mode(args):
    """预览模式 - 测试少量 Chunk 的蒸馏效果"""
    root = get_project_root()

    # 加载配置
    config = DistillConfig()
    if args.api_base:
        config.api_base = args.api_base
    if args.model:
        config.model_name = args.model
    config.max_concurrent = 1

    # 加载 chunks
    chunks_path = args.chunks or str(root / "experiments" / "chunks.json")
    chunks = load_json(chunks_path)
    if isinstance(chunks, dict):
        chunks = chunks.get("chunks", chunks.get("data", []))

    num = args.num_samples or 5
    test_chunks = chunks[:num]

    print(f"Preview mode: Testing {num} chunks with model {config.model_name}")

    factory = DistillFactory(config)

    for chunk in test_chunks:
        print(f"\n{'='*60}")
        print(f"Chunk: {chunk['chunk_id']} - {chunk.get('lecture_title', '')[:50]}")
        print(f"{'='*60}")
        print(f"Text (first 300 chars): {chunk['text'][:300]}...")
        print()

        samples = await factory.process_chunk(chunk)

        if samples:
            print("Generated samples:")
            for s in samples:
                print(f"  [{s['type']:15}] {s['instruction'][:80]}...")
                print(f"    → {s['output']}")
        else:
            print("No samples generated")

    await factory.client.close()


def main():
    parser = argparse.ArgumentParser(description="Contextual Distillation Factory")
    parser.add_argument("--mode", type=str, default="preview",
                        choices=["distill", "filter", "preview"],
                        help="Mode: distill (full), filter (clean only), preview (test)")
    parser.add_argument("--chunks", type=str, default=None, help="Chunks JSON path")
    parser.add_argument("--input", type=str, default=None, help="Input for filter mode")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--api_base", type=str, default=None, help="API base URL (e.g., http://localhost:11434/v1)")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent requests")
    parser.add_argument("--num_samples", type=int, default=5, help="Preview: number of samples")
    parser.add_argument("--num_test", type=int, default=0, help="Test mode: process only N chunks (0=all)")

    args = parser.parse_args()

    if args.mode == "distill":
        asyncio.run(run_distill_mode(args))
    elif args.mode == "filter":
        asyncio.run(run_filter_mode(args))
    elif args.mode == "preview":
        asyncio.run(run_preview_mode(args))


if __name__ == "__main__":
    main()
