from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import json
import uuid
import logging

from .base_agent import BaseAgent, AIMessage, MessageRole, CleaningResult
from .agent_executor import AgentExecutor
from .cleaning_operation import CleaningOperation, OperationStatus
from ..ai_providers.base_provider import BaseAIProvider
from ..prompts.prompt_manager import PromptManager
from ..operations.missing_value_handler import MissingValueHandler
from ..operations.deduplication import Deduplication
from ..operations.type_converter import TypeConverter

logger = logging.getLogger(__name__)


@dataclass
class DataCleaningAgentConfig:
    """数据清洗 Agent 配置"""
    max_iterations: int = 5
    convergence_threshold: float = 0.95
    sample_size: int = 100


class DataCleaningAgent(BaseAgent):
    """
    数据清洗 Agent
    核心职责：
    1. 与 AI 供应商交互，获取清洗策略
    2. 调度执行具体的清洗操作
    3. 管理清洗流程的迭代
    """

    def __init__(
        self,
        ai_provider: BaseAIProvider,
        prompt_manager: PromptManager,
        config: Optional[DataCleaningAgentConfig] = None
    ):
        self.ai_provider = ai_provider
        self.prompt_manager = prompt_manager
        self.config = config or DataCleaningAgentConfig()

        # 初始化执行器并注册操作
        self.executor = AgentExecutor()
        self._register_default_operations()

        # 状态
        self.data_sample: Optional[pd.DataFrame] = None
        self.full_data: Optional[pd.DataFrame] = None
        self.conversation_history: List[AIMessage] = []
        self.operations_log: List[Dict[str, Any]] = []

    def _register_default_operations(self) -> None:
        """注册默认的清洗操作"""
        self.executor.register_operation("missing_value", MissingValueHandler)
        self.executor.register_operation("deduplication", Deduplication)
        self.executor.register_operation("type_convert", TypeConverter)

    def register_operation(self, operation_type: str, operation_class) -> None:
        """注册自定义清洗操作"""
        self.executor.register_operation(operation_type, operation_class)

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用的工具定义"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "handle_missing_values",
                    "description": "Handle missing values in specified columns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Columns to process"
                            },
                            "strategy": {
                                "type": "string",
                                "enum": ["drop", "fill_mean", "fill_median", "fill_mode",
                                        "fill_constant", "interpolate", "forward_fill", "backward_fill"],
                                "description": "Strategy to handle missing values"
                            },
                            "fill_value": {
                                "type": "string",
                                "description": "Constant value to fill (used with fill_constant strategy)"
                            }
                        },
                        "required": ["columns", "strategy"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "deduplicate",
                    "description": "Remove duplicate rows",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Columns to check for duplicates (empty for all columns)"
                            },
                            "keep": {
                                "type": "string",
                                "enum": ["first", "last", "none"],
                                "description": "Which duplicate to keep"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_type",
                    "description": "Convert column data type",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string",
                                "description": "Column to convert"
                            },
                            "target_type": {
                                "type": "string",
                                "enum": ["int", "float", "string", "datetime", "boolean", "category"],
                                "description": "Target data type"
                            },
                            "errors": {
                                "type": "string",
                                "enum": ["raise", "coerce", "ignore"],
                                "description": "How to handle conversion errors"
                            }
                        },
                        "required": ["column", "target_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_cleaning",
                    "description": "Mark the cleaning process as complete",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Summary of cleaning operations performed"
                            },
                            "quality_score": {
                                "type": "number",
                                "description": "Estimated data quality score (0-1)"
                            }
                        },
                        "required": ["summary"]
                    }
                }
            }
        ]

    async def clean(self, data: pd.DataFrame) -> CleaningResult:
        """执行数据清洗"""
        self.full_data = data.copy()

        # Step 1: 数据采样
        self.data_sample = self._sample_data(data)

        # Step 2: 初始化对话历史
        self.conversation_history = []
        self.operations_log = []

        logger.info(f"Starting cleaning process with {len(data)} rows, sample size: {len(self.data_sample)}")

        # Step 3: 迭代清洗
        for iteration in range(self.config.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")

            # 向 AI 请求清洗策略
            strategy = await self._request_cleaning_strategy()

            if not strategy:
                logger.warning("No strategy returned from AI, stopping")
                break

            # 检查是否完成
            if strategy.get("done"):
                logger.info(f"AI signaled cleaning complete: {strategy.get('summary')}")
                break

            # 解析并执行操作
            operations = self._parse_operations(strategy)
            if not operations:
                logger.info("No operations to execute, stopping")
                break

            for op in operations:
                await self._execute_operation(op)

            # 检查收敛
            quality_score = strategy.get("quality_score", 0)
            if quality_score >= self.config.convergence_threshold:
                logger.info(f"Converged with quality score: {quality_score}")
                break

        # 生成最终结果
        return self._generate_result()

    async def _sample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """采样数据用于 AI 分析"""
        if len(data) <= self.config.sample_size:
            return data.copy()
        return data.sample(n=self.config.sample_size, random_state=42).reset_index(drop=True)

    async def _request_cleaning_strategy(self) -> Dict[str, Any]:
        """向 AI 请求清洗策略"""
        # 构建分析请求 Prompt
        data_sample_json = self.data_sample.to_json(orient="records")
        data_schema = self._get_schema_description()
        conversation = self._format_history()

        prompt = self.prompt_manager.render("analysis_request", {
            "data_sample": data_sample_json,
            "data_schema": json.dumps(data_schema, indent=2, default=str),
            "conversation_history": conversation
        })

        messages = [
            AIMessage(role=MessageRole.SYSTEM, content=self.prompt_manager.get_system_prompt()),
            AIMessage(role=MessageRole.USER, content=prompt)
        ]

        # 添加对话历史
        for hist_msg in self.conversation_history[-10:]:  # 保留最近 10 条
            messages.append(hist_msg)

        # 调用 AI
        tools = self._get_available_tools()
        response = await self.ai_provider.chat(messages, tools=tools)

        # 解析 AI 响应
        return self._parse_ai_response(response)

    def _get_schema_description(self) -> Dict[str, Any]:
        """获取数据模式描述"""
        if self.data_sample is None:
            return {}

        return {
            "columns": list(self.data_sample.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.data_sample.dtypes.items()},
            "shape": self.data_sample.shape,
            "missing_values": self.data_sample.isnull().sum().to_dict(),
            "sample": self.data_sample.head(5).to_dict(orient="records")
        }

    def _format_history(self) -> str:
        """格式化对话历史"""
        if not self.conversation_history:
            return "No previous conversation."

        lines = []
        for msg in self.conversation_history[-6:]:  # 最近 6 条
            role = msg.role.value.upper()
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"**{role}**: {content}")

        return "\n".join(lines)

    def _parse_ai_response(self, response) -> Dict[str, Any]:
        """解析 AI 响应"""
        result = {
            "content": response.content,
            "done": False,
            "operations": [],
            "quality_score": 0.0
        }

        # 处理工具调用
        if response.tool_calls:
            for tc in response.tool_calls:
                func = tc.get("function", {})
                name = func.get("name")
                arguments = func.get("arguments", {})

                # 如果是字符串，尝试解析为 JSON
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                if name == "finish_cleaning":
                    result["done"] = True
                    result["summary"] = arguments.get("summary", "")
                    result["quality_score"] = arguments.get("quality_score", 1.0)
                else:
                    result["operations"].append({
                        "tool": name,
                        "arguments": arguments
                    })

        # 添加到对话历史
        self.conversation_history.append(
            AIMessage(role=MessageRole.ASSISTANT, content=response.content)
        )

        return result

    def _parse_operations(self, strategy: Dict[str, Any]) -> List[CleaningOperation]:
        """解析 AI 响应中的操作"""
        operations = []

        for op_data in strategy.get("operations", []):
            tool = op_data.get("tool")
            args = op_data.get("arguments", {})

            if tool == "handle_missing_values":
                operations.append(CleaningOperation.create(
                    operation_type="missing_value",
                    target_columns=args.get("columns", []),
                    parameters={
                        "strategy": args.get("strategy", "drop"),
                        "fill_value": args.get("fill_value")
                    }
                ))
            elif tool == "deduplicate":
                operations.append(CleaningOperation.create(
                    operation_type="deduplication",
                    target_columns=args.get("columns", []),
                    parameters={"keep": args.get("keep", "first")}
                ))
            elif tool == "convert_type":
                operations.append(CleaningOperation.create(
                    operation_type="type_convert",
                    target_columns=[args.get("column", "")],
                    parameters={
                        "target_type": args.get("target_type"),
                        "errors": args.get("errors", "coerce")
                    }
                ))

        return operations

    async def _execute_operation(self, operation: CleaningOperation) -> None:
        """执行单个清洗操作"""
        logger.info(f"Executing: {operation.operation_type} on {operation.target_columns}")

        try:
            operation.status = OperationStatus.IN_PROGRESS
            result = await self.executor.execute(operation, self.data_sample)

            # 更新数据样本
            self.data_sample = result.data

            # 记录操作
            self.operations_log.append({
                "operation_id": operation.operation_id,
                "operation_type": operation.operation_type,
                "status": operation.status.value,
                "affected_rows": result.affected_rows,
                "success": result.success
            })

            # 向 AI 反馈结果
            await self._report_operation_result(operation, result)

            logger.info(f"Operation completed: {operation.operation_type}, affected {result.affected_rows} rows")

        except Exception as e:
            logger.error(f"Operation failed: {operation.operation_type}, error: {e}")
            operation.status = OperationStatus.FAILED
            operation.error = str(e)

    async def _report_operation_result(self, operation: CleaningOperation, result) -> None:
        """向 AI 报告操作执行结果"""
        prompt = self.prompt_manager.render("operation_result", {
            "operation_name": operation.operation_type,
            "success": result.success,
            "affected_rows": result.affected_rows,
            "details": json.dumps(result.details, default=str),
            "warnings": result.warnings
        })

        self.conversation_history.append(
            AIMessage(role=MessageRole.USER, content=prompt)
        )

    def _generate_result(self) -> CleaningResult:
        """生成清洗结果"""
        # 计算最终数据质量报告
        quality_report = {
            "total_rows": len(self.full_data),
            "total_columns": len(self.full_data.columns),
            "missing_values": self.full_data.isnull().sum().to_dict(),
            "duplicate_rows": int(self.full_data.duplicated().sum()),
            "operations_count": len(self.operations_log)
        }

        return CleaningResult(
            total_rows=len(self.full_data),
            cleaned_rows=len(self.data_sample),
            operations_applied=self.operations_log,
            data_quality_report=quality_report
        )

    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析数据质量"""
        sample = self._sample_data(data)
        return {
            "schema": {
                "columns": list(sample.columns),
                "dtypes": {col: str(dtype) for col, dtype in sample.dtypes.items()},
                "shape": sample.shape
            },
            "quality": {
                "missing_values": sample.isnull().sum().to_dict(),
                "duplicate_rows": int(sample.duplicated().sum()),
                "memory_usage": int(sample.memory_usage(deep=True).sum())
            },
            "column_stats": {
                col: {
                    "unique": int(sample[col].nunique()),
                    "missing": int(sample[col].isnull().sum())
                }
                for col in sample.columns
            }
        }
