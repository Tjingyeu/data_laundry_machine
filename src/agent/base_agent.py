from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class AIMessage:
    role: MessageRole
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class CleaningResult:
    """清洗结果"""
    total_rows: int
    cleaned_rows: int
    operations_applied: List[Dict[str, Any]]
    data_quality_report: Dict[str, Any]
    export_path: Optional[str] = None


class BaseAgent(ABC):
    """基础 Agent 抽象类"""

    @abstractmethod
    async def clean(self, data) -> CleaningResult:
        """执行数据清洗"""
        pass

    @abstractmethod
    async def analyze(self, data) -> Dict[str, Any]:
        """分析数据质量"""
        pass
