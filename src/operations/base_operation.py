from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import pandas as pd


class OperationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CleaningOperation:
    """清洗操作定义"""
    operation_id: str
    operation_type: str
    target_columns: List[str]
    parameters: Dict[str, Any]
    status: OperationStatus = OperationStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class OperationResult:
    """操作执行结果"""
    success: bool
    data: pd.DataFrame
    affected_rows: int
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class BaseOperation(ABC):
    """清洗操作基类"""

    @property
    @abstractmethod
    def operation_type(self) -> str:
        """操作类型标识"""
        pass

    @property
    def description(self) -> str:
        """操作描述"""
        return f"{self.operation_type} operation"

    @abstractmethod
    async def execute(
        self,
        data: pd.DataFrame,
        operation: CleaningOperation
    ) -> OperationResult:
        """执行操作"""
        pass

    def _validate_columns(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> tuple[bool, Optional[str]]:
        """验证列是否存在"""
        missing = [col for col in columns if col not in data.columns]
        if missing:
            return False, f"Missing columns: {', '.join(missing)}"
        return True, None

    def _get_column_stats(
        self,
        data: pd.DataFrame,
        column: str
    ) -> Dict[str, Any]:
        """获取列统计信息"""
        col_data = data[column]
        stats = {
            "dtype": str(col_data.dtype),
            "count": len(col_data),
            "missing": int(col_data.isnull().sum()),
            "unique": int(col_data.nunique())
        }

        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                "min": float(col_data.min()) if not col_data.isnull().all() else None,
                "max": float(col_data.max()) if not col_data.isnull().all() else None,
                "mean": float(col_data.mean()) if not col_data.isnull().all() else None,
                "median": float(col_data.median()) if not col_data.isnull().all() else None
            })
        else:
            stats.update({
                "top_values": col_data.value_counts().head(5).to_dict()
            })

        return stats
