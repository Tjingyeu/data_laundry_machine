from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import uuid


class OperationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CleaningOperation:
    """清洗操作"""
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: str = ""
    target_columns: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: OperationStatus = OperationStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @classmethod
    def create(
        cls,
        operation_type: str,
        target_columns: List[str],
        parameters: Dict[str, Any]
    ) -> "CleaningOperation":
        """创建清洗操作"""
        return cls(
            operation_type=operation_type,
            target_columns=target_columns,
            parameters=parameters
        )
