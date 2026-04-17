from typing import Dict, Type, List
import pandas as pd
from dataclasses import dataclass, field

from ..operations.base_operation import BaseOperation, OperationResult
from .cleaning_operation import CleaningOperation, OperationStatus


class AgentExecutor:
    """Agent 指令执行器"""

    def __init__(self):
        self._operations_registry: Dict[str, Type[BaseOperation]] = {}
        self.execution_history: List[Dict] = []

    def register_operation(self, operation_type: str, operation_class: Type[BaseOperation]) -> None:
        """注册清洗操作"""
        self._operations_registry[operation_type] = operation_class

    def get_available_operations(self) -> List[str]:
        """获取所有可用的操作类型"""
        return list(self._operations_registry.keys())

    async def execute(
        self,
        operation: CleaningOperation,
        data: pd.DataFrame
    ) -> OperationResult:
        """执行单个清洗操作"""
        op_type = operation.operation_type

        if op_type not in self._operations_registry:
            raise ValueError(f"Unknown operation type: {op_type}")

        op_handler = self._operations_registry[op_type]()
        result = await op_handler.execute(data, operation)

        self.execution_history.append({
            "operation_id": operation.operation_id,
            "operation_type": op_type,
            "status": operation.status.value,
            "affected_rows": result.affected_rows,
            "success": result.success
        })

        return result

    async def execute_batch(
        self,
        operations: List[CleaningOperation],
        data: pd.DataFrame
    ) -> tuple[pd.DataFrame, List[OperationResult]]:
        """批量执行清洗操作"""
        current_data = data
        results = []

        for op in operations:
            result = await self.execute(op, current_data)
            current_data = result.data
            results.append(result)

            if not result.success:
                break

        return current_data, results

    def get_execution_history(self) -> List[Dict]:
        """获取执行历史"""
        return self.execution_history

    def clear_history(self) -> None:
        """清空执行历史"""
        self.execution_history.clear()
