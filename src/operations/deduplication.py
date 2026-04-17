from typing import Dict, Any, List
import pandas as pd
from .base_operation import BaseOperation, CleaningOperation, OperationResult, OperationStatus


class Deduplication(BaseOperation):
    """重复数据删除"""

    @property
    def operation_type(self) -> str:
        return "deduplication"

    @property
    def description(self) -> str:
        return "Remove duplicate rows from data"

    async def execute(
        self,
        data: pd.DataFrame,
        operation: CleaningOperation
    ) -> OperationResult:
        """执行重复数据删除"""
        columns = operation.target_columns
        keep = operation.parameters.get("keep", "first")

        # 验证 keep 参数
        if keep not in ["first", "last", "none"]:
            operation.status = OperationStatus.FAILED
            operation.error = f"Invalid keep parameter: {keep}"
            return OperationResult(
                success=False,
                data=data,
                affected_rows=0,
                details={"error": f"Invalid keep parameter: {keep}"}
            )

        result_data = data.copy()
        rows_before = len(result_data)

        try:
            if columns:
                # 验证列
                valid, error = self._validate_columns(data, columns)
                if not valid:
                    operation.status = OperationStatus.FAILED
                    operation.error = error
                    return OperationResult(
                        success=False,
                        data=data,
                        affected_rows=0,
                        details={"error": error}
                    )
                result_data = result_data.drop_duplicates(subset=columns, keep=keep)
            else:
                # 全行去重
                result_data = result_data.drop_duplicates(keep=keep)

            rows_after = len(result_data)
            duplicates_removed = rows_before - rows_after

            operation.status = OperationStatus.COMPLETED
            operation.result = {
                "rows_before": rows_before,
                "rows_after": rows_after,
                "duplicates_removed": duplicates_removed,
                "columns_checked": columns if columns else "all columns"
            }

            return OperationResult(
                success=True,
                data=result_data,
                affected_rows=duplicates_removed,
                details={
                    "rows_before": rows_before,
                    "rows_after": rows_after,
                    "duplicates_removed": duplicates_removed,
                    "columns": columns if columns else "all columns",
                    "keep": keep
                }
            )

        except Exception as e:
            operation.status = OperationStatus.FAILED
            operation.error = str(e)
            return OperationResult(
                success=False,
                data=data,
                affected_rows=0,
                details={"error": str(e)}
            )
