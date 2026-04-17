from typing import Dict, Any, List
import pandas as pd
from .base_operation import BaseOperation, CleaningOperation, OperationResult, OperationStatus


class MissingValueHandler(BaseOperation):
    """缺失值处理"""

    @property
    def operation_type(self) -> str:
        return "missing_value"

    @property
    def description(self) -> str:
        return "Handle missing values in data"

    async def execute(
        self,
        data: pd.DataFrame,
        operation: CleaningOperation
    ) -> OperationResult:
        """执行缺失值处理"""
        columns = operation.target_columns
        strategy = operation.parameters.get("strategy", "drop")
        fill_value = operation.parameters.get("fill_value", None)

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

        result_data = data.copy()
        original_missing = {}
        affected_rows = 0

        # 统计处理前的缺失值
        for col in columns:
            original_missing[col] = int(data[col].isnull().sum())

        try:
            if strategy == "drop":
                # 删除包含缺失值的行
                rows_before = len(result_data)
                result_data = result_data.dropna(subset=columns)
                affected_rows = rows_before - len(result_data)

            elif strategy == "fill_mean":
                for col in columns:
                    if pd.api.types.is_numeric_dtype(result_data[col]):
                        fill_val = result_data[col].mean()
                        missing_count = result_data[col].isnull().sum()
                        result_data[col].fillna(fill_val, inplace=True)
                        affected_rows += missing_count

            elif strategy == "fill_median":
                for col in columns:
                    if pd.api.types.is_numeric_dtype(result_data[col]):
                        fill_val = result_data[col].median()
                        missing_count = result_data[col].isnull().sum()
                        result_data[col].fillna(fill_val, inplace=True)
                        affected_rows += missing_count

            elif strategy == "fill_mode":
                for col in columns:
                    if not result_data[col].mode().empty:
                        fill_val = result_data[col].mode()[0]
                        missing_count = result_data[col].isnull().sum()
                        result_data[col].fillna(fill_val, inplace=True)
                        affected_rows += missing_count

            elif strategy == "fill_constant":
                if fill_value is not None:
                    for col in columns:
                        missing_count = result_data[col].isnull().sum()
                        result_data[col].fillna(fill_value, inplace=True)
                        affected_rows += missing_count

            elif strategy == "interpolate":
                for col in columns:
                    if pd.api.types.is_numeric_dtype(result_data[col]):
                        missing_count = result_data[col].isnull().sum()
                        result_data[col] = result_data[col].interpolate(
                            method=self.parameters.get("method", "linear")
                        )
                        affected_rows += missing_count

            elif strategy == "forward_fill":
                for col in columns:
                    missing_count = result_data[col].isnull().sum()
                    result_data[col] = result_data[col].fillna(method="ffill")
                    affected_rows += missing_count

            elif strategy == "backward_fill":
                for col in columns:
                    missing_count = result_data[col].isnull().sum()
                    result_data[col] = result_data[col].fillna(method="bfill")
                    affected_rows += missing_count

            operation.status = OperationStatus.COMPLETED
            operation.result = {
                "strategy": strategy,
                "original_missing": original_missing,
                "columns_processed": columns
            }

            return OperationResult(
                success=True,
                data=result_data,
                affected_rows=affected_rows,
                details={
                    "strategy": strategy,
                    "original_missing": original_missing,
                    "columns": columns
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
