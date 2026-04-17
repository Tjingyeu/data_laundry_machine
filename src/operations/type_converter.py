from typing import Dict, Any, List
import pandas as pd
from .base_operation import BaseOperation, CleaningOperation, OperationResult, OperationStatus


class TypeConverter(BaseOperation):
    """数据类型转换"""

    @property
    def operation_type(self) -> str:
        return "type_convert"

    @property
    def description(self) -> str:
        return "Convert column data types"

    async def execute(
        self,
        data: pd.DataFrame,
        operation: CleaningOperation
    ) -> OperationResult:
        """执行数据类型转换"""
        column = operation.target_columns[0] if operation.target_columns else None
        target_type = operation.parameters.get("target_type")
        errors = operation.parameters.get("errors", "coerce")

        if column is None:
            operation.status = OperationStatus.FAILED
            operation.error = "No column specified for type conversion"
            return OperationResult(
                success=False,
                data=data,
                affected_rows=0,
                details={"error": "No column specified"}
            )

        if target_type is None:
            operation.status = OperationStatus.FAILED
            operation.error = "No target type specified"
            return OperationResult(
                success=False,
                data=data,
                affected_rows=0,
                details={"error": "No target type specified"}
            )

        # 验证列
        valid, error = self._validate_columns(data, [column])
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
        original_dtype = str(data[column].dtype)
        conversion_errors = 0

        try:
            if target_type == "int":
                result_data[column] = pd.to_numeric(
                    result_data[column],
                    errors=errors
                ).astype("Int64")
            elif target_type == "float":
                result_data[column] = pd.to_numeric(
                    result_data[column],
                    errors=errors
                ).astype("float64")
            elif target_type == "string":
                result_data[column] = result_data[column].astype(str)
            elif target_type == "datetime":
                format_str = operation.parameters.get("format")
                result_data[column] = pd.to_datetime(
                    result_data[column],
                    format=format_str,
                    errors=errors
                )
            elif target_type == "boolean":
                result_data[column] = result_data[column].astype(bool)
            elif target_type == "category":
                result_data[column] = result_data[column].astype("category")
            else:
                operation.status = OperationStatus.FAILED
                operation.error = f"Unsupported target type: {target_type}"
                return OperationResult(
                    success=False,
                    data=data,
                    affected_rows=0,
                    details={"error": f"Unsupported target type: {target_type}"}
                )

            if errors == "coerce":
                conversion_errors = result_data[column].isnull().sum() - data[column].isnull().sum()

            operation.status = OperationStatus.COMPLETED
            operation.result = {
                "column": column,
                "original_dtype": original_dtype,
                "target_dtype": str(result_data[column].dtype),
                "conversion_errors": int(conversion_errors)
            }

            return OperationResult(
                success=True,
                data=result_data,
                affected_rows=len(result_data),
                details={
                    "column": column,
                    "original_dtype": original_dtype,
                    "target_dtype": str(result_data[column].dtype),
                    "conversion_errors": int(conversion_errors)
                },
                warnings=[f"{conversion_errors} values could not be converted"] if conversion_errors > 0 else []
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
