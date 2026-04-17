from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class BaseDataAdapter(ABC):
    """数据源适配器基类"""

    def __init__(
        self,
        source: str,
        options: Optional[Dict[str, Any]] = None
    ):
        self.source = source
        self.options = options or {}

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """读取数据"""
        pass

    @abstractmethod
    def write(self, data: pd.DataFrame, destination: str) -> bool:
        """写入数据"""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """验证数据源"""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """获取数据模式"""
        df = self.read()
        return {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }

    def get_quality_report(self) -> Dict[str, Any]:
        """获取数据质量报告"""
        df = self.read()
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": {},
            "duplicate_rows": 0,
            "column_stats": {}
        }

        # 缺失值统计
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
            report["missing_values"][col] = {
                "count": int(missing_count),
                "percentage": round(missing_pct, 2)
            }

        # 重复行
        report["duplicate_rows"] = int(df.duplicated().sum())

        # 列统计
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "unique_count": int(df[col].nunique()),
                "missing_count": int(df[col].isnull().sum())
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None
                })

            report["column_stats"][col] = col_stats

        return report
