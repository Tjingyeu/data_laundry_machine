import os
from typing import Optional, Dict, Any
import pandas as pd
from .base_adapter import BaseDataAdapter


class CSVAdapter(BaseDataAdapter):
    """CSV 数据适配器"""

    def read(self) -> pd.DataFrame:
        """读取 CSV 文件"""
        return pd.read_csv(
            self.source,
            encoding=self.options.get("encoding", "utf-8"),
            delimiter=self.options.get("delimiter", ","),
            header=self.options.get("header", 0),
            skiprows=self.options.get("skiprows", None),
            na_values=self.options.get("na_values", ['', 'NA', 'N/A', 'null']),
            dtype=self.options.get("dtype", None)
        )

    def write(self, data: pd.DataFrame, destination: str) -> bool:
        """写入 CSV 文件"""
        data.to_csv(
            destination,
            index=self.options.get("index", False),
            encoding=self.options.get("encoding", "utf-8"),
            sep=self.options.get("delimiter", ",")
        )
        return True

    def validate(self) -> bool:
        """验证 CSV 文件是否存在"""
        return os.path.exists(self.source) and os.path.isfile(self.source)

    @classmethod
    def from_config(cls, path: str, **kwargs) -> "CSVAdapter":
        """从配置创建适配器"""
        return cls(source=path, options=kwargs)
