import os
from typing import Optional, Dict, Any
import pandas as pd
from .base_adapter import BaseDataAdapter


class JSONAdapter(BaseDataAdapter):
    """JSON 数据适配器"""

    def read(self) -> pd.DataFrame:
        """读取 JSON 文件"""
        orient = self.options.get("orient", "records")
        return pd.read_json(
            self.source,
            orient=orient,
            encoding=self.options.get("encoding", "utf-8")
        )

    def write(self, data: pd.DataFrame, destination: str) -> bool:
        """写入 JSON 文件"""
        orient = self.options.get("orient", "records")
        indent = self.options.get("indent", 2)
        data.to_json(
            destination,
            orient=orient,
            indent=indent,
            force_ascii=self.options.get("force_ascii", False)
        )
        return True

    def validate(self) -> bool:
        """验证 JSON 文件是否存在"""
        return os.path.exists(self.source) and os.path.isfile(self.source)

    @classmethod
    def from_config(cls, path: str, **kwargs) -> "JSONAdapter":
        """从配置创建适配器"""
        return cls(source=path, options=kwargs)


class JSONLinesAdapter(BaseDataAdapter):
    """JSON Lines (.jsonl) 数据适配器"""

    def read(self) -> pd.DataFrame:
        """读取 JSON Lines 文件"""
        return pd.read_json(
            self.source,
            orient="records",
            lines=True,
            encoding=self.options.get("encoding", "utf-8")
        )

    def write(self, data: pd.DataFrame, destination: str) -> bool:
        """写入 JSON Lines 文件"""
        data.to_json(
            destination,
            orient="records",
            lines=True,
            force_ascii=self.options.get("force_ascii", False)
        )
        return True

    def validate(self) -> bool:
        """验证 JSON Lines 文件是否存在"""
        return os.path.exists(self.source) and os.path.isfile(self.source)
