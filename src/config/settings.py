import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AIConfig:
    """AI 配置"""
    provider: str
    api_key: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 4096


@dataclass
class AgentConfig:
    """Agent 配置"""
    max_iterations: int = 5
    convergence_threshold: float = 0.95
    sample_size: int = 100


@dataclass
class DataConfig:
    """数据配置"""
    input_path: str
    output_path: str


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class AppConfig:
    """应用配置"""
    ai: AIConfig
    agent: AgentConfig
    data: DataConfig
    logging: LoggingConfig


class Settings:
    """配置管理"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config()
        self._config: Dict[str, Any] = {}
        self._load()

    def _find_config(self) -> Optional[str]:
        """查找配置文件"""
        possible_paths = [
            "config.yaml",
            "config.yml",
            os.path.expanduser("~/.data_laundry_machine/config.yaml"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def _load(self) -> None:
        """加载配置"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}

    def _resolve_env_vars(self, value: Any) -> Any:
        """解析环境变量"""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, "")
        return value

    def _resolve_dict_env_vars(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """解析字典中的环境变量"""
        result = {}
        for k, v in d.items():
            if isinstance(v, str):
                result[k] = self._resolve_env_vars(v)
            elif isinstance(v, dict):
                result[k] = self._resolve_dict_env_vars(v)
            else:
                result[k] = v
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def get_ai_config(self) -> AIConfig:
        """获取 AI 配置"""
        ai_config = self._config.get("ai", {})
        ai_config = self._resolve_dict_env_vars(ai_config)
        return AIConfig(
            provider=ai_config.get("provider", "openai"),
            api_key=ai_config.get("api_key", ""),
            model=ai_config.get("model", "gpt-4o"),
            temperature=ai_config.get("temperature", 0.1),
            max_tokens=ai_config.get("max_tokens", 4096)
        )

    def get_agent_config(self) -> AgentConfig:
        """获取 Agent 配置"""
        agent_config = self._config.get("agent", {})
        return AgentConfig(
            max_iterations=agent_config.get("max_iterations", 5),
            convergence_threshold=agent_config.get("convergence_threshold", 0.95),
            sample_size=agent_config.get("sample_size", 100)
        )

    def get_data_config(self) -> DataConfig:
        """获取数据配置"""
        data_config = self._config.get("data", {})
        return DataConfig(
            input_path=data_config.get("input_path", "data/raw/input.csv"),
            output_path=data_config.get("output_path", "data/cleaned/output.csv")
        )

    def get_logging_config(self) -> LoggingConfig:
        """获取日志配置"""
        log_config = self._config.get("logging", {})
        return LoggingConfig(
            level=log_config.get("level", "INFO"),
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )


# 全局配置实例
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """获取配置实例"""
    global _settings
    if _settings is None:
        _settings = Settings(config_path)
    return _settings


def reload_settings(config_path: Optional[str] = None) -> Settings:
    """重新加载配置"""
    global _settings
    _settings = Settings(config_path)
    return _settings
