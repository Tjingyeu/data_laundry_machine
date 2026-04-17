from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class AIMessage:
    role: MessageRole
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class AIResponse:
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None
    model: str
    finish_reason: str


class BaseAIProvider(ABC):
    """AI 供应商抽象基类"""

    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.extra_params = kwargs

    @abstractmethod
    async def chat(
        self,
        messages: List[AIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AIResponse:
        """发送对话请求"""
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[AIMessage],
        **kwargs
    ):
        """流式对话请求"""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """供应商名称"""
        pass

    def _convert_messages_to_dict(self, messages: List[AIMessage]) -> List[Dict[str, str]]:
        """将 AIMessage 列表转换为 API 格式"""
        result = []
        for msg in messages:
            msg_dict = {"role": msg.role.value, "content": msg.content}
            result.append(msg_dict)
        return result
