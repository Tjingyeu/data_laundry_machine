from typing import List, Optional, Dict, Any, Iterator
import anthropic
from .base_provider import BaseAIProvider, AIMessage, AIResponse, MessageRole


class AnthropicProvider(BaseAIProvider):
    """Anthropic 供应商实现"""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        **kwargs
    ):
        super().__init__(api_key, model, **kwargs)
        self.client = anthropic.Anthropic(api_key=api_key)

    def _convert_messages_for_anthropic(
        self,
        messages: List[AIMessage]
    ) -> List[Dict[str, Any]]:
        """将消息转换为 Anthropic 格式"""
        result = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                result.append({
                    "role": "user",
                    "content": f"[System] {msg.content}"
                })
            else:
                result.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        return result

    async def chat(
        self,
        messages: List[AIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AIResponse:
        """发送对话请求到 Anthropic"""
        anthropic_messages = self._convert_messages_for_anthropic(messages)

        params = {
            "model": self.model,
            "messages": anthropic_messages,
            **self.extra_params,
            **kwargs
        }

        if tools:
            params["tools"] = tools

        response = self.client.messages.create(**params)

        tool_calls = None
        content = ""

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": block.input
                    }
                })

        return AIResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            model=response.model,
            finish_reason=str(response.stop_reason)
        )

    async def chat_stream(
        self,
        messages: List[AIMessage],
        **kwargs
    ) -> Iterator[str]:
        """流式对话请求"""
        anthropic_messages = self._convert_messages_for_anthropic(messages)

        params = {
            "model": self.model,
            "messages": anthropic_messages,
            stream=True,
            **self.extra_params,
            **kwargs
        }

        with self.client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text
