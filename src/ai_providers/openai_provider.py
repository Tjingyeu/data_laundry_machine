from typing import List, Optional, Dict, Any
import openai
from .base_provider import BaseAIProvider, AIMessage, AIResponse, MessageRole


class OpenAIProvider(BaseAIProvider):
    """OpenAI 供应商实现"""

    @property
    def provider_name(self) -> str:
        return "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        **kwargs
    ):
        super().__init__(api_key, model, **kwargs)
        self.client = openai.OpenAI(api_key=api_key)

    async def chat(
        self,
        messages: List[AIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AIResponse:
        """发送对话请求到 OpenAI"""
        openai_messages = self._convert_messages_to_dict(messages)

        params = {
            "model": self.model,
            "messages": openai_messages,
            **self.extra_params,
            **kwargs
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**params)

        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]

        return AIResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            model=response.model,
            finish_reason=choice.finish_reason
        )

    async def chat_stream(
        self,
        messages: List[AIMessage],
        **kwargs
    ):
        """流式对话请求"""
        openai_messages = self._convert_messages_to_dict(messages)

        params = {
            "model": self.model,
            "messages": openai_messages,
            stream=True,
            **self.extra_params,
            **kwargs
        }

        response = self.client.chat.completions.create(**params)

        for chunk in response:
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                yield delta.content

            if choice.finish_reason:
                break
