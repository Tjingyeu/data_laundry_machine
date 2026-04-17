from typing import Dict, Type, Optional
from .base_provider import BaseAIProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider


class ProviderFactory:
    """AI 供应商工厂"""

    _providers: Dict[str, Type[BaseAIProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: Type[BaseAIProvider]
    ) -> None:
        """注册新的供应商"""
        cls._providers[name] = provider_class

    @classmethod
    def create(
        cls,
        provider_name: str,
        api_key: str,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseAIProvider:
        """创建供应商实例"""
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: {provider_name}. Available: {available}"
            )

        provider_class = cls._providers[provider_name]

        # 设置默认模型
        if model is None:
            if provider_name == "openai":
                model = "gpt-4o"
            elif provider_name == "anthropic":
                model = "claude-3-5-sonnet-20241022"

        return provider_class(api_key=api_key, model=model, **kwargs)

    @classmethod
    def available_providers(cls) -> list:
        """获取可用的供应商列表"""
        return list(cls._providers.keys())
