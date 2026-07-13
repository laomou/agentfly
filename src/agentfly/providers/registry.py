"""Provider 实例工厂 — 根据 ProviderConfig 创建对应的 Provider 实例."""

from __future__ import annotations

from agentfly.models.schema import ProviderConfig
from agentfly.models.types import ProviderType
from agentfly.providers.anthropic import AnthropicProvider
from agentfly.providers.custom import CustomProvider
from agentfly.providers.deepseek import DeepSeekProvider
from agentfly.providers.openai import OpenAIProvider

_PROVIDER_CLASSES: dict[ProviderType, type] = {
    ProviderType.ANTHROPIC: AnthropicProvider,
    ProviderType.OPENAI: OpenAIProvider,
    ProviderType.DEEPSEEK: DeepSeekProvider,
    ProviderType.CUSTOM: CustomProvider,
}


def get_provider(config: ProviderConfig):
    """根据 ProviderConfig 创建对应的 Provider 实例."""
    cls = _PROVIDER_CLASSES.get(config.type)
    return cls(config) if cls else None
