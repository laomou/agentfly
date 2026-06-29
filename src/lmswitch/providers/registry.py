"""Provider 实例工厂 — 根据 ProviderConfig 创建对应的 Provider 实例."""

from __future__ import annotations

from lmswitch.models.schema import ProviderConfig
from lmswitch.models.types import ProviderType
from lmswitch.providers.anthropic import AnthropicProvider
from lmswitch.providers.custom import CustomProvider
from lmswitch.providers.deepseek import DeepSeekProvider
from lmswitch.providers.openai import OpenAIProvider

_PROVIDER_CLASSES: dict[ProviderType, type] = {
    ProviderType.ANTHROPIC: AnthropicProvider,
    ProviderType.OPENAI: OpenAIProvider,
    ProviderType.DEEPSEEK: DeepSeekProvider,
    ProviderType.CUSTOM: CustomProvider,
}


def get_provider(config: ProviderConfig):
    """根据 ProviderConfig 创建对应的 Provider 实例."""
    cls = _PROVIDER_CLASSES.get(config.name)
    return cls(config) if cls else None
