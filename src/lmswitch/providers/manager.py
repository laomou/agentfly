"""Provider 管理器 — 增删查改."""

from __future__ import annotations

from typing import Optional

from lmswitch.core.config import save_config
from lmswitch.models.schema import ProviderConfig, UnifiedConfig
from lmswitch.models.types import ProviderType


class ProviderManager:
    """Provider 配置管理器.

    负责 CRUD 操作，修改 UnifiedConfig 并持久化.
    """

    def __init__(self, config: UnifiedConfig):
        self._config = config

    def list(self) -> list[ProviderConfig]:
        """列出所有已配置的 Provider."""
        return list(self._config.providers.values())

    def get(self, name: str | ProviderType) -> Optional[ProviderConfig]:
        """获取指定 Provider 配置."""
        if isinstance(name, ProviderType):
            name = name.value
        return self._config.providers.get(name)

    def add(self, provider: ProviderConfig, key: str | None = None) -> None:
        """添加或更新 Provider 配置.

        Args:
            provider: Provider 配置.
            key: 存储键名。内置 Provider 默认为 name.value (如 "anthropic")，
                 CUSTOM Provider 必须指定 (如 "my-deepseek").
        """
        k = key or provider.name.value
        self._config.providers[k] = provider

    def remove(self, name: str | ProviderType) -> None:
        """删除 Provider 配置."""
        if isinstance(name, ProviderType):
            name = name.value
        if name not in self._config.providers:
            raise KeyError(f"Provider '{name}' 未配置")
        del self._config.providers[name]

    def save(self) -> None:
        """持久化到配置文件."""
        save_config(self._config)
