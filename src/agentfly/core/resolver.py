"""配置解析器 — 解析 ${ENV_VAR} 引用、合并优先级."""

from __future__ import annotations

import os
import re

from agentfly.models.schema import (
    AgentConfig,
    ProviderConfig,
    ResolvedConfig,
    UnifiedConfig,
)
from agentfly.models.types import AgentType, ProviderType

# 匹配 ${VAR_NAME} 模式
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_ref(value: str) -> str:
    """解析字符串中的 ${ENV_VAR} 引用.

    Args:
        value: 包含 ${VAR} 引用的字符串.

    Returns:
        解析后的字符串.

    Raises:
        ValueError: 引用的环境变量不存在.
    """
    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        env_val = os.environ.get(var_name)
        if env_val is None:
            raise ValueError(
                f"环境变量 '{var_name}' 未设置，"
                f"但在配置中通过 '${{{var_name}}}' 引用"
            )
        return env_val

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _resolve_provider(provider: ProviderConfig) -> ProviderConfig:
    """解析 ProviderConfig 中的环境变量引用."""
    return ProviderConfig(
        name=provider.name,
        api_key=_resolve_env_ref(provider.api_key),
        endpoints=provider.endpoints,
        models=provider.models,
        default_model=provider.default_model,
    )


def _pick_endpoint(provider: ProviderConfig, preferred_format: str) -> str:
    """按 Agent 需要的格式选对应的 API Base URL.

    Raises:
        ValueError: Provider 未配置该格式的 endpoint.
    """
    url = provider.endpoints.get(preferred_format)
    if not url:
        raise ValueError(
            f"Provider '{provider.name.value}' 未配置 {preferred_format} endpoint. "
            f"已配置: {list(provider.endpoints)}"
        )
    return url


class ConfigResolver:
    """配置解析器.

    职责:
    1. 根据 agent name 定位 AgentConfig 和 ProviderConfig
    2. 解析所有 ${ENV_VAR} 引用
    3. 处理覆盖优先级: Agent overrides > Agent 指定 > Provider 默认
    """

    def __init__(self, config: UnifiedConfig):
        self._config = config

    def resolve(self, agent_name: str, preferred_format: str = "openai",
                provider_key: str | None = None) -> ResolvedConfig:
        """解析指定 Agent 的完整配置.

        Args:
            agent_name: Agent 名称 (如 'claude').
            preferred_format: Agent 需要的 API 格式 ("openai" / "anthropic").
            provider_key: 覆盖 Agent 绑定的 Provider.

        Returns:
            ResolvedConfig — 已解析、可直接使用的配置.

        Raises:
            KeyError: Provider 未找到.
            ValueError: Provider 不支持 Agent 需要的格式.
        """
        # 1. 查找 AgentConfig（可能不存在，允许 provider_key 覆盖）
        agent = self._config.agents.get(agent_name)

        # 2. 确定 provider_key
        pk = provider_key or (agent.provider if agent else "")
        if not pk:
            pk = next(iter(self._config.providers)) if self._config.providers else ""

        # 3. 查找 ProviderConfig
        provider = self._config.providers.get(pk)
        if provider is None:
            raise KeyError(
                f"Provider '{pk}' 未配置.\n"
                f"请运行 'agentfly provider add {pk} --api-base <url> --api-key <key>'"
            )

        # 4. 解析环境变量引用
        resolved_provider = _resolve_provider(provider)

        # 5. 根据 Agent 格式选择正确的 API Base
        api_base = _pick_endpoint(resolved_provider, preferred_format)

        # 6. 合并 agent 配置（agent 可能为 None）
        if agent:
            resolved_agent = AgentConfig(
                name=agent.name,
                provider=pk,
                model=agent.model or resolved_provider.default_model,
            )
        else:
            resolved_agent = AgentConfig(
                name=AgentType(agent_name) if agent_name in AgentType.__members__.values() else AgentType("openai"),
                provider=pk,
                model=resolved_provider.default_model,
            )

        return ResolvedConfig(
            agent=resolved_agent,
            provider=resolved_provider,
            effective_api_base=api_base,
            effective_api_format=preferred_format,
        )

    def get_provider(self, provider_name: str | ProviderType) -> ProviderConfig:
        """获取单个 Provider 配置（已解析环境变量）."""
        if isinstance(provider_name, ProviderType):
            provider_name = provider_name.value

        provider = self._config.providers.get(provider_name)
        if provider is None:
            raise KeyError(f"Provider '{provider_name}' 未配置")
        return _resolve_provider(provider)
