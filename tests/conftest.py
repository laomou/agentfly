"""共享测试 fixtures."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import yaml

from agentfly.models.schema import (
    AgentConfig,
    ProviderConfig,
    ResolvedConfig,
    UnifiedConfig,
)
from agentfly.models.types import AgentType, ProviderType


@pytest.fixture
def provider_config():
    """标准 Anthropic Provider 配置."""
    return ProviderConfig(
        name=ProviderType.ANTHROPIC,
        api_key="test-key-123",
        base_url="https://api.anthropic.com",
        models=["claude-sonnet-4-6", "claude-opus-4-8"],
        default_model="claude-sonnet-4-6",
    )


@pytest.fixture
def agent_config():
    """标准 Claude Code Agent 配置."""
    return AgentConfig(
        name=AgentType.CLAUDE,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )


@pytest.fixture
def unified_config(provider_config, agent_config):
    """完整统一配置."""
    return UnifiedConfig(
        version="1",
        providers={"anthropic": provider_config},
        agents={agent_config.name.value: agent_config},
    )


@pytest.fixture
def resolved_config(provider_config, agent_config):
    """已解析的配置."""
    return ResolvedConfig(
        agent=agent_config,
        provider=provider_config,
        effective_api_base="https://api.anthropic.com",
        effective_api_format="anthropic",
    )


@pytest.fixture
def temp_config_file(unified_config):
    """创建临时配置文件用于测试."""
    data = unified_config.model_dump(mode="json")

    with NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(data, f, allow_unicode=True)

    yield f.name
    # 清理
    Path(f.name).unlink(missing_ok=True)
