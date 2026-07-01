"""Codex Adapter 测试."""

from __future__ import annotations

from agentfly.agents.codex import Codex
from agentfly.models.schema import AgentConfig, ProviderConfig, ResolvedConfig
from agentfly.models.types import AgentType, ProviderType


def make_codex_config():
    provider = ProviderConfig(
        name=ProviderType.OPENAI,
        api_key="sk-test",
        endpoints={"openai": "http://x"},
        models=["gpt-4o"],
        default_model="gpt-4o",
    )
    agent = AgentConfig(
        name=AgentType.CODEX,
        provider="openai",
        model="gpt-4o",
    )
    return ResolvedConfig(
        agent=agent,
        provider=provider,
        effective_api_base="https://api.openai.com",
        effective_api_format="openai",
    )


class TestCodex:
    """Codex 适配器测试."""

    def test_identity(self):
        adapter = Codex()
        assert adapter.name == AgentType.CODEX
        assert adapter.display_name == "Codex"

    def test_env_vars(self):
        adapter = Codex()
        env = adapter.env_vars(make_codex_config())

        assert env["OPENAI_API_KEY"] == "sk-test"
        assert "OPENAI_BASE_URL" in env

    def test_launch_command(self):
        adapter = Codex()
        config = make_codex_config()
        cmd = adapter.launch_command(config)

        assert "codex" in cmd[0]
        assert "--model" in cmd
        assert "gpt-4o" in cmd
