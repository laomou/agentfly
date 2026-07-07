"""Codex Adapter 测试."""

from __future__ import annotations

from agentfly.agents.codex import Codex
from agentfly.models.schema import AgentConfig, ProviderConfig, ResolvedConfig
from agentfly.models.types import AgentType, ProviderType


def make_codex_config(api_base: str = "https://api.openai.com") -> ResolvedConfig:
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
        effective_api_base=api_base,
        effective_api_format="openai",
    )


class TestCodex:
    """Codex 适配器测试."""

    def setup_method(self):
        self._adapter = Codex()

    def test_identity(self):
        assert self._adapter.name == AgentType.CODEX
        assert self._adapter.display_name == "Codex"

    def test_env_vars(self):
        env = self._adapter.env_vars(make_codex_config())

        assert env["OPENAI_API_KEY"] == "sk-test"
        assert env["OPENAI_BASE_URL"].endswith("/v1")

    def test_launch_command(self):
        cmd = self._adapter.launch_command(make_codex_config())

        assert "codex" in cmd[0]
        assert "--model" in cmd
        assert "gpt-4o" in cmd

    def test_pre_launch_silent_for_official_endpoint(self, capsys):
        self._adapter.pre_launch(make_codex_config("https://api.openai.com"))
        assert capsys.readouterr().err == ""

    def test_pre_launch_warns_for_third_party_endpoint(self, capsys):
        self._adapter.pre_launch(make_codex_config("https://api.deepseek.com"))
        err = capsys.readouterr().err
        assert "Responses API" in err
        assert "https://github.com/openai/codex/discussions/7782" in err
