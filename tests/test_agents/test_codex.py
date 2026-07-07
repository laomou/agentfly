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

    def setup_method(self):
        self._adapter = Codex()

    def test_identity(self):
        assert self._adapter.name == AgentType.CODEX
        assert self._adapter.display_name == "Codex"

    def test_env_vars_only_api_key(self):
        # 不再用 OPENAI_BASE_URL 重定向 (codex 内置 provider 仍走 responses)
        env = self._adapter.env_vars(make_codex_config())
        assert env == {"OPENAI_API_KEY": "sk-test"}

    def test_launch_command_forces_chat_completions(self):
        cmd = self._adapter.launch_command(make_codex_config())
        joined = " ".join(cmd)

        assert cmd[0] == "codex"
        # 用 -c 覆盖注入自定义 chat_completions provider
        assert "-c" in cmd
        assert 'model_provider="agentfly"' in cmd
        assert 'wire_api="chat_completions"' in joined
        # base_url 需以 /v1 结尾
        assert 'base_url="https://api.openai.com/v1"' in joined
        assert 'env_key="OPENAI_API_KEY"' in joined
        # 模型透传
        assert "--model" in cmd
        assert "gpt-4o" in cmd

    def test_no_model_no_model_flag(self):
        cfg = make_codex_config()
        cfg.agent.model = None
        cmd = self._adapter.launch_command(cfg)
        assert "--model" not in cmd
