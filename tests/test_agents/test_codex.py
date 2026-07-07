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

    def test_env_vars_only_api_key(self):
        # OPENAI_BASE_URL 在 codex 0.142+ 已失效, 不再注入
        env = self._adapter.env_vars(make_codex_config())
        assert env == {"OPENAI_API_KEY": "sk-test"}

    def test_launch_command_injects_responses_provider(self):
        cmd = self._adapter.launch_command(make_codex_config())
        joined = " ".join(cmd)

        assert cmd[0] == "codex"
        # -c 注入自定义 responses provider (替代失效的 OPENAI_BASE_URL)
        assert "-c" in cmd
        assert 'model_provider="agentfly"' in cmd
        assert 'wire_api="responses"' in joined
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
