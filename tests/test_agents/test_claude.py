"""Claude Adapter 测试."""

from __future__ import annotations

from agentfly.agents.claude import Claude
from agentfly.models.types import AgentType


class TestClaude:
    """Claude 适配器测试."""

    def test_identity(self):
        adapter = Claude()
        assert adapter.name == AgentType.CLAUDE
        assert adapter.display_name == "Claude Code"

    def test_preferred_format(self):
        assert Claude().preferred_format == "anthropic"

    def test_env_vars(self, resolved_config):
        env = Claude().env_vars(resolved_config)
        assert env["ANTHROPIC_AUTH_TOKEN"] == "test-key-123"
        assert "ANTHROPIC_BASE_URL" in env

    def test_env_vars_no_model_derivation(self, resolved_config):
        """adapter 不推导模型角色，由 provider.env_for 负责."""
        env = Claude().env_vars(resolved_config)
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in env
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" not in env
        assert env["ANTHROPIC_MODEL"] == resolved_config.agent.model

    def test_small_fast_model_follows_main(self, resolved_config):
        """后台小模型指向同一模型, 避免网关缺 haiku 导致 404."""
        env = Claude().env_vars(resolved_config)
        assert env["ANTHROPIC_SMALL_FAST_MODEL"] == resolved_config.agent.model

    def test_launch_command(self, resolved_config):
        assert Claude().launch_command(resolved_config)[0] == "claude"
