"""Claude Code Adapter 测试."""

from __future__ import annotations

from lmswitch.agents.claude_code import ClaudeCode
from lmswitch.models.types import AgentType


class TestClaudeCode:
    """Claude Code 适配器测试."""

    def test_identity(self):
        adapter = ClaudeCode()
        assert adapter.name == AgentType.CLAUDE_CODE
        assert adapter.display_name == "Claude Code"

    def test_preferred_format(self):
        adapter = ClaudeCode()
        assert adapter.preferred_format == "anthropic"

    def test_env_vars(self, resolved_config):
        adapter = ClaudeCode()
        env = adapter.env_vars(resolved_config)

        assert env["ANTHROPIC_AUTH_TOKEN"] == "test-key-123"
        assert "ANTHROPIC_BASE_URL" in env

    def test_env_vars_no_model_derivation(self, resolved_config):
        """adapter 不推导模型角色，由 provider.env_for 负责."""
        adapter = ClaudeCode()
        env = adapter.env_vars(resolved_config)

        assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in env
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" not in env
        assert "ANTHROPIC_MODEL" not in env

    def test_launch_command(self, resolved_config):
        adapter = ClaudeCode()
        cmd = adapter.launch_command(resolved_config)

        assert cmd[0] == "claude"

