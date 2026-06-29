"""Claude Code Adapter 测试."""

from __future__ import annotations

from lmswitch.agents.claude_code import ClaudeCodeAdapter
from lmswitch.models.types import AgentType, ProviderType


class TestClaudeCodeAdapter:
    """Claude Code 适配器测试."""

    def test_identity(self):
        adapter = ClaudeCodeAdapter()
        assert adapter.name == AgentType.CLAUDE_CODE
        assert adapter.display_name == "Claude Code"

    def test_preferred_format(self):
        adapter = ClaudeCodeAdapter()
        assert adapter.preferred_format == "anthropic"

    def test_env_vars(self, resolved_config):
        adapter = ClaudeCodeAdapter()
        env = adapter.env_vars(resolved_config)

        assert env["ANTHROPIC_API_KEY"] == "test-key-123"
        assert "ANTHROPIC_BASE_URL" in env

    def test_env_vars_with_model(self, resolved_config):
        resolved_config.agent.model = "claude-opus-4-8"
        adapter = ClaudeCodeAdapter()
        env = adapter.env_vars(resolved_config)

        assert env["ANTHROPIC_MODEL"] == "claude-opus-4-8"

    def test_launch_command(self, resolved_config):
        adapter = ClaudeCodeAdapter()
        cmd = adapter.launch_command(resolved_config)

        assert cmd[0] == "claude"

    def test_launch_command_with_extra_args(self, resolved_config):
        resolved_config.agent.extra_args = ["--verbose", "--project", "/tmp"]
        adapter = ClaudeCodeAdapter()
        cmd = adapter.launch_command(resolved_config)

        assert cmd == ["claude", "--verbose", "--project", "/tmp"]
