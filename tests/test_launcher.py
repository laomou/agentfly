"""启动器测试."""

from __future__ import annotations

from lmswitch.agents.claude_code import ClaudeCode
from lmswitch.core.launcher import AgentLauncher


class TestAgentLauncher:
    """Agent 启动器测试."""

    def test_launch_env_vars(self, resolved_config):
        adapter = ClaudeCode()
        launcher = AgentLauncher(adapter)

        env = adapter.env_vars(resolved_config)
        assert "ANTHROPIC_AUTH_TOKEN" in env
        assert "ANTHROPIC_BASE_URL" in env
