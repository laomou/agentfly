"""启动器测试."""

from __future__ import annotations

from lmswitch.agents.claude_code import ClaudeCodeAdapter
from lmswitch.core.launcher import AgentLauncher


class TestAgentLauncher:
    """Agent 启动器测试."""

    def test_dry_run(self, resolved_config):
        adapter = ClaudeCodeAdapter()
        launcher = AgentLauncher(adapter)

        result = launcher.dry_run(resolved_config)

        assert result["agent"] == "claude-code"
        assert "ANTHROPIC_API_KEY" in result["env_vars"]
        assert isinstance(result["launch_command"], list)

    def test_launch_dry_run_flag(self, resolved_config):
        adapter = ClaudeCodeAdapter()
        launcher = AgentLauncher(adapter)

        exit_code = launcher.launch(resolved_config, dry_run=True)
        assert exit_code == 0
