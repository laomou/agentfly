"""Claude Code Agent 适配器."""

from __future__ import annotations

from lmswitch.agents.base import Agent
from lmswitch.models.schema import ResolvedConfig
from lmswitch.models.types import AgentType


class ClaudeCode(Agent):
    """Claude Code (Anthropic 官方 CLI) 适配器."""

    name = AgentType.CLAUDE_CODE
    display_name = "Claude Code"
    preferred_format = "anthropic"

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        env: dict[str, str] = {
            "ANTHROPIC_AUTH_TOKEN": config.provider.api_key,
        }
        if config.effective_api_base:
            env["ANTHROPIC_BASE_URL"] = config.effective_api_base
        return env

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        return ["claude"]
