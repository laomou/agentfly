"""Claude Agent 适配器 — Anthropic 官方 Claude Code CLI (二进制 `claude`)."""

from __future__ import annotations

from agentfly.agents.base import Agent
from agentfly.models.schema import ResolvedConfig
from agentfly.models.types import AgentType


class Claude(Agent):
    """Claude — Anthropic 官方 Claude Code CLI 适配器.

    模型角色映射 (OPUS/SONNET/HAIKU 等) 由 provider.env_for 提供，
    见 providers/{provider}.json 的 "claude" 条目。
    """

    name = AgentType.CLAUDE
    display_name = "Claude Code"
    preferred_format = "anthropic"

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        env: dict[str, str] = {
            "ANTHROPIC_AUTH_TOKEN": config.provider.api_key,
        }
        if config.effective_api_base:
            env["ANTHROPIC_BASE_URL"] = config.effective_api_base
            env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
        if config.agent.model:
            env["ANTHROPIC_MODEL"] = config.agent.model
        return env

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        return ["claude"]
