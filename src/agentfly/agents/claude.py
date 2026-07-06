"""Claude Agent 适配器 — Anthropic 官方 Claude Code CLI (二进制 `claude`)."""

from __future__ import annotations

from agentfly.agents.base import Agent
from agentfly.models.schema import ResolvedConfig
from agentfly.models.types import AgentType
from agentfly.providers.anthropic import is_official_anthropic_base


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
        base = config.effective_api_base
        if base and not is_official_anthropic_base(base):
            # 仅自定义网关: 覆盖 BASE_URL, 并关掉 attribution header
            # (网关多不认该 header, 官方端点则保持 Claude Code 默认行为)
            env["ANTHROPIC_BASE_URL"] = base
            env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
        if config.agent.model:
            env["ANTHROPIC_MODEL"] = config.agent.model
            # 后台小任务 (标题/摘要等) 也走同一模型, 否则默认打 claude-3-5-haiku-*,
            # 自定义网关多半没有该模型 → 后台调用 404
            env["ANTHROPIC_SMALL_FAST_MODEL"] = config.agent.model
        return env

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        return ["claude"]
