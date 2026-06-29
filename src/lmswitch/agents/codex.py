"""Codex (OpenAI) Agent 适配器."""

from __future__ import annotations

from lmswitch.agents.base import AgentAdapter
from lmswitch.models.schema import ResolvedConfig
from lmswitch.models.types import AgentType


class CodexAdapter(AgentAdapter):
    """OpenAI Codex CLI 适配器.

    需要环境变量:
    - OPENAI_API_KEY
    """

    name = AgentType.CODEX
    display_name = "Codex"
    preferred_format = "openai"

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        env = {
            "OPENAI_API_KEY": config.provider.api_key,
        }
        if config.effective_api_base:
            env["OPENAI_BASE_URL"] = config.effective_api_base
        return env

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        cmd = ["codex"]
        if config.agent.model:
            cmd.extend(["--model", config.agent.model])
        cmd.extend(config.agent.extra_args)
        return cmd
