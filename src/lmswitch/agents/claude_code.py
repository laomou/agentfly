"""Claude Code Agent 适配器."""

from __future__ import annotations

import json
import os
from pathlib import Path

from lmswitch.agents.base import AgentAdapter
from lmswitch.models.schema import ResolvedConfig
from lmswitch.models.types import AgentType


class ClaudeCodeAdapter(AgentAdapter):
    """Claude Code (Anthropic 官方 CLI) 适配器.

    需要环境变量:
    - ANTHROPIC_API_KEY
    - ANTHROPIC_BASE_URL (可选)
    - ANTHROPIC_MODEL (可选)
    """

    name = AgentType.CLAUDE_CODE
    display_name = "Claude Code"
    preferred_format = "anthropic"

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        env = {
            "ANTHROPIC_API_KEY": config.provider.api_key,
        }
        if config.effective_api_base:
            env["ANTHROPIC_BASE_URL"] = config.effective_api_base
        if config.agent.model:
            env["ANTHROPIC_MODEL"] = config.agent.model
        return env

    def config_files(self, config: ResolvedConfig) -> dict[str, str]:
        """注入 Claude Code settings.json 配置."""
        settings_path = Path.home() / ".claude" / "settings.json"

        settings = {}
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        # 注入模型配置
        if config.agent.model:
            settings.setdefault("model", config.agent.model)

        if not settings:
            return {}

        return {str(settings_path): json.dumps(settings, indent=2, ensure_ascii=False)}

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        cmd = ["claude"]
        cmd.extend(config.agent.extra_args)
        return cmd
