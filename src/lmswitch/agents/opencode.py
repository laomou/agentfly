"""OpenCode Agent 适配器.

通过 OPENCODE_CONFIG_CONTENT 环境变量内联传入整份 opencode 配置，进程退出
即消失，完全不落盘——也绕开了 opencode 自定义 provider options 偶尔不生效
的已知问题。参考 ollama cmd/launch/opencode.go。
"""

from __future__ import annotations

import json

from lmswitch.agents.base import Agent, openai_base_url
from lmswitch.models.schema import ResolvedConfig
from lmswitch.models.types import AgentType

_PROVIDER = "lmswitch"


class OpenCode(Agent):
    """OpenCode — 开源 AI 编程 Agent (配置经 OPENCODE_CONFIG_CONTENT 注入)."""

    name = AgentType.OPENCODE
    display_name = "OpenCode"
    preferred_format = "openai"

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        model = config.agent.model or ""
        cfg: dict = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                _PROVIDER: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "LMSwitch",
                    "options": {
                        "baseURL": openai_base_url(config.effective_api_base),
                        "apiKey": config.provider.api_key,
                    },
                    "models": {model: {"name": model}} if model else {},
                }
            },
        }
        if model:
            cfg["model"] = f"{_PROVIDER}/{model}"
        return {"OPENCODE_CONFIG_CONTENT": json.dumps(cfg, ensure_ascii=False)}

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        return ["opencode"]
