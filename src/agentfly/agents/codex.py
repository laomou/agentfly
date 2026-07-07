"""Codex (OpenAI) Agent 适配器.

Codex 自 2026/02 (PR #10157) 起彻底移除 Chat Completions wire_api 支持，
model_providers[*].wire_api 仅接受 "responses"，所有 provider 必须实现
Responses API (/v1/responses)。

本适配器用 OPENAI_BASE_URL 把 codex 内置 openai provider 重定向到第三方端点
(该 provider 固定走 responses wire_api)。若第三方端点只支持 /v1/chat/completions，
调用会失败——需经支持 Responses API 的网关/代理 (LiteLLM、llama.cpp server、
VibeAround 等) 转换。pre_launch 会对非 OpenAI 官方端点打印提示。

详见 https://github.com/openai/codex/discussions/7782
"""

from __future__ import annotations

import sys
from urllib.parse import urlparse

from agentfly.agents.base import Agent, openai_base_url
from agentfly.models.schema import ResolvedConfig
from agentfly.models.types import AgentType

# OpenAI 官方端点 host —— 命中则视为原生支持 Responses API，无需提示
_OFFICIAL_HOSTS = {"api.openai.com"}


class Codex(Agent):
    """OpenAI Codex CLI 适配器.

    需要环境变量:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (以 /v1 结尾，重定向内置 openai provider)

    注意: 自 2026/02 起 Codex 仅支持 Responses API (/v1/responses)，
    仅支持 Chat Completions 的第三方 provider 需经代理转换。
    """

    name = AgentType.CODEX
    display_name = "Codex"
    preferred_format = "openai"

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        env = {
            "OPENAI_API_KEY": config.provider.api_key,
        }
        if config.effective_api_base:
            env["OPENAI_BASE_URL"] = openai_base_url(config.effective_api_base)
        return env

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        cmd = ["codex"]
        if config.agent.model:
            cmd.extend(["--model", config.agent.model])
        return cmd

    def pre_launch(self, config: ResolvedConfig) -> None:
        """第三方端点提示: codex 仅支持 Responses API,纯 Chat Completions 端点需代理."""
        host = (urlparse(config.effective_api_base).hostname or "").lower()
        if not host or host in _OFFICIAL_HOSTS:
            return
        print(
            f"  ⚠ codex 仅支持 Responses API (/v1/responses)。端点 "
            f"{config.effective_api_base} 若只支持 Chat Completions，需经支持"
            f" Responses API 的网关/代理 (如 LiteLLM、llama.cpp server) 转换，"
            f"否则调用会失败。\n"
            f"  详见 https://github.com/openai/codex/discussions/7782",
            file=sys.stderr,
        )
