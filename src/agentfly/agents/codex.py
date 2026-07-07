"""Codex (OpenAI) Agent 适配器.

Codex 自 2026/02 起内置 openai provider 固定走 Responses API (/v1/responses)，
仅靠 OPENAI_BASE_URL 重定向无法让仅支持 Chat Completions 的第三方 provider
调通 (codex 仍会请求 /v1/responses)。

解法: 用 codex 的 `-c key=value` 配置覆盖 (点路径可创建新表) 注入一个自定义
model_provider，设 wire_api="chat_completions"，让 codex 走 /v1/chat/completions
调用第三方 OpenAI 兼容端点。全程不落盘、不碰用户 ~/.codex，退出即焚。

参考 ollama cmd/launch/codex.go 的 `-c model_providers.<name>.wire_api=...` 用法
(ollama 自家 server 实现 Responses API 故用 "responses"; 此处面向第三方端点用
"chat_completions")。
"""

from __future__ import annotations

from agentfly.agents.base import Agent, openai_base_url
from agentfly.models.schema import ResolvedConfig
from agentfly.models.types import AgentType

# 我们注入的 provider key (与 codex 内置 openai 区分)
_PROVIDER = "agentfly"


class Codex(Agent):
    """OpenAI Codex CLI 适配器.

    需要环境变量:
    - OPENAI_API_KEY  (codex 经 env_key 读取)

    注意: 自 2026/02 起 Codex 仅原生支持 Responses API (/v1/responses)，
    本适配器通过 wire_api="chat_completions" 让其兼容第三方 OpenAI 兼容端点。
    """

    name = AgentType.CODEX
    display_name = "Codex"
    preferred_format = "openai"

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        return {
            "OPENAI_API_KEY": config.provider.api_key,
        }

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        base_url = openai_base_url(config.effective_api_base)
        p = _PROVIDER
        cmd = ["codex"]
        # -c 覆盖: 定义一个 chat_completions provider 并选用它
        cmd += [
            "-c", f'model_provider="{p}"',
            "-c", f'model_providers.{p}.name="AgentFly"',
            "-c", f'model_providers.{p}.base_url="{base_url}"',
            "-c", f'model_providers.{p}.env_key="OPENAI_API_KEY"',
            "-c", f'model_providers.{p}.wire_api="chat_completions"',
        ]
        if config.agent.model:
            cmd += ["--model", config.agent.model]
        return cmd
