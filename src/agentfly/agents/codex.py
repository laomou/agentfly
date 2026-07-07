"""Codex (OpenAI) Agent 适配器.

Codex 0.142+ 不再认 OPENAI_BASE_URL 环境变量, 必须用 -c 配置覆盖定义自定义
model_provider. codex 内置 openai provider 固定走 Responses API (wire_api=
responses), 第三方端点需经 -c 重定向 base_url 到自身, 且端点须实现
/v1/responses (codex 2026/02 起彻底移除 Chat Completions wire_api).

仅支持 Chat Completions 的端点 (deepseek/qwen 等) 无法直接被 codex 调用,
需经支持 Responses API 的网关/代理 (LiteLLM、llama.cpp server 等) 转换.
agentfly test 的 API 列会标出端点是否支持 responses.
"""

from __future__ import annotations

from agentfly.agents.base import Agent, openai_base_url
from agentfly.models.schema import ResolvedConfig
from agentfly.models.types import AgentType

# 注入的自定义 provider key (与 codex 内置 openai 区分)
_PROVIDER = "agentfly"


class Codex(Agent):
    """OpenAI Codex CLI 适配器.

    需要环境变量:
    - OPENAI_API_KEY  (codex 经 env_key 读取)

    注意: 自 2026/02 起 Codex 仅支持 Responses API (/v1/responses),
    经 -c 注入自定义 provider 重定向 base_url; OPENAI_BASE_URL 已无效.
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
        # -c 覆盖: 定义一个 responses provider 并选用它 (OPENAI_BASE_URL 已失效)
        cmd += [
            "-c", f'model_provider="{p}"',
            "-c", f'model_providers.{p}.name="AgentFly"',
            "-c", f'model_providers.{p}.base_url="{base_url}"',
            "-c", f'model_providers.{p}.env_key="OPENAI_API_KEY"',
            "-c", f'model_providers.{p}.wire_api="responses"',
        ]
        if config.agent.model:
            cmd += ["--model", config.agent.model]
        return cmd
