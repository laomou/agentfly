"""OpenClaw Agent 适配器."""

from __future__ import annotations

from lmswitch.agents.base import AgentAdapter
from lmswitch.models.schema import ResolvedConfig
from lmswitch.models.types import AgentType, ProviderType


class OpenClawAdapter(AgentAdapter):
    """OpenClaw (多 Provider 支持) 适配器.

    OpenClaw 支持多种 Provider，根据配置动态设置环境变量.
    """

    name = AgentType.OPENCLAW
    display_name = "OpenClaw"
    preferred_format = "openai"  # 默认 OpenAI，多 Provider 支持

    # Provider → OpenClaw 环境变量映射
    _PROVIDER_ENV_MAP = {
        ProviderType.ANTHROPIC: {
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url_env": "ANTHROPIC_BASE_URL",
        },
        ProviderType.OPENAI: {
            "api_key_env": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_BASE_URL",
        },
        ProviderType.GOOGLE: {
            "api_key_env": "GOOGLE_API_KEY",
            "base_url_env": "GOOGLE_BASE_URL",
        },
    }

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        mapping = self._PROVIDER_ENV_MAP.get(config.provider.name, {})
        env = {}
        if api_key_env := mapping.get("api_key_env"):
            env[api_key_env] = config.provider.api_key
        if base_url_env := mapping.get("base_url_env") and config.effective_api_base:
            env[base_url_env] = config.effective_api_base
        if config.agent.model:
            env["OPENCLAW_MODEL"] = config.agent.model
        return env

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        cmd = ["openclaw"]
        if config.agent.model:
            cmd.extend(["--model", config.agent.model])
        cmd.extend(config.agent.extra_args)
        return cmd
