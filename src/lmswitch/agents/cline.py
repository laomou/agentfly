"""Cline Agent 适配器."""

from lmswitch.agents.base import Agent
from lmswitch.models.schema import ResolvedConfig
from lmswitch.models.types import AgentType


class Cline(Agent):
    """Cline — VS Code AI 编程助手.

    支持 OpenAI / Anthropic，通过环境变量配置.
    """

    name = AgentType.CLINE
    display_name = "Cline"
    preferred_format = "openai"

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        env: dict[str, str] = {}
        fmt = config.effective_api_format
        if fmt == "anthropic":
            env["ANTHROPIC_API_KEY"] = config.provider.api_key
            if config.effective_api_base:
                env["ANTHROPIC_BASE_URL"] = config.effective_api_base
        else:
            env["OPENAI_API_KEY"] = config.provider.api_key
            if config.effective_api_base:
                env["OPENAI_BASE_URL"] = config.effective_api_base
        if config.agent.model:
            env["CLINE_MODEL"] = config.agent.model
        return env

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        return ["cline"]
