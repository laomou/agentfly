"""Pi Agent 适配器."""

from lmswitch.agents.base import AgentAdapter
from lmswitch.models.schema import ResolvedConfig
from lmswitch.models.types import AgentType


class PiAdapter(AgentAdapter):
    """Pi — AI 编程 Agent."""

    name = AgentType.PI
    display_name = "Pi"
    preferred_format = "openai"

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        env = {"OPENAI_API_KEY": config.provider.api_key}
        if config.effective_api_base:
            env["OPENAI_BASE_URL"] = config.effective_api_base
        if config.agent.model:
            env["PI_MODEL"] = config.agent.model
        return env

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        return ["pi"] + config.agent.extra_args
