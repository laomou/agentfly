"""Pi (Pi Coding Agent) 适配器.

Pi 的自定义 provider 需要写 ~/.pi/agent/models.json（baseUrl 只能在文件里，
没有 --base-url flag）。这里用 ScopedConfigFile 临时写入、退出还原，模型用
`pi --provider lmswitch --model <model>` 选择。

参考 ollama cmd/launch/pi.go 与 pi custom-provider 文档。
"""

from __future__ import annotations

from pathlib import Path

from lmswitch.agents.base import Agent, openai_base_url
from lmswitch.agents.configfile import ScopedConfigFile
from lmswitch.models.schema import ResolvedConfig
from lmswitch.models.types import AgentType

_PROVIDER = "lmswitch"
_MODELS_JSON = (".pi", "agent", "models.json")


class Pi(Agent):
    """Pi — 极简终端编程 Agent (custom provider via ~/.pi/agent/models.json)."""

    name = AgentType.PI
    display_name = "Pi"
    preferred_format = "openai"

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        cmd = ["pi"]
        if config.agent.model:
            cmd += ["--provider", _PROVIDER, "--model", config.agent.model]
        return cmd

    def pre_launch(self, config: ResolvedConfig) -> None:
        self._patch = ScopedConfigFile(Path.home().joinpath(*_MODELS_JSON))
        self._patch.apply(lambda d: _merge_models(d, config))

    def post_launch(self) -> None:
        patch = getattr(self, "_patch", None)
        if patch is not None:
            patch.restore()
            self._patch = None


def _merge_models(d: dict, config: ResolvedConfig) -> None:
    providers = d.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    model = config.agent.model or ""
    providers[_PROVIDER] = {
        "baseUrl": openai_base_url(config.effective_api_base),
        "api": "openai-completions",
        "apiKey": config.provider.api_key,
        "models": [{"id": model}] if model else [],
    }
    d["providers"] = providers
