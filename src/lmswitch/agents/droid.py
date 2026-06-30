"""Droid (Factory) Agent 适配器.

Factory Droid 不读环境变量指定 provider，自定义模型配置在
~/.factory/settings.json 的 customModels[]。这里用 ScopedConfigFile 在
launch 期间临时写入、退出还原，不持久污染用户配置。

参考 ollama cmd/launch/droid.go 与 https://docs.factory.ai/cli/byok/overview
"""

from __future__ import annotations

from pathlib import Path

from lmswitch.agents.base import Agent, openai_base_url
from lmswitch.agents.configfile import ScopedConfigFile
from lmswitch.models.schema import ResolvedConfig
from lmswitch.models.types import AgentType

# 我们写入条目的固定标识，便于 merge 时只重建自己这条
_MODEL_ID = "lmswitch"
_SETTINGS = (".factory", "settings.json")


class Droid(Agent):
    """Droid — Factory AI 编程 Agent (BYOK via ~/.factory/settings.json)."""

    name = AgentType.DROID
    display_name = "Droid"
    preferred_format = "openai"

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        return ["droid"]

    def pre_launch(self, config: ResolvedConfig) -> None:
        self._patch = ScopedConfigFile(Path.home().joinpath(*_SETTINGS))
        self._patch.apply(lambda d: _merge_settings(d, config))

    def post_launch(self) -> None:
        patch = getattr(self, "_patch", None)
        if patch is not None:
            patch.restore()
            self._patch = None


def _merge_settings(d: dict, config: ResolvedConfig) -> None:
    model = config.agent.model or ""
    entry = {
        "model": model,
        "displayName": model,
        "baseUrl": openai_base_url(config.effective_api_base),
        "apiKey": config.provider.api_key,
        "provider": "generic-chat-completion-api",
        "maxOutputTokens": 64000,
        "id": _MODEL_ID,
        "index": 0,
    }
    # 保留用户原有的其它自定义模型，只重建自己这条
    others = [
        m for m in d.get("customModels", [])
        if isinstance(m, dict) and m.get("id") != _MODEL_ID
    ]
    d["customModels"] = [entry, *others]
    sds = d.get("sessionDefaultSettings")
    if not isinstance(sds, dict):
        sds = {}
    sds["model"] = _MODEL_ID
    d["sessionDefaultSettings"] = sds
