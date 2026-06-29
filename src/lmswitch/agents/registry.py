"""Agent 注册表 — 管理所有已注册的 Agent 适配器."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Iterator, Optional

from lmswitch.agents.base import Agent


class AgentRegistry:
    """Agent 注册表.

    支持两种注册方式:
    1. 内置: 通过 register() 方法显式注册
    2. 插件: 通过 setuptools entry_points (组: lmswitch.agents)
    """

    def __init__(self):
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        self._agents[agent.name.value] = agent

    def get(self, name: str) -> Optional[Agent]:
        return self._agents.get(name)

    def list(self) -> Iterator[Agent]:
        yield from self._agents.values()

    def names(self) -> list[str]:
        """返回所有已注册的 Agent 名称."""
        return list(self._agents.keys())

    def discover_from_entry_points(self) -> None:
        """从 setuptools entry_points 自动发现 Agent 适配器.

        使用 entry_point group: 'lmswitch.agents'
        """
        try:
            eps = entry_points(group="lmswitch.agents")
        except TypeError:
            # Python < 3.12 兼容
            eps = entry_points().get("lmswitch.agents", [])

        for ep in eps:
            try:
                cls = ep.load()
                adapter = cls()
                self.register(adapter)
            except Exception:
                # 跳过加载失败的插件
                pass


# 全局单例
_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """获取全局 Agent 注册表单例."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
        _registry.discover_from_entry_points()
    return _registry
