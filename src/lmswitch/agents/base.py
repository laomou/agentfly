"""Agent 基类."""

from __future__ import annotations

from abc import ABC, abstractmethod

from lmswitch.models.schema import ResolvedConfig
from lmswitch.models.types import AgentType


class Agent(ABC):
    """Agent 基类.

    每个 AI Agent 实现此接口，定义:
    - 需要哪些环境变量
    - 需要写入哪些配置文件
    - 如何启动
    - 需要什么 API 格式 (preferred_format)
    """

    name: AgentType
    display_name: str = ""
    preferred_format: str = "openai"  # "openai" | "anthropic" — Agent 需要的 API 协议

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        """返回该 Agent 需要的环境变量字典."""
        return {}

    @abstractmethod
    def launch_command(self, config: ResolvedConfig) -> list[str]:
        """返回启动命令 (传递给 subprocess.Popen)."""
        ...

    def pre_launch(self, config: ResolvedConfig) -> None:
        """启动前钩子 — 检查前置条件、写入配置文件等."""
        pass

    def post_launch(self) -> None:
        """启动后钩子 — 清理临时文件等."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name.value}>"
