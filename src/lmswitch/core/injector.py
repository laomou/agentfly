"""环境变量注入器 — 将 resolved config 转化为实际的环境变量."""

from __future__ import annotations

import os

from lmswitch.models.schema import ResolvedConfig


class EnvInjector:
    """环境变量注入器."""

    @staticmethod
    def build_env(
        resolved: ResolvedConfig,
        agent_env: dict[str, str],
        *,
        inherit: bool = True,
    ) -> dict[str, str]:
        """构建完整的子进程环境变量字典.

        优先级 (低 → 高):
        1. Agent 产出的环境变量
        2. 继承当前 Shell 环境 (已有值优先)
        3. Agent 显式覆盖 (最高)
        """
        env: dict[str, str] = {}
        env.update(agent_env)
        if inherit:
            for k, v in os.environ.items():
                env.setdefault(k, v)
        return env

