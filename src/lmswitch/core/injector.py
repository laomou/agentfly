"""环境变量注入器 — 将 resolved config 转化为实际的环境变量."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from lmswitch.models.schema import ResolvedConfig


class EnvInjector:
    """环境变量注入器.

    负责:
    1. 将 AgentAdapter 产出的 env_vars dict 注入到当前进程或子进程
    2. 管理配置文件写入
    """

    @staticmethod
    def build_env(
        resolved: ResolvedConfig,
        agent_env: dict[str, str],
        *,
        inherit: bool = True,
    ) -> dict[str, str]:
        """构建完整的子进程环境变量字典.

        Args:
            resolved: 已解析的配置.
            agent_env: AgentAdapter 产出的环境变量.
            inherit: 是否继承当前进程的环境变量.

        Returns:
            完整的环境变量字典.
        """
        env: dict[str, str] = {}

        if inherit:
            env.update(os.environ)

        # Provider extra env
        env.update(resolved.provider.extra_env)

        # Agent env overrides (最高优先级)
        env.update(agent_env)

        # Agent 显式覆盖
        env.update(resolved.agent.env_overrides)

        return env

    @staticmethod
    def export_shell(env_vars: dict[str, str]) -> str:
        """生成 shell export 命令字符串.

        >>> EnvInjector.export_shell({"FOO": "bar"})
        'export FOO="bar"'
        """
        lines = []
        for k, v in env_vars.items():
            # 安全转义双引号
            escaped = v.replace('"', '\\"')
            lines.append(f'export {k}="{escaped}"')
        return "\n".join(lines)

    @staticmethod
    def write_config_files(
        files: dict[str, str],
        backup: bool = True,
    ) -> list[Path]:
        """写入 Agent 配置文件.

        Args:
            files: {路径: 内容} 映射.
            backup: 是否备份已有文件.

        Returns:
            写入的文件路径列表.
        """
        written = []
        for path_str, content in files.items():
            p = Path(path_str).expanduser()

            # 备份已有文件
            if backup and p.exists():
                backup_path = p.with_suffix(p.suffix + ".lmswitch.bak")
                backup_path.write_text(p.read_text(encoding="utf-8"))

            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            written.append(p)

        return written
