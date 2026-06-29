"""Agent 启动器 — 通过 subprocess 启动 Agent 进程."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from lmswitch.models.schema import ResolvedConfig
from lmswitch.agents.base import AgentAdapter
from lmswitch.core.injector import EnvInjector


class LaunchError(Exception):
    """启动失败异常."""

    pass


class AgentLauncher:
    """Agent 启动器.

    负责:
    1. 调用 AgentAdapter 获取 env_vars / config_files / launch_command
    2. 注入环境变量和配置文件
    3. 通过 subprocess 启动 Agent
    """

    def __init__(self, adapter: AgentAdapter):
        self._adapter = adapter
        self._injector = EnvInjector()

    def dry_run(self, resolved: ResolvedConfig) -> dict:
        """预览启动配置，不实际执行.

        Returns:
            包含 env_vars, config_files, launch_command 的字典.
        """
        return {
            "agent": self._adapter.name,
            "env_vars": self._adapter.env_vars(resolved),
            "config_files": self._adapter.config_files(resolved),
            "launch_command": self._adapter.launch_command(resolved),
        }

    def launch(
        self,
        resolved: ResolvedConfig,
        *,
        cwd: Optional[str] = None,
        dry_run: bool = False,
    ) -> int:
        """启动 Agent.

        Args:
            resolved: 已解析的配置.
            cwd: 工作目录.
            dry_run: 仅预览，不实际启动.

        Returns:
            进程退出码 (dry_run 时返回 0).

        Raises:
            LaunchError: 启动失败.
        """
        # 1. 获取 agent 产出
        env_vars = self._adapter.env_vars(resolved)
        config_files = self._adapter.config_files(resolved)
        launch_cmd = self._adapter.launch_command(resolved)

        if dry_run:
            import json
            info = self.dry_run(resolved)
            print(json.dumps(info, indent=2, ensure_ascii=False))
            return 0

        # 2. 写入配置文件
        self._adapter.pre_launch(resolved)
        if config_files:
            self._injector.write_config_files(config_files)

        # 3. 构建完整环境变量
        full_env = self._injector.build_env(resolved, env_vars)

        # 4. 启动子进程
        try:
            proc = subprocess.Popen(
                launch_cmd,
                env=full_env,
                cwd=cwd,
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            proc.wait()

            # 5. 启动后清理
            self._adapter.post_launch()

            return proc.returncode

        except FileNotFoundError:
            raise LaunchError(
                f"无法找到启动命令: {' '.join(launch_cmd)}\n"
                f"请确认 '{self._adapter.name}' 已安装且在 PATH 中."
            )
        except Exception as e:
            raise LaunchError(f"启动 {self._adapter.name} 失败: {e}")
