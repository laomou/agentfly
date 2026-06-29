"""Agent 启动器 — 通过 subprocess 启动 Agent 进程."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Optional

from lmswitch.models.schema import ResolvedConfig
from lmswitch.agents.base import Agent
from lmswitch.core.injector import EnvInjector
from lmswitch.providers.registry import get_provider


class LaunchError(Exception):
    """启动失败异常."""

    pass


class AgentLauncher:
    """Agent 启动器.

    负责:
    1. 调用 Agent 获取 env_vars / launch_command
    2. 注入环境变量
    3. 通过 subprocess 启动 Agent
    """

    def __init__(self, agent: Agent):
        self._adapter = agent
        self._injector = EnvInjector()

    def launch(
        self,
        resolved: ResolvedConfig,
        *,
        cwd: Optional[str] = None,
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
        launch_cmd = self._adapter.launch_command(resolved)

        # 2. Provider 环境变量覆盖 (来自远程 JSON 配置)
        provider = get_provider(resolved.provider)
        if provider:
            for k, v in provider.env_for(resolved.agent.name.value).items():
                env_vars[k] = v  # provider 覆盖 adapter 通用值

        # 3. 检查 Agent 是否已安装
        cmd_name = launch_cmd[0]
        if not shutil.which(cmd_name):
            raise LaunchError(
                f"未检测到 '{cmd_name}'，请先安装 {self._adapter.name}"
            )

        # 4. 构建完整环境变量
        self._adapter.pre_launch(resolved)
        full_env = self._injector.build_env(resolved, env_vars)

        # 5. 启动子进程
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
            self._adapter.post_launch()

            return proc.returncode

        except Exception as e:
            raise LaunchError(f"启动 {self._adapter.name} 失败: {e}")
