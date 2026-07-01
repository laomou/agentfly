"""AgentFly CLI 入口."""

from __future__ import annotations

import click

from agentfly import __version__
from agentfly.cli.doctor import doctor
from agentfly.cli.launch import launch
from agentfly.cli.provider_cmd import provider_group
from agentfly.cli.test_cmd import test


@click.group()
@click.version_option(version=__version__, prog_name="agentfly")
def cli():
    """AgentFly — AI Agent 客制化配置中心.

    统一管理多 AI Agent 配置，一键启动.
    """


# 注册子命令
cli.add_command(doctor)
cli.add_command(launch)
cli.add_command(test)
cli.add_command(provider_group)


if __name__ == "__main__":
    cli()
