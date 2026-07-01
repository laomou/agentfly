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


@cli.command(name="completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), required=False, default=None)
@click.option("--install", "-i", is_flag=True, default=False, help="自动安装到 shell 配置文件")
def completion(shell: str | None, install: bool):
    """Tab 补全安装.

    \b
    用法:
      agentfly completion bash      输出 bash 补全脚本
      agentfly completion zsh       输出 zsh 补全脚本
      agentfly completion --install 自动检测 shell 并安装
    """
    import os

    if not shell and not install:
        raise click.UsageError(
            "请指定 shell 或使用 --install 自动安装.\n\n"
            "  agentfly completion bash       # 打印 bash 补全脚本\n"
            "  agentfly completion zsh        # 打印 zsh 补全脚本\n"
            "  agentfly completion --install  # 自动安装"
        )

    if install:
        if not shell:
            shell = os.path.basename(os.environ.get("SHELL", "bash"))
            if shell not in ("bash", "zsh", "fish"):
                shell = "bash"
        rc_files = {"bash": "~/.bashrc", "zsh": "~/.zshrc", "fish": "~/.config/fish/config.fish"}
        rc = os.path.expanduser(rc_files.get(shell, "~/.bashrc"))
        line = f'eval "$(agentfly completion {shell})"'
        _ensure_rc(rc, line)
        click.echo(f"✓ 已写入 {rc}")
        click.echo(f"  运行 source {rc} 或重开终端即可生效")
        return
    env = f"_AGENTFLY_COMPLETE={shell}_source"
    script = os.popen(f"{env} agentfly").read()
    click.echo(script, nl=False)


def _ensure_rc(rc_path: str, line: str) -> None:
    """追加到 rc 文件（去重）."""
    import os
    content = ""
    if os.path.exists(rc_path):
        content = open(rc_path).read()
    if line not in content:
        with open(rc_path, "a") as f:
            if content and not content.endswith("\n"):
                f.write("\n")
            f.write(f"{line}\n")


# 注册子命令
cli.add_command(doctor)
cli.add_command(launch)
cli.add_command(test)
cli.add_command(provider_group)


if __name__ == "__main__":
    cli()
