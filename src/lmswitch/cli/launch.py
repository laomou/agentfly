"""[lmswitch launch] 启动 AI Agent."""

from __future__ import annotations

import sys

import click

from lmswitch import __version__
from lmswitch.agents.registry import get_registry
from lmswitch.core.config import ensure_config_exists
from lmswitch.core.launcher import AgentLauncher, LaunchError
from lmswitch.core.resolver import ConfigResolver
from lmswitch.models.schema import AgentConfig
from lmswitch.models.types import AgentType


@click.command(name="launch")
@click.argument("agent_name", required=False)
@click.option("--provider", "-P", default=None, help="指定 Provider (覆盖 YAML 绑定)")
@click.option("--model", "-m", default=None, help="覆盖默认模型")
@click.option("--project", "-p", default=None, help="指定项目/工作目录")
@click.option("--list", "list_agents", is_flag=True, default=False, help="列出所有可启动的 Agent")
def launch(
    agent_name: str | None,
    provider: str | None,
    model: str | None,
    project: str | None,
    list_agents: bool,
) -> None:
    """启动 AI Agent.

    \b
    示例:
      lmswitch launch claude-code                           # 使用 YAML 绑定的 Provider
      lmswitch launch codex --provider deepseek             # 直接指定 Provider
      lmswitch launch claude-code --model claude-opus-4-8   # 覆盖模型
      lmswitch launch --list                                # 列出 Agent
    """
    registry = get_registry()

    if list_agents:
        _list_available_agents(registry)
        return

    if not agent_name:
        click.secho("用法: lmswitch launch <agent-name> [--provider <name>]", fg="yellow")
        click.echo("  lmswitch launch --list    查看可用 Agent")
        sys.exit(1)

    adapter = registry.get(agent_name)
    if adapter is None:
        click.secho(f"未找到 Agent '{agent_name}'  可用: {registry.names()}", fg="red")
        sys.exit(1)

    # 加载配置
    config, _ = ensure_config_exists()

    # ── provider 选择逻辑 ──
    # 1. --provider 显式指定 > 2. YAML agent 绑定 > 3. 第一个 provider
    if provider:
        provider_key = provider
        # 创建临时 AgentConfig（不写入 YAML）
        agent_cfg = AgentConfig(
            name=AgentType(agent_name),
            provider=provider_key,
            model=model,
        )
        # 如果 config 里没有这个 agent，临时注入
        if agent_name not in config.agents:
            config.agents[agent_name] = agent_cfg
    else:
        agent_cfg = config.agents.get(agent_name)
        if agent_cfg and agent_cfg.provider:
            provider_key = agent_cfg.provider
        elif config.providers:
            provider_key = next(iter(config.providers))
        else:
            click.secho("无可用 Provider。请运行 'lmswitch provider add'", fg="red")
            sys.exit(1)

    # 解析配置
    try:
        resolver = ConfigResolver(config)
        resolved = resolver.resolve(agent_name, preferred_format=adapter.preferred_format,
                                     provider_key=provider_key)
    except KeyError as e:
        click.secho(f"配置缺失: {e}", fg="red")
        sys.exit(1)
    except ValueError as e:
        click.secho(f"格式不兼容: {e}", fg="red")
        sys.exit(1)

    if model:
        resolved.agent.model = model

    launcher = AgentLauncher(adapter)

    click.echo(f"  LMSwitch v{__version__}")
    click.echo(f"  {adapter.display_name} · {provider_key}")
    click.echo()

    try:
        exit_code = launcher.launch(resolved, cwd=project)
        sys.exit(exit_code)
    except LaunchError as e:
        click.secho(f"启动失败: {e}", fg="red")
        sys.exit(1)


def _list_available_agents(registry) -> None:
    """列出所有可用的 Agent."""
    click.echo("可用的 AI Agent:")
    click.echo()
    for adapter in registry.list():
        click.echo(f"  {adapter.name.value:<18} {adapter.display_name}")
        click.echo(f"    需要的 API 格式: {adapter.preferred_format}")
        click.echo()
