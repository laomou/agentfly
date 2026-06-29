"""[lmswitch launch] 启动 AI Agent."""

from __future__ import annotations

import sys

import click

from lmswitch.agents.registry import get_registry
from lmswitch.core.config import load_config, ensure_config_exists
from lmswitch.core.launcher import AgentLauncher, LaunchError
from lmswitch.core.resolver import ConfigResolver
from lmswitch.models.schema import AgentConfig
from lmswitch.models.types import AgentType


@click.command(name="launch")
@click.argument("agent_name", required=False)
@click.option("--provider", "-P", default=None, help="指定 Provider (覆盖 YAML 绑定)")
@click.option("--model", "-m", default=None, help="覆盖默认模型")
@click.option("--project", "-p", default=None, help="指定项目/工作目录")
@click.option("--dry-run", is_flag=True, default=False, help="仅预览启动配置，不实际启动")
@click.option("--list", "list_agents", is_flag=True, default=False, help="列出所有可启动的 Agent")
def launch(
    agent_name: str | None,
    provider: str | None,
    model: str | None,
    project: str | None,
    dry_run: bool,
    list_agents: bool,
) -> None:
    """启动 AI Agent.

    \b
    示例:
      lmswitch launch claude-code                           # 使用 YAML 绑定的 Provider
      lmswitch launch codex --provider deepseek             # 直接指定 Provider
      lmswitch launch claude-code --model claude-opus-4-8   # 覆盖模型
      lmswitch launch claude-code --dry-run                 # 预览配置
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
    # 1. --provider 显式指定 > 2. YAML agent 绑定 > 3. 默认 provider
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
        provider_key = config.agents.get(agent_name, AgentConfig(name=AgentType(agent_name), provider="")).provider
        if not provider_key:
            provider_key = config.default_provider
        if not provider_key:
            click.secho("未指定 Provider。请用 --provider 或先配置默认 Provider", fg="red")
            click.echo("  lmswitch provider add <name> --api-base <url> --api-key <key>")
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

    if dry_run:
        launcher.launch(resolved, dry_run=True)
        return

    click.echo(f"  Agent:    {adapter.display_name} ({adapter.name.value})")
    click.echo(f"  Provider: {provider_key}")
    click.echo(f"  Format:   {resolved.effective_api_format}")
    click.echo(f"  Endpoint: {resolved.effective_api_base}")
    click.echo(f"  Model:    {resolved.agent.model}")
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
