"""[lmswitch agent] 查看 Agent 绑定."""

from __future__ import annotations

import click

from lmswitch.core.config import ensure_config_exists


@click.group(name="agent")
def agent_group():
    """查看 Agent → Provider 绑定."""
    pass


@agent_group.command(name="list")
def list_agents():
    """列出所有 Agent 绑定."""
    config, _ = ensure_config_exists()

    if not config.agents:
        click.echo("无 Agent 绑定. 运行 'lmswitch provider add <name>' 自动创建.")
        return

    click.echo("Agent 绑定:")
    click.echo()
    for name, agent in config.agents.items():
        provider = config.providers.get(agent.provider)
        model_count = len(provider.models) if provider else 0
        click.echo(f"  {name:<18} → {agent.provider}")
        if agent.model:
            click.echo(f"    模型: {agent.model}")
        elif provider:
            click.echo(f"    默认模型: {provider.default_model} ({model_count} 个可用)")
