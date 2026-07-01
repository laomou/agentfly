"""[agentfly launch] 启动 AI Agent."""

from __future__ import annotations

import sys
from typing import Any

import click

from agentfly import __version__
from agentfly.agents.registry import get_registry
from agentfly.core.config import ensure_config_exists
from agentfly.core.launcher import AgentLauncher, LaunchError
from agentfly.core.resolver import ConfigResolver
from agentfly.models.schema import AgentConfig, ProviderConfig, UnifiedConfig
from agentfly.models.types import AgentType


def _complete_agents(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[Any]:
    """Tab 补全: Agent 名称."""
    registry = get_registry()
    return [
        click.shell_completion.CompletionItem(a.name.value, help=a.display_name)
        for a in registry.list() if a.name.value.startswith(incomplete)
    ]


@click.command(name="launch")
@click.argument("agent_name", required=False, shell_complete=_complete_agents)
@click.option("--provider", "-P", default=None, help="指定 Provider (覆盖 YAML 绑定)")
@click.option("--model", "-m", default=None, help="覆盖默认模型")
@click.option("--project", "-p", default=None, help="指定项目/工作目录")
@click.option("--list", "list_agents", is_flag=True, default=False, help="列出所有可启动的 Agent")
@click.argument("agent_args", nargs=-1)
def launch(
    agent_name: str | None,
    provider: str | None,
    model: str | None,
    project: str | None,
    list_agents: bool,
    agent_args: tuple[str, ...] = (),
) -> None:
    """启动 AI Agent.

    \b
    示例:
      agentfly launch claude                                # 使用 YAML 绑定的 Provider
      agentfly launch codex --provider deepseek             # 直接指定 Provider
      agentfly launch claude --model claude-opus-4-8        # 覆盖模型
      agentfly launch --list                                # 列出 Agent
    """
    registry = get_registry()

    if list_agents:
        _list_available_agents(registry)
        return

    if not agent_name:
        click.secho("用法: agentfly launch <agent-name> [--provider <name>]", fg="yellow")
        click.echo("  agentfly launch --list    查看可用 Agent")
        sys.exit(1)

    adapter = registry.get(agent_name)
    if adapter is None:
        click.secho(f"未找到 Agent '{agent_name}'  可用: {registry.names()}", fg="red")
        sys.exit(1)

    # 加载配置
    config, _ = ensure_config_exists()

    # ── provider 选择逻辑 ──
    # 优先级: --provider 显式指定 > YAML agent 绑定 > 交互式选择
    if provider:
        provider_key = provider
        # config 里没有这个 agent 时临时注入（不写回 YAML）
        if agent_name not in config.agents:
            config.agents[agent_name] = AgentConfig(
                name=AgentType(agent_name),
                provider=provider_key,
                model=model,
            )
    else:
        agent_cfg = config.agents.get(agent_name)
        if agent_cfg and agent_cfg.provider:
            provider_key = agent_cfg.provider
        else:
            # 既没显式指定也没绑定 → 让用户选 Provider（必要时再选 Model）
            provider_key = _select_provider(config, agent_name, adapter.preferred_format)
            if model is None:
                model = _select_model(config.providers[provider_key])

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

    click.echo(f"  AgentFly v{__version__}")
    click.echo(f"  {adapter.display_name} · {provider_key} · {resolved.agent.model or '默认模型'}")
    click.echo()

    try:
        exit_code = launcher.launch(resolved, cwd=project, extra_args=list(agent_args))
        sys.exit(exit_code)
    except LaunchError as e:
        click.secho(f"启动失败: {e}", fg="red")
        sys.exit(1)


def _prompt_select(label: str, items: list[str], default: str | None = None) -> str:
    """打印编号菜单让用户选择，返回选中项.

    default 命中某项时，回车即选中该项.
    """
    default_idx: int | None = None
    click.echo(f"  选择 {label}:")
    for i, item in enumerate(items, 1):
        suffix = ""
        if default and item == default:
            suffix = "  (默认)"
            default_idx = i
        click.echo(f"    {i}) {item}{suffix}")
    idx = click.prompt(
        "  输入编号",
        type=click.IntRange(1, len(items)),
        default=default_idx,
        show_default=default_idx is not None,
    )
    return items[idx - 1]


def _select_provider(config: UnifiedConfig, agent_name: str, preferred_format: str) -> str:
    """无 --provider 且无 YAML 绑定时，选择 Provider.

    只在「支持 agent 所需格式」的 Provider 中挑选:
      - 0 个   → 报错退出
      - 1 个   → 自动选用
      - 多个   → TTY 弹菜单选择; 非 TTY 报错要求显式 --provider
    """
    if not config.providers:
        click.secho("无可用 Provider。请运行 'agentfly provider add'", fg="red")
        sys.exit(1)

    compatible = [k for k, p in config.providers.items() if preferred_format in p.endpoints]
    if not compatible:
        click.secho(
            f"没有支持 {preferred_format} 格式的 Provider。"
            f"已配置: {', '.join(config.providers)}",
            fg="red",
        )
        sys.exit(1)

    if len(compatible) == 1:
        click.echo(f"  使用 Provider: {compatible[0]}")
        return compatible[0]

    if not sys.stdin.isatty():
        click.secho(
            f"'{agent_name}' 未绑定 Provider，且未指定 --provider。"
            f"请用 --provider 指定 (可选: {', '.join(compatible)})",
            fg="red",
        )
        sys.exit(1)

    return _prompt_select("Provider", compatible)


def _select_model(provider_cfg: ProviderConfig) -> str | None:
    """Provider 选定后选择 Model.

    返回 None 表示沿用 Provider 的 default_model:
      - 模型 ≤1 个 或 非 TTY → None
      - 多个 → 弹菜单, default_model 高亮为默认项
    """
    models = provider_cfg.models
    if len(models) <= 1 or not sys.stdin.isatty():
        return None
    default = provider_cfg.default_model or models[0]
    return _prompt_select("Model", models, default=default)


def _list_available_agents(registry) -> None:
    """列出所有可用的 Agent."""
    click.echo("可用的 AI Agent:")
    click.echo()
    for adapter in registry.list():
        click.echo(f"  {adapter.name.value:<18} {adapter.display_name}")
        click.echo(f"    需要的 API 格式: {adapter.preferred_format}")
        click.echo()
