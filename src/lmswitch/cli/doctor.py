"""[lmswitch doctor] 配置健康检查."""

from __future__ import annotations

import click

from lmswitch.core.config import ensure_config_exists


@click.command(name="doctor")
def doctor():
    """检查配置健康状态."""
    config, path = ensure_config_exists()
    issues: list[str] = []

    click.echo(f"  配置文件: {path}")
    click.echo(f"  版本:     {config.version}")

    # ── 环境变量 ──
    click.echo()
    click.echo("  环境变量:")
    for pk, pc in config.providers.items():
        key = pc.api_key
        if key.startswith("${") and key.endswith("}"):
            var_name = key[2:-1]
            if os.environ.get(var_name):
                click.secho(f"    ✓ {var_name} 已设置", fg="green")
            else:
                issues.append(f"{var_name} 未设置 (Provider: {pk})")
                click.secho(f"    ✗ {var_name} 未设置", fg="red")
        else:
            click.echo(f"    {pk}: 明文 Key (建议改用环境变量)")

    # ── Provider ──
    click.echo()
    if config.providers:
        click.echo(f"  Provider ({len(config.providers)}):")
        for pk, pc in config.providers.items():
            n_endpoints = len(pc.endpoints)
            n_models = len(pc.models)
            fmt_list = ", ".join(pc.endpoints.keys())
            click.echo(f"    {pk}: {n_endpoints} endpoint(s) [{fmt_list}], {n_models} model(s)")
            if not pc.endpoints:
                issues.append(f"Provider '{pk}' 无 endpoint")
            if not pc.models:
                issues.append(f"Provider '{pk}' 无模型列表")
    else:
        issues.append("无 Provider 配置")
        click.echo("  无 Provider — 运行 'lmswitch provider add' 添加")

    # ── 总结 ──
    click.echo()
    if issues:
        click.secho(f"  发现 {len(issues)} 个问题:", fg="yellow")
        for i in issues:
            click.secho(f"    • {i}", fg="yellow")
    else:
        click.secho("  ✓ 一切正常", fg="green")
