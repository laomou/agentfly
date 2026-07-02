"""CLI Tab 补全的共享助手 — Provider / Model 名称补全.

各子命令的 provider/model 补全逻辑一致, 仅「provider 名来自哪个参数」不同
(launch 用 --provider, test 用位置参数 target), 故 model 补全做成工厂.
"""

from __future__ import annotations

from typing import Any, Callable

import click

from agentfly.core.config import ensure_config_exists


def complete_providers(
    ctx: click.Context, param: click.Parameter, incomplete: str
) -> list[Any]:
    """补全已配置的 Provider 名称."""
    config, _ = ensure_config_exists()
    return [
        click.shell_completion.CompletionItem(name)
        for name in config.providers if name.startswith(incomplete)
    ]


def model_completer(provider_param: str) -> Callable[..., list[Any]]:
    """构造 model 补全器: 从 ctx.params[provider_param] 定位 Provider 再列模型.

    前缀补全. 带通配符 (如 name*) 时无模型以字面 '*' 开头, 自然返回空 —
    不补全, 从而保留用户输入的 '*' 不被 shell 用公共前缀改写掉.
    """
    def _complete(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[Any]:
        config, _ = ensure_config_exists()
        pc = config.providers.get(ctx.params.get(provider_param, ""))
        if not pc:
            return []
        return [
            click.shell_completion.CompletionItem(m)
            for m in pc.model_names if m.startswith(incomplete)
        ]
    return _complete
