"""Shell 集成工具."""

from __future__ import annotations


def export_cmd(env_vars: dict[str, str]) -> str:
    """生成 shell export 命令.

    >>> export_cmd({"FOO": "bar", "BAZ": "qux"})
    'export FOO="bar"\\nexport BAZ="qux"'
    """
    lines = []
    for k, v in env_vars.items():
        escaped = v.replace('"', '\\"')
        lines.append(f'export {k}="{escaped}"')
    return "\n".join(lines)


def is_interactive() -> bool:
    """检测是否为交互式终端."""
    import sys
    return sys.stdout.isatty()
