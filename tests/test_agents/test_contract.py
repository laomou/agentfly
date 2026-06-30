"""全 Agent adapter 契约测试 — 防止 env_vars 产出非法 key 或空启动命令.

参数化扫描所有注册的 adapter，统一断言契约。这类测试能拦住
openclaw.py 那种 `env[<URL>] = <URL>` 的 env-var 名 bug。
"""

from __future__ import annotations

import re

import pytest

from lmswitch.agents.claude import Claude
from lmswitch.agents.claude_code import ClaudeCode
from lmswitch.agents.cline import Cline
from lmswitch.agents.codex import Codex
from lmswitch.agents.droid import Droid
from lmswitch.agents.opencode import OpenCode
from lmswitch.agents.openclaw import OpenClaw
from lmswitch.agents.pi import Pi
from lmswitch.models.schema import AgentConfig, ProviderConfig, ResolvedConfig
from lmswitch.models.types import ProviderType

# POSIX 环境变量名: 字母/下划线开头，后续字母数字下划线
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

ADAPTERS = [Claude(), ClaudeCode(), Cline(), Codex(), Droid(), OpenCode(), OpenClaw(), Pi()]


def _resolved_for(adapter) -> ResolvedConfig:
    """按 adapter 需要的格式造一个可用的 ResolvedConfig."""
    fmt = adapter.preferred_format
    if fmt == "anthropic":
        ptype, base = ProviderType.ANTHROPIC, "https://api.anthropic.com"
    else:
        ptype, base = ProviderType.OPENAI, "https://api.openai.com"
    return ResolvedConfig(
        agent=AgentConfig(name=adapter.name, provider=ptype.value, model="m1"),
        provider=ProviderConfig(
            name=ptype, api_key="sk-x", endpoints={fmt: base},
            models=["m1"], default_model="m1",
        ),
        effective_api_base=base,
        effective_api_format=fmt,
    )


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name.value)
class TestAdapterContract:
    """所有 adapter 必须满足的统一契约."""

    def test_launch_command_nonempty(self, adapter):
        cmd = adapter.launch_command(_resolved_for(adapter))
        assert isinstance(cmd, list) and cmd, "launch_command 不能为空"
        assert isinstance(cmd[0], str) and cmd[0], "首元素必须是非空可执行名"

    def test_env_var_keys_are_valid_names(self, adapter):
        env = adapter.env_vars(_resolved_for(adapter))
        assert isinstance(env, dict)
        for k, v in env.items():
            assert _ENV_NAME.match(k), f"{adapter.name.value}: 非法环境变量名 {k!r}"
            assert isinstance(v, str), f"{adapter.name.value}: {k} 的值不是 str"


def test_cline_anthropic_branch():
    """Cline 在 anthropic 格式下走 ANTHROPIC_* 分支."""
    from lmswitch.agents.cline import Cline
    from lmswitch.models.types import AgentType

    rc = ResolvedConfig(
        agent=AgentConfig(name=AgentType.CLINE, provider="anthropic", model="m1"),
        provider=ProviderConfig(
            name=ProviderType.ANTHROPIC, api_key="sk-x",
            endpoints={"anthropic": "https://api.anthropic.com"}, models=["m1"]),
        effective_api_base="https://api.anthropic.com",
        effective_api_format="anthropic",
    )
    env = Cline().env_vars(rc)
    assert env["ANTHROPIC_API_KEY"] == "sk-x"
    assert env["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
