"""launch 命令的 Provider/Model 选择逻辑测试."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentfly.cli import launch as launch_mod
from agentfly.cli.launch import (
    _prompt_select,
    _select_model,
    _select_provider,
    launch,
)
from agentfly.models.schema import AgentConfig, ProviderConfig, UnifiedConfig
from agentfly.models.types import AgentType, ProviderType


def _provider(fmt: str = "openai", models: list[str] | None = None,
              default: str = "") -> ProviderConfig:
    return ProviderConfig(
        name=ProviderType.CUSTOM,
        api_key="sk-x",
        endpoints={fmt: "http://example"},
        models=models or [],
        default_model=default,
    )


class _FakeStdin:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture
def tty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(True))


@pytest.fixture
def no_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(False))


class TestSelectProvider:
    """_select_provider 选择/兜底逻辑（仅在无绑定时触达）."""

    def test_single_compatible_autopicks(self):
        cfg = UnifiedConfig(providers={"a": _provider("openai")})
        assert _select_provider(cfg, "codex", "openai") == "a"

    def test_filters_by_format(self):
        # 只有一个支持 anthropic → 自动选它，跳过 openai-only 的
        cfg = UnifiedConfig(providers={
            "oa": _provider("openai"),
            "an": _provider("anthropic"),
        })
        assert _select_provider(cfg, "claude", "anthropic") == "an"

    def test_no_compatible_exits(self):
        cfg = UnifiedConfig(providers={"oa": _provider("openai")})
        with pytest.raises(SystemExit):
            _select_provider(cfg, "claude", "anthropic")

    def test_no_providers_exits(self):
        with pytest.raises(SystemExit):
            _select_provider(UnifiedConfig(providers={}), "codex", "openai")

    def test_multiple_non_tty_exits(self, no_tty):
        cfg = UnifiedConfig(providers={"a": _provider(), "b": _provider()})
        with pytest.raises(SystemExit):
            _select_provider(cfg, "codex", "openai")

    def test_multiple_tty_prompts(self, tty, monkeypatch):
        # 模拟用户选第 2 项
        monkeypatch.setattr("agentfly.cli.launch.click.prompt", lambda *a, **k: 2)
        cfg = UnifiedConfig(providers={"a": _provider(), "b": _provider()})
        assert _select_provider(cfg, "codex", "openai") == "b"


class TestSelectModel:
    """_select_model 选择/兜底逻辑."""

    def test_empty_models_returns_none(self, tty):
        assert _select_model(_provider(models=[])) is None

    def test_single_model_returns_none(self, tty):
        assert _select_model(_provider(models=["m1"])) is None

    def test_non_tty_returns_none(self, no_tty):
        assert _select_model(_provider(models=["m1", "m2"])) is None

    def test_multiple_tty_prompts(self, tty, monkeypatch):
        monkeypatch.setattr("agentfly.cli.launch.click.prompt", lambda *a, **k: 2)
        assert _select_model(_provider(models=["m1", "m2"], default="m1")) == "m2"


class TestPromptSelect:
    """_prompt_select 编号 → 选项 映射."""

    def test_returns_indexed_item(self, monkeypatch):
        monkeypatch.setattr("agentfly.cli.launch.click.prompt", lambda *a, **k: 1)
        assert _prompt_select("X", ["foo", "bar"]) == "foo"

    def test_default_index_points_to_default(self, monkeypatch):
        captured: dict = {}

        def fake_prompt(*a, **k):
            captured.update(k)
            return k.get("default")

        monkeypatch.setattr("agentfly.cli.launch.click.prompt", fake_prompt)
        # default="bar" 是第 2 项，回车应选中它
        assert _prompt_select("X", ["foo", "bar"], default="bar") == "bar"
        assert captured["default"] == 2


class _FakeAdapter:
    preferred_format = "anthropic"
    display_name = "Claude Code"
    name = AgentType.CLAUDE


class _FakeRegistry:
    def get(self, name):
        return _FakeAdapter()

    def list(self):
        return [_FakeAdapter()]

    def names(self):
        return ["claude"]


class _FakeLauncher:
    """记录 resolved 配置，不真正起进程."""

    last: dict = {}

    def __init__(self, adapter):
        pass

    def launch(self, resolved, cwd=None, extra_args=None):
        _FakeLauncher.last = {
            "provider_key": resolved.agent.provider,
            "model": resolved.agent.model,
        }
        return 0


class TestLaunchBinding:
    """launch() 层：YAML 绑定优先于交互式选择."""

    def test_bound_agent_skips_selection(self, monkeypatch):
        # 两个都支持 anthropic 的 Provider；若忽略绑定，非 TTY 多选会报错退出
        cfg = UnifiedConfig(
            providers={
                "anthropic": _provider("anthropic", models=["m1"], default="m1"),
                "anthropic2": _provider("anthropic", models=["m1"], default="m1"),
            },
            agents={
                "claude": AgentConfig(
                    name=AgentType.CLAUDE, provider="anthropic", model="bound-model",
                ),
            },
        )
        monkeypatch.setattr(launch_mod, "get_registry", lambda: _FakeRegistry())
        monkeypatch.setattr(launch_mod, "ensure_config_exists", lambda: (cfg, Path("cfg")))
        monkeypatch.setattr(launch_mod, "AgentLauncher", _FakeLauncher)

        result = CliRunner().invoke(launch, ["claude"])

        assert result.exit_code == 0
        assert _FakeLauncher.last["provider_key"] == "anthropic"
        assert _FakeLauncher.last["model"] == "bound-model"


class _EmptyRegistry:
    def get(self, name):
        return None

    def names(self):
        return ["claude"]


class TestLaunchCommand:
    """launch 命令的各入口路径."""

    def test_list_agents(self, monkeypatch):
        monkeypatch.setattr(launch_mod, "get_registry", lambda: _FakeRegistry())
        r = CliRunner().invoke(launch, ["--list"])
        assert r.exit_code == 0
        assert "claude" in r.output

    def test_no_agent_name_exits(self, monkeypatch):
        monkeypatch.setattr(launch_mod, "get_registry", lambda: _FakeRegistry())
        r = CliRunner().invoke(launch, [])
        assert r.exit_code == 1

    def test_unknown_agent_exits(self, monkeypatch):
        monkeypatch.setattr(launch_mod, "get_registry", lambda: _EmptyRegistry())
        r = CliRunner().invoke(launch, ["claude"])
        assert r.exit_code == 1

    def test_explicit_provider_and_model(self, monkeypatch):
        cfg = UnifiedConfig(
            providers={"prox": _provider("anthropic", models=["x"], default="x")},
            agents={},
        )
        monkeypatch.setattr(launch_mod, "get_registry", lambda: _FakeRegistry())
        monkeypatch.setattr(launch_mod, "ensure_config_exists", lambda: (cfg, Path("cfg")))
        monkeypatch.setattr(launch_mod, "AgentLauncher", _FakeLauncher)
        r = CliRunner().invoke(launch, ["claude", "-P", "prox", "-m", "mymodel"])
        assert r.exit_code == 0
        assert _FakeLauncher.last["provider_key"] == "prox"
        assert _FakeLauncher.last["model"] == "mymodel"

