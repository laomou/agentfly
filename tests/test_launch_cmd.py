"""launch 命令的 Provider/Model 选择逻辑测试."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentfly.cli import launch as launch_mod
from agentfly.cli.launch import (
    _match_name,
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
        # 只有一个支持 anthropic → 自动选它
        cfg = UnifiedConfig(providers={
            "oa": ProviderConfig(name=ProviderType.OPENAI, api_key="k",
                                 endpoints={"openai": "http://x"}, models=["m1"]),
            "an": ProviderConfig(name=ProviderType.ANTHROPIC, api_key="k",
                                 endpoints={"anthropic": "http://x"}, models=["m1"]),
        })
        assert _select_provider(cfg, "claude", "anthropic") == "an"

    def test_no_compatible_exits(self):
        cfg = UnifiedConfig(providers={
            "oa": ProviderConfig(name=ProviderType.OPENAI, api_key="k",
                                 endpoints={"openai": "http://x"}, models=["m1"]),
        })
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
        monkeypatch.setattr("agentfly.cli.launch.click.prompt", lambda *a, **k: "2")
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
        monkeypatch.setattr("agentfly.cli.launch.click.prompt", lambda *a, **k: "2")
        assert _select_model(_provider(models=["m1", "m2"], default="m1")) == "m2"


class TestMatchName:
    """_match_name 名称匹配逻辑."""

    def test_exact_match(self):
        assert _match_name("foo", ["foo", "bar", "baz"]) == "foo"

    def test_exact_match_case_insensitive(self):
        assert _match_name("FOO", ["foo", "bar"]) == "foo"

    def test_prefix_match(self):
        assert _match_name("foz", ["foo", "bar", "foz"]) == "foz"

    def test_prefix_ambiguous_returns_none(self):
        assert _match_name("f", ["foo", "bar", "foz"]) is None

    def test_substring_match(self):
        assert _match_name("oo", ["foo", "bar"]) == "foo"

    def test_substring_ambiguous_returns_none(self):
        assert _match_name("o", ["foo", "foz", "bar"]) is None

    def test_no_match_returns_none(self):
        assert _match_name("x", ["foo", "bar"]) is None

    def test_empty_text_returns_none(self):
        assert _match_name("", ["foo", "bar"]) is None


class TestPromptSelect:
    """_prompt_select 编号/名称 → 选项 映射."""

    def test_number_selects_item(self, monkeypatch):
        monkeypatch.setattr("agentfly.cli.launch.click.prompt", lambda *a, **k: "1")
        assert _prompt_select("X", ["foo", "bar"]) == "foo"

    def test_name_selects_item(self, monkeypatch):
        monkeypatch.setattr("agentfly.cli.launch.click.prompt", lambda *a, **k: "bar")
        assert _prompt_select("X", ["foo", "bar"]) == "bar"

    def test_name_case_insensitive(self, monkeypatch):
        monkeypatch.setattr("agentfly.cli.launch.click.prompt", lambda *a, **k: "BAR")
        assert _prompt_select("X", ["foo", "bar"]) == "bar"

    def test_name_prefix(self, monkeypatch):
        monkeypatch.setattr("agentfly.cli.launch.click.prompt", lambda *a, **k: "ba")
        assert _prompt_select("X", ["foo", "bar"]) == "bar"

    def test_invalid_then_valid_retries(self, monkeypatch):
        """首次无效，二次有效."""
        calls = iter(["99", "2"])

        def fake(*a, **k):
            return next(calls)

        monkeypatch.setattr("agentfly.cli.launch.click.prompt", fake)
        assert _prompt_select("X", ["foo", "bar", "baz"]) == "bar"

    def test_default_index_points_to_default(self, monkeypatch):
        captured: dict = {}

        def fake_prompt(*a, **k):
            captured.update(k)
            return k.get("default")

        monkeypatch.setattr("agentfly.cli.launch.click.prompt", fake_prompt)
        # default="bar" 是第 2 项，回车应选中它
        assert _prompt_select("X", ["foo", "bar"], default="bar") == "bar"
        assert captured["default"] == "2"


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

    def test_explicit_provider_no_model_selects_model(self, monkeypatch, tty):
        """--provider 显式指定但无 --model, TTY 多模型 → 调 _select_model 选模型 (回归).

        旧逻辑: --provider 分支完全跳过模型选择, 直接用 default_model.
        新逻辑: provider 已定且无 --model → 调 _select_model.
        """
        cfg = UnifiedConfig(
            providers={"prox": _provider("anthropic", models=["m1", "m2"], default="m1")},
            agents={},
        )
        monkeypatch.setattr(launch_mod, "get_registry", lambda: _FakeRegistry())
        monkeypatch.setattr(launch_mod, "ensure_config_exists", lambda: (cfg, Path("cfg")))
        monkeypatch.setattr(launch_mod, "AgentLauncher", _FakeLauncher)
        # CliRunner 会隔离 stdin 使 isatty()=False, 直接 mock _select_model 验证它被调用
        monkeypatch.setattr(launch_mod, "_select_model", lambda pc: "m2")
        r = CliRunner().invoke(launch, ["claude", "-P", "prox"])
        assert r.exit_code == 0
        assert _FakeLauncher.last["provider_key"] == "prox"
        assert _FakeLauncher.last["model"] == "m2"

