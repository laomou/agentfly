"""provider 子命令流程测试 (list/show/remove/add/reload)."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from agentfly.cli import provider_cmd as pc
from agentfly.cli.provider_cmd import (
    add_provider,
    list_providers,
    reload_models,
    remove_provider,
    show_provider,
)
from agentfly.models.schema import ProviderConfig, UnifiedConfig
from agentfly.models.types import ProviderType


def _cfg():
    return UnifiedConfig(providers={
        "deepseek": ProviderConfig(
            name=ProviderType.DEEPSEEK, api_key="${DS_KEY}",
            endpoints={"openai": "http://x"},
            models=["m1", "m2"], default_model="m1",
        )
    })


@pytest.fixture
def tmp_cfg_env(tmp_path, monkeypatch):
    """让任何 save_config 写到 tmp，不污染真实 ~/.config."""
    monkeypatch.setenv("LMSWITCH_CONFIG", str(tmp_path / "config.yaml"))


class TestListShow:
    def test_list(self, monkeypatch):
        monkeypatch.setattr(pc, "ensure_config_exists", lambda: (_cfg(), "p"))
        r = CliRunner().invoke(list_providers)
        assert r.exit_code == 0
        assert "deepseek" in r.output

    def test_list_empty(self, monkeypatch):
        monkeypatch.setattr(pc, "ensure_config_exists", lambda: (UnifiedConfig(), "p"))
        r = CliRunner().invoke(list_providers)
        assert r.exit_code == 0
        assert "暂无" in r.output

    def test_show(self, monkeypatch):
        monkeypatch.setattr(pc, "ensure_config_exists", lambda: (_cfg(), "p"))
        r = CliRunner().invoke(show_provider, ["deepseek"])
        assert r.exit_code == 0
        assert "deepseek" in r.output and "m1" in r.output

    def test_show_unknown_exits(self, monkeypatch):
        monkeypatch.setattr(pc, "ensure_config_exists", lambda: (_cfg(), "p"))
        r = CliRunner().invoke(show_provider, ["ghost"])
        assert r.exit_code != 0


class TestRemove:
    def test_remove(self, tmp_cfg_env, monkeypatch):
        monkeypatch.setattr(pc, "ensure_config_exists", lambda: (_cfg(), "p"))
        r = CliRunner().invoke(remove_provider, ["deepseek"])
        assert r.exit_code == 0
        assert "已移除" in r.output

    def test_remove_unknown_exits(self, tmp_cfg_env, monkeypatch):
        monkeypatch.setattr(pc, "ensure_config_exists", lambda: (_cfg(), "p"))
        r = CliRunner().invoke(remove_provider, ["ghost"])
        assert r.exit_code != 0


class TestAdd:
    def test_add_env_key(self, tmp_cfg_env, monkeypatch):
        # env-var key 跳过网络探测；显式 --models 跳过拉取；endpoints 空 → 提示输入格式
        monkeypatch.setattr(pc, "_get_known_providers", lambda: {})
        monkeypatch.setattr(pc, "_refresh_env_cache", lambda key: None)
        r = CliRunner().invoke(
            add_provider,
            ["myproxy", "--api-base", "http://h:3000",
             "--api-key", "${MYKEY}", "--models", "a,b"],
            input="openai\n",
        )
        assert r.exit_code == 0, r.output
        assert "myproxy" in r.output


class TestReload:
    def test_reload(self, tmp_cfg_env, monkeypatch):
        monkeypatch.setattr(pc, "ensure_config_exists", lambda: (_cfg(), "p"))
        monkeypatch.setattr(pc, "_fetch_models", lambda base, key: ["x", "y"])
        monkeypatch.setattr(pc, "_refresh_env_cache", lambda key: None)
        r = CliRunner().invoke(reload_models, ["deepseek"])
        assert r.exit_code == 0
        assert "x" in r.output and "y" in r.output

    def test_reload_unknown_exits(self, monkeypatch):
        monkeypatch.setattr(pc, "ensure_config_exists", lambda: (_cfg(), "p"))
        r = CliRunner().invoke(reload_models, ["ghost"])
        assert r.exit_code != 0
