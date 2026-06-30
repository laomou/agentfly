"""lmswitch doctor 测试 — 含 ${VAR} key 的回归保护 (曾因缺 import os 崩溃)."""

from __future__ import annotations

from click.testing import CliRunner

from lmswitch.cli import doctor as doctor_mod
from lmswitch.cli.doctor import doctor
from lmswitch.models.schema import ProviderConfig, UnifiedConfig
from lmswitch.models.types import ProviderType


def _cfg(api_key, endpoints=None, models=None):
    return UnifiedConfig(providers={
        "deepseek": ProviderConfig(
            name=ProviderType.DEEPSEEK, api_key=api_key,
            endpoints=endpoints if endpoints is not None else {"openai": "http://x"},
            models=models if models is not None else ["m1"],
        )
    })


class TestDoctor:
    def test_env_var_set(self, monkeypatch):
        monkeypatch.setenv("DS_KEY", "secret")
        monkeypatch.setattr(doctor_mod, "ensure_config_exists",
                            lambda: (_cfg("${DS_KEY}"), "p"))
        r = CliRunner().invoke(doctor)
        assert r.exit_code == 0
        assert "DS_KEY" in r.output and "✓" in r.output

    def test_env_var_unset_is_issue(self, monkeypatch):
        monkeypatch.delenv("DS_KEY", raising=False)
        monkeypatch.setattr(doctor_mod, "ensure_config_exists",
                            lambda: (_cfg("${DS_KEY}"), "p"))
        r = CliRunner().invoke(doctor)
        assert r.exit_code == 0
        assert "未设置" in r.output

    def test_no_providers(self, monkeypatch):
        monkeypatch.setattr(doctor_mod, "ensure_config_exists",
                            lambda: (UnifiedConfig(), "p"))
        r = CliRunner().invoke(doctor)
        assert r.exit_code == 0
        assert "无 Provider" in r.output

    def test_missing_endpoints_and_models_are_issues(self, monkeypatch):
        monkeypatch.setattr(doctor_mod, "ensure_config_exists",
                            lambda: (_cfg("plain-key", endpoints={}, models=[]), "p"))
        r = CliRunner().invoke(doctor)
        assert r.exit_code == 0
        assert "无 endpoint" in r.output
        assert "无模型" in r.output
