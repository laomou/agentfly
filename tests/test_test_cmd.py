"""agentfly test 命令测试 — 纯函数 + 命令派发."""

from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

from agentfly.cli import test_cmd as tc
from agentfly.cli.test_cmd import _base, _icon, _pad, _resolve, test
from agentfly.models.schema import ProviderConfig, TestResult, UnifiedConfig
from agentfly.models.types import ProviderType


class TestPad:
    """_pad 数值格式化."""

    def test_zero_and_negative(self):
        assert _pad(0) == "-"
        assert _pad(-5) == "-"

    def test_milliseconds(self):
        assert _pad(500) == "500ms"
        assert _pad(999) == "999ms"

    def test_seconds(self):
        assert _pad(1000) == "1.0s"
        assert _pad(1500) == "1.5s"


class TestIcon:
    """_icon 状态 → emoji."""

    def test_known(self):
        assert _icon("ok") == "✅"
        assert _icon("timeout") == "⏳"
        assert _icon("error") == "❌"
        assert _icon("unauthorized") == "❌"

    def test_unknown(self):
        assert _icon("whatever") == "❓"


class TestBase:
    """_base 取第一个 endpoint."""

    def test_first_endpoint(self):
        pc = ProviderConfig(name=ProviderType.OPENAI, api_key="k",
                            base_url="http://x", models=["m"])
        assert _base(pc) == "http://x"

    def test_empty(self):
        pc = ProviderConfig(name=ProviderType.OPENAI, api_key="k",
                            base_url="", models=["m"])
        assert _base(pc) == ""


def _cfg():
    return UnifiedConfig(providers={
        "deepseek": ProviderConfig(
            name=ProviderType.DEEPSEEK, api_key="sk-x",
            base_url="https://api.deepseek.com",
            models=["m1", "m2"], default_model="m1",
        )
    })


class _FakeProvider:
    """不发网络请求，返回固定 TestResult."""

    def list_models(self):
        return ["m1", "m2"]

    def test_model(self, model, api_key=None, api_base=None, provider_key=""):
        return TestResult(
            provider=provider_key or "deepseek", model=model, status="ok",
            latency_ms=120.0, ttft_ms=50.0, tokens_per_sec=30.0,
        )


class TestResolve:
    """_resolve 错误路径."""

    def test_unknown_provider_raises(self):
        with pytest.raises(click.ClickException):
            _resolve(_cfg(), "nope")

    def test_unsupported_provider_type_raises(self, monkeypatch):
        # get_provider 返回 None → 不支持的 Provider 类型
        monkeypatch.setattr(tc, "get_provider", lambda pc: None)
        with pytest.raises(click.ClickException):
            _resolve(_cfg(), "deepseek")


class TestCommand:
    """命令派发：单模型 / 单 Provider / 全部 / JSON."""

    def _patch(self, monkeypatch):
        monkeypatch.setattr(tc, "ensure_config_exists", lambda: (_cfg(), "p"))
        monkeypatch.setattr(tc, "get_provider", lambda pc: _FakeProvider())

    def test_single_model(self, monkeypatch):
        self._patch(monkeypatch)
        r = CliRunner().invoke(test, ["deepseek:m1"])
        assert r.exit_code == 0
        assert "m1" in r.output and "deepseek" in r.output

    def test_single_provider_stream(self, monkeypatch):
        self._patch(monkeypatch)
        r = CliRunner().invoke(test, ["deepseek"])
        assert r.exit_code == 0
        assert "m1" in r.output and "m2" in r.output

    def test_all_providers(self, monkeypatch):
        self._patch(monkeypatch)
        r = CliRunner().invoke(test, [])
        assert r.exit_code == 0
        assert "deepseek" in r.output

    def test_json_format(self, monkeypatch):
        self._patch(monkeypatch)
        r = CliRunner().invoke(test, ["deepseek:m1", "--format", "json"])
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert data[0]["model"] == "m1"
        assert data[0]["status"] == "ok"
