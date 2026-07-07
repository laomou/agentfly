"""agentfly test 命令测试 — 纯函数 + 命令派发."""

from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

from agentfly.cli import test_cmd as tc
from agentfly.cli.test_cmd import _clear_api_type, _expand_models, _icon, _pad, _resolve, _summary, _tps, test
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



class TestTps:
    """_tps 吞吐量格式化 (tokens/s, 非时间)."""

    def test_zero_and_negative(self):
        assert _tps(0) == "-"
        assert _tps(-1) == "-"

    def test_format(self):
        assert _tps(6.0) == "6.0"
        assert _tps(30.0) == "30.0"
        assert _tps(123.4) == "123.4"



def _cfg():
    return UnifiedConfig(providers={
        "deepseek": ProviderConfig(
            name=ProviderType.DEEPSEEK, api_key="sk-x",
            endpoints={"openai": "https://api.deepseek.com"},
            models=["m1", "m2"], default_model="m1",
        )
    })


class _FakeProvider:
    """不发网络请求，返回固定 TestResult."""

    def list_models(self):
        return ["m1", "m2"]

    def test_model(self, model, api_key=None, provider_key="", timeout=30.0):
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
        r = CliRunner().invoke(test, ["deepseek", "m1"])
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
        r = CliRunner().invoke(test, ["deepseek", "m1", "--format", "json"])
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert data[0]["model"] == "m1"
        assert data[0]["status"] == "ok"

    def test_wildcard_matches_multiple(self, monkeypatch):
        self._patch(monkeypatch)
        r = CliRunner().invoke(test, ["deepseek", "m*"])
        assert r.exit_code == 0
        assert "m1" in r.output and "m2" in r.output

    def test_wildcard_no_match_errors(self, monkeypatch):
        self._patch(monkeypatch)
        r = CliRunner().invoke(test, ["deepseek", "zzz*"])
        assert r.exit_code != 0
        assert "未找到匹配模型" in r.output


class TestExpandModels:
    """_expand_models: 通配符按 fnmatch 展开, 否则精确单模型."""

    @staticmethod
    def _pc():
        return ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k",
            endpoints={"openai": "http://x"},
            models=["glm-5.2", "glm-5.2-jigan", "glm-5.2:ksyun", "deepseek-v3"],
        )

    def test_exact_no_wildcard(self):
        assert _expand_models(self._pc(), _FakeProvider(), "glm-5.2") == ["glm-5.2"]

    def test_exact_returns_literal_even_if_absent(self):
        # 无通配符时原样返回, 不校验存在性 (交给后续请求报错)
        assert _expand_models(self._pc(), _FakeProvider(), "nope") == ["nope"]

    def test_prefix_star(self):
        assert _expand_models(self._pc(), _FakeProvider(), "glm-5.2*") == [
            "glm-5.2", "glm-5.2-jigan", "glm-5.2:ksyun",
        ]

    def test_suffix_star(self):
        assert _expand_models(self._pc(), _FakeProvider(), "*jigan") == ["glm-5.2-jigan"]

    def test_infix_star(self):
        assert _expand_models(self._pc(), _FakeProvider(), "*5.2*") == [
            "glm-5.2", "glm-5.2-jigan", "glm-5.2:ksyun",
        ]

    def test_question_mark(self):
        assert _expand_models(self._pc(), _FakeProvider(), "glm-?.2") == ["glm-5.2"]

    def test_no_match_returns_empty(self):
        assert _expand_models(self._pc(), _FakeProvider(), "zzz*") == []


class TestSummary:
    """_summary 按状态计数."""

    @staticmethod
    def _r(status):
        return TestResult(provider="p", model="m", status=status)

    def test_counts_ordered(self):
        results = [self._r("ok"), self._r("ok"), self._r("timeout"), self._r("error")]
        assert _summary(results) == "2 ok, 1 timeout, 1 error"

    def test_single_status(self):
        assert _summary([self._r("ok")]) == "1 ok"


class TestRefresh:
    """--refresh 清空 api_type 缓存."""

    def test_clear_api_type(self):
        pc = ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k", endpoints={"openai": "http://x"},
            models={"m1": "openai", "m2": ""},
        )
        _clear_api_type(pc)
        assert all(v == "" for v in pc.models.values())


class TestParallelAndTimeout:
    """并发执行 + timeout 透传."""

    def test_parallel_runs_all_models(self, monkeypatch):
        monkeypatch.setattr(tc, "ensure_config_exists", lambda: (_cfg(), "p"))
        monkeypatch.setattr(tc, "get_provider", lambda pc: _FakeProvider())
        r = CliRunner().invoke(test, ["deepseek", "--parallel", "2"])
        assert r.exit_code == 0
        assert "m1" in r.output and "m2" in r.output
        assert "2 ok" in r.output  # 汇总行

    def test_timeout_passed_through(self, monkeypatch):
        seen = {}

        class RecordingProvider(_FakeProvider):
            def test_model(self, model, api_key=None, provider_key="", timeout=30.0):
                seen["timeout"] = timeout
                return super().test_model(model, api_key, provider_key, timeout)

        monkeypatch.setattr(tc, "ensure_config_exists", lambda: (_cfg(), "p"))
        monkeypatch.setattr(tc, "get_provider", lambda pc: RecordingProvider())
        r = CliRunner().invoke(test, ["deepseek", "m1", "-t", "7"])
        assert r.exit_code == 0
        assert seen["timeout"] == 7.0
