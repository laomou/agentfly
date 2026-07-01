"""Provider.test_model 流式探测 + SSE 解析测试 (mock httpx)."""

from __future__ import annotations

import httpx
import pytest

from agentfly.models.schema import ProviderConfig
from agentfly.models.types import ProviderType
from agentfly.providers.anthropic import AnthropicProvider
from agentfly.providers.openai import OpenAIProvider


def _openai_provider():
    return OpenAIProvider(ProviderConfig(
        name=ProviderType.OPENAI, api_key="sk-x",
        base_url="https://api.openai.com", models=["gpt-4o"],
    ))


def _anthropic_provider():
    return AnthropicProvider(ProviderConfig(
        name=ProviderType.ANTHROPIC, api_key="k",
        base_url="https://api.anthropic.com", models=["claude-opus-4-8"],
    ))


class _FakeStream:
    """模拟 httpx 的 stream 响应上下文."""

    def __init__(self, status_code, lines):
        self.status_code = status_code
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self):
        yield from self._lines


def _patch_client(monkeypatch, *, stream=None, exc=None):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, *a, **k):
            if exc is not None:
                raise exc
            return stream

    monkeypatch.setattr("agentfly.providers.base.httpx.Client", FakeClient)


class TestTestModel:
    """test_model 各状态分支."""

    def test_unauthorized(self, monkeypatch):
        _patch_client(monkeypatch, stream=_FakeStream(401, []))
        assert _openai_provider().test_model("gpt-4o").status == "unauthorized"

    def test_http_error(self, monkeypatch):
        _patch_client(monkeypatch, stream=_FakeStream(500, []))
        assert _openai_provider().test_model("gpt-4o").status == "error"

    def test_timeout(self, monkeypatch):
        _patch_client(monkeypatch, exc=httpx.TimeoutException("x"))
        assert _openai_provider().test_model("gpt-4o").status == "timeout"

    def test_connect_error(self, monkeypatch):
        _patch_client(monkeypatch, exc=httpx.ConnectError("x"))
        r = _openai_provider().test_model("gpt-4o")
        assert r.status == "error"
        assert "无法连接" in r.error_message

    def test_ok_with_metrics(self, monkeypatch):
        lines = [
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]
        _patch_client(monkeypatch, stream=_FakeStream(200, lines))
        r = _openai_provider().test_model("gpt-4o", provider_key="deepseek")
        assert r.status == "ok"
        assert r.provider == "deepseek"  # provider_key 透传
        assert r.latency_ms >= 0
        assert r.ttft_ms >= 0


class TestParseStreamChunk:
    """SSE 行解析 (OpenAI + Anthropic)."""

    def test_openai_content(self):
        line = 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        assert _openai_provider()._parse_stream_chunk(line) == "hi"

    def test_openai_reasoning_fallback(self):
        line = 'data: {"choices":[{"delta":{"reasoning_content":"t"}}]}'
        assert _openai_provider()._parse_stream_chunk(line) == "t"

    def test_done(self):
        assert _openai_provider()._parse_stream_chunk("data: [DONE]") is None

    def test_bad_json(self):
        assert _openai_provider()._parse_stream_chunk("data: not-json") is None

    def test_non_data_line(self):
        assert _openai_provider()._parse_stream_chunk(": comment") is None

    def test_anthropic_text(self):
        line = 'data: {"type":"content_block_delta","delta":{"text":"hi"}}'
        assert _anthropic_provider()._parse_stream_chunk(line) == "hi"

    def test_anthropic_message_delta_none(self):
        line = 'data: {"type":"message_delta","delta":{}}'
        assert _anthropic_provider()._parse_stream_chunk(line) is None


class TestProviderMethods:
    """各 Provider 的端点 / 请求体 / 模型列表."""

    def test_openai(self):
        p = _openai_provider()
        assert p._test_endpoint("gpt-4o") == "/v1/chat/completions"
        assert p._build_test_request("m")["model"] == "m"
        assert "gpt-4o" in p.list_models()

    def test_anthropic(self):
        p = _anthropic_provider()
        assert p._test_endpoint("claude-opus-4-8") == "/v1/messages"
        assert p._build_test_request("m")["max_tokens"] == 64
        assert p.list_models()  # 非空

    def test_deepseek(self):
        from agentfly.providers.deepseek import DeepSeekProvider
        p = DeepSeekProvider(ProviderConfig(
            name=ProviderType.DEEPSEEK, api_key="k",
            base_url="http://x", models=["d1"]))
        assert p._test_endpoint("d1") == "/v1/chat/completions"
        assert p._build_test_request("m")["messages"]
        assert p.list_models() == ["d1"]

    def test_custom_openai_only(self, monkeypatch):
        """只有 openai endpoint: 直接走 openai, 没有 fallback."""
        from agentfly.providers.custom import CustomProvider
        p = CustomProvider(ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k",
            base_url="http://x", models=["c1"]))
        _patch_client(monkeypatch, stream=_FakeStream(200, [
            'data: {"choices":[{"delta":{"content":"hi"}}]}',
            'data: {"choices":[{"delta":{"content":"!"}}]}',
            'data: [DONE]',
        ]))
        r = p.test_model("c1")
        assert r.status == "ok"

    def test_custom_anthropic_only(self, monkeypatch):
        """只有 anthropic endpoint: 直接走 anthropic."""
        from agentfly.providers.custom import CustomProvider
        p = CustomProvider(ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k",
            base_url="http://x", models=["c1"]))
        _patch_client(monkeypatch, stream=_FakeStream(200, [
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"text":"hi"}}',
            'event: message_delta',
            'data: {"type":"message_delta","delta":{}}',
        ]))
        r = p.test_model("c1")
        assert r.status == "ok"

    def test_custom_anthropic_400_fallback_to_openai(self, monkeypatch):
        """anthropic 400 自动回退 openai, 成功."""
        from agentfly.providers.custom import CustomProvider
        p = CustomProvider(ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k",
            base_url="http://x",
            models=["c1"]))
        # 第一次调用 (anthropic) 400, 第二次调用 (openai) 成功
        calls = []
        real_client = httpx.Client

        class TrackingClient:
            _call_no = 0

            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a, c=_call_no):
                return False

            def stream(self, method, url, **k):
                TrackingClient._call_no += 1
                n = TrackingClient._call_no
                calls.append(url)
                if n == 1:
                    return _FakeStream(400, [])
                return _FakeStream(200, [
                    'data: {"choices":[{"delta":{"content":"ok"}}]}',
                    'data: [DONE]',
                ])

        monkeypatch.setattr("agentfly.providers.base.httpx.Client", TrackingClient)
        r = p.test_model("c1")
        assert r.status == "ok"
        assert any("/v1/messages" in u for u in calls), "先试了 anthropic"
        assert any("/v1/chat/completions" in u for u in calls), "fallback 到 openai"
        assert r.latency_ms >= 0
        # 回退成功 → 缓存 api_type + 标记 dirty
        assert p._find_entry("c1").api_type == "openai"
        assert p._cache_dirty is True

    def test_custom_cached_api_type_no_fallback_not_dirty(self, monkeypatch):
        """已缓存 api_type → 只试一次, 不标记 dirty."""
        from agentfly.providers.custom import CustomProvider
        from agentfly.models.schema import ModelEntry
        p = CustomProvider(ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k", base_url="http://x",
            models=[ModelEntry(name="c1", api_type="openai")]))

        class Client:
            _calls = 0

            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def stream(self, method, url, **k):
                Client._calls += 1
                assert "/v1/chat/completions" in url  # 直接走缓存的 openai
                return _FakeStream(200, ['data: {"choices":[{"delta":{"content":"ok"}}]}', 'data: [DONE]'])

        monkeypatch.setattr("agentfly.providers.base.httpx.Client", Client)
        r = p.test_model("c1")
        assert r.status == "ok"
        assert Client._calls == 1  # 无探测 anthropic
        assert p._cache_dirty is False

    def test_custom_both_400_no_fallback(self, monkeypatch):
        """两个端点都 400 → error."""
        from agentfly.providers.custom import CustomProvider
        p = CustomProvider(ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k",
            base_url="http://x",
            models=["c1"]))

        class FailClient:
            _call_no = 0

            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def stream(self, *a, **k):
                FailClient._call_no += 1
                return _FakeStream(400, [])

        monkeypatch.setattr("agentfly.providers.base.httpx.Client", FailClient)
        r = p.test_model("c1")
        assert r.status == "error"

    def test_custom_anthropic_401_no_fallback(self, monkeypatch):
        """anthropic 401 → 直接返回, 不回退."""
        from agentfly.providers.custom import CustomProvider
        p = CustomProvider(ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k",
            base_url="http://x",
            models=["c1"]))

        class AuthClient:
            _calls = 0

            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def stream(self, *a, **k):
                AuthClient._calls += 1
                return _FakeStream(401, [])

        monkeypatch.setattr("agentfly.providers.base.httpx.Client", AuthClient)
        r = p.test_model("c1")
        assert r.status == "unauthorized"
        assert AuthClient._calls == 1  # 只试了一次, 没 fallback

    def test_custom_anthropic_timeout_no_fallback(self, monkeypatch):
        """anthropic 超时 → 直接返回, 不回退."""
        from agentfly.providers.custom import CustomProvider
        p = CustomProvider(ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k",
            base_url="http://x",
            models=["c1"]))

        class TimeoutClient:
            _calls = 0

            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def stream(self, *a, **k):
                TimeoutClient._calls += 1
                raise httpx.TimeoutException("timeout")

        monkeypatch.setattr("agentfly.providers.base.httpx.Client", TimeoutClient)
        r = p.test_model("c1")
        assert r.status == "timeout"
        assert TimeoutClient._calls == 1

    def test_custom_no_base_url_returns_error(self, monkeypatch):
        """没有 base_url → error."""
        from agentfly.providers.custom import CustomProvider
        p = CustomProvider(ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k",
            base_url="", models=["c1"]))
        r = p.test_model("c1")
        assert r.status == "error"

    def test_custom_parse_both_formats(self):
        """CustomProvider 的 parse 兼容两种 SSE."""
        from agentfly.providers.custom import CustomProvider
        p = CustomProvider(ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k", base_url="", models=[]))
        assert p._parse_stream_chunk('data: {"choices":[{"delta":{"content":"hi"}}]}') == "hi"
        assert p._parse_stream_chunk('data: {"type":"content_block_delta","delta":{"text":"hi"}}') == "hi"
        assert p._parse_stream_chunk('data: {"type":"message_delta","delta":{}}') is None
        assert p._parse_stream_chunk("data: [DONE]") is None
        assert p._parse_stream_chunk("not data") is None


class TestAnthropicParseEdge:
    """Anthropic SSE 解析的非 data / 坏 JSON 行."""

    def test_non_data_line(self):
        assert _anthropic_provider()._parse_stream_chunk("event: ping") is None

    def test_bad_json(self):
        assert _anthropic_provider()._parse_stream_chunk("data: nope") is None


class TestShouldFallback:
    """回退判断基于 status_code (int), 不是 error_message 子串."""

    @staticmethod
    def _r(status, code=0, msg=""):
        from agentfly.models.schema import TestResult
        return TestResult(provider="p", model="m", status=status, status_code=code, error_message=msg)

    def test_400_fallbacks(self):
        from agentfly.providers.base import _should_fallback
        assert _should_fallback(self._r("error", 400)) is True

    def test_404_405_501_fallback(self):
        from agentfly.providers.base import _should_fallback
        assert _should_fallback(self._r("error", 404)) is True
        assert _should_fallback(self._r("error", 405)) is True
        assert _should_fallback(self._r("error", 501)) is True

    def test_500_no_fallback_even_if_msg_has_400(self):
        """500 且消息含 '400' 子串 → 不回退 (旧子串匹配的 bug)."""
        from agentfly.providers.base import _should_fallback
        assert _should_fallback(self._r("error", 500, "reset after 400 bytes")) is False

    def test_non_error_status_no_fallback(self):
        from agentfly.providers.base import _should_fallback
        assert _should_fallback(self._r("unauthorized", 401)) is False
        assert _should_fallback(self._r("timeout", 0)) is False
        assert _should_fallback(self._r("ok", 200)) is False
