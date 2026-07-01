"""provider_cmd 网络探测 helper 测试 (mock httpx)."""

from __future__ import annotations

import httpx

from agentfly.cli import provider_cmd as pc


class _Resp:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json


def _patch_client(monkeypatch, *, post=None, get=None, post_exc=None):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **k):
            if post_exc is not None:
                raise post_exc
            return post

        def get(self, *a, **k):
            return get

    monkeypatch.setattr(pc.httpx, "Client", FakeClient)


class TestProbeEndpoint:
    def test_200_true(self, monkeypatch):
        _patch_client(monkeypatch, post=_Resp(200))
        assert pc._probe_endpoint("http://x", "k", "openai") is True

    def test_404_false(self, monkeypatch):
        _patch_client(monkeypatch, post=_Resp(404))
        assert pc._probe_endpoint("http://x", "k", "openai") is False

    def test_connect_error_false(self, monkeypatch):
        _patch_client(monkeypatch, post_exc=httpx.ConnectError("x"))
        assert pc._probe_endpoint("http://x", "k", "openai") is False


class TestDetectProtocols:
    def test_filters_by_probe(self, monkeypatch):
        monkeypatch.setattr(pc, "_probe_endpoint", lambda base, key, fmt: fmt == "openai")
        assert pc._detect_protocols("http://x/", "k") == {"openai"}


class TestFetchModels:
    def test_filters_and_sorts(self, monkeypatch):
        data = {"data": [{"id": "gpt-4o"}, {"id": "text-embedding-3"}, {"id": "aaa"}]}
        _patch_client(monkeypatch, get=_Resp(200, data))
        models = pc._fetch_models("http://x", "k")
        assert "text-embedding-3" not in models  # embedding 被过滤
        assert models == sorted(models)
        assert "gpt-4o" in models and "aaa" in models
