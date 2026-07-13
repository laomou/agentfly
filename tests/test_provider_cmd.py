"""Provider 命令层测试."""

from __future__ import annotations

import pytest
from agentfly.cli.provider_cmd import _auto_name, _mask_key, _is_plaintext_key
from agentfly.models.schema import ProviderConfig, TestResult
from agentfly.models.types import ProviderType
from agentfly.providers.base import Provider


class TestAutoName:
    """_auto_name URL → 名称 提取."""

    def test_http_url(self):
        assert _auto_name("http://192.168.1.100:3000") == "192.168.1.100"

    def test_https_url(self):
        assert _auto_name("https://api.openai.com") == "api.openai.com"

    def test_url_with_path(self):
        assert _auto_name("http://host:8000/v1") == "host"

    def test_no_protocol(self):
        assert _auto_name("192.168.1.100:3000") == "192.168.1.100"

    def test_no_protocol_no_port(self):
        assert _auto_name("api.deepseek.com") == "api.deepseek.com"

    def test_localhost_default(self):
        assert _auto_name("http://localhost:8000") == "localhost:8000"

    def test_localhost_custom_port(self):
        assert _auto_name("http://localhost:3000") == "localhost:3000"

    def test_localhost_no_port(self):
        assert _auto_name("http://localhost") == "localhost:8000"

    def test_invalid(self):
        assert _auto_name("") == "custom"


class TestMaskKey:
    """_mask_key API Key 脱敏."""

    def test_env_var_unchanged(self):
        assert _mask_key("${MY_KEY}") == "${MY_KEY}"

    def test_plaintext_long(self):
        key = "sk-abcdefgh12345678xyz"
        masked = _mask_key(key)
        assert masked == key[:5] + "..." + key[-4:]

    def test_plaintext_short(self):
        masked = _mask_key("ab12xy78")
        assert masked == "ab***78"


class TestIsPlaintextKey:
    """_is_plaintext_key 判定."""

    def test_env_var(self):
        assert _is_plaintext_key("${MY_KEY}") is False

    def test_plaintext(self):
        assert _is_plaintext_key("sk-abc123") is True

    def test_dollar_not_env(self):
        assert _is_plaintext_key("$NOT_BRACED") is True


class FakeProvider(Provider):
    """用于测试 test_model 的假 Provider."""

    name = ProviderType.CUSTOM
    display_name = "Fake"

    def _build_test_request(self, model: str) -> dict:
        return {"model": model, "messages": [{"role": "user", "content": "hi"}]}

    def _default_model(self) -> str:
        return "test-model"

    def list_models(self) -> list[str]:
        return ["test-model"]


class TestModelProviderKey:
    """test_model 传递 provider_key."""

    def test_uses_provided_key(self):
        config = ProviderConfig(
            type=ProviderType.CUSTOM,
            api_key="sk-test",
            endpoints={"openai": "http://localhost:9999"},
            models=["test-model"],
            default_model="test-model",
        )
        provider = FakeProvider(config)
        result = provider.test_model("test-model", provider_key="my-proxy")
        assert result.provider == "my-proxy"

    def test_fallback_when_empty(self):
        config = ProviderConfig(
            type=ProviderType.DEEPSEEK,
            api_key="sk-test",
            endpoints={"openai": "http://localhost:9999"},
            models=["test-model"],
            default_model="test-model",
        )
        provider = FakeProvider(config)
        result = provider.test_model("test-model")
        assert result.provider == "deepseek"
