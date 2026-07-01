"""ConfigResolver 解析与错误路径测试."""

from __future__ import annotations

import pytest

from agentfly.core.resolver import ConfigResolver
from agentfly.models.schema import AgentConfig, ProviderConfig, UnifiedConfig
from agentfly.models.types import AgentType, ProviderType


def _cfg(api_key="sk-x", base_url="http://x"):
    return UnifiedConfig(
        providers={
            "p": ProviderConfig(
                name=ProviderType.OPENAI, api_key=api_key,
                base_url=base_url,
                models=["m1", "m2"], default_model="m1",
            )
        },
        agents={"codex": AgentConfig(name=AgentType.CODEX, provider="p", model="m2")},
    )


class TestResolve:
    def test_basic(self):
        r = ConfigResolver(_cfg()).resolve("codex", preferred_format="openai")
        assert r.effective_api_base == "http://x"
        assert r.agent.model == "m2"
        assert r.provider.name == ProviderType.OPENAI

    def test_agent_not_in_config_uses_default_model(self):
        cfg = UnifiedConfig(providers=_cfg().providers, agents={})
        r = ConfigResolver(cfg).resolve("codex", preferred_format="openai", provider_key="p")
        assert r.agent.model == "m1"  # provider.default_model
        assert r.agent.provider == "p"

    def test_missing_env_var_raises(self):
        cfg = _cfg(api_key="${LMSW_DEFINITELY_MISSING}")
        with pytest.raises(ValueError):
            ConfigResolver(cfg).resolve("codex", preferred_format="openai")

    def test_unsupported_format_raises(self):
        # provider 只暴露 openai，agent 要 anthropic
        with pytest.raises(ValueError):
            ConfigResolver(_cfg()).resolve("codex", preferred_format="anthropic")

    def test_missing_provider_raises(self):
        with pytest.raises(KeyError):
            ConfigResolver(_cfg()).resolve("codex", preferred_format="openai", provider_key="ghost")


class TestGetProvider:
    def test_resolves_env_ref(self, monkeypatch):
        monkeypatch.setenv("LMSW_KEY_X", "real-secret")
        pc = ConfigResolver(_cfg(api_key="${LMSW_KEY_X}")).get_provider("p")
        assert pc.api_key == "real-secret"

    def test_missing_raises(self):
        with pytest.raises(KeyError):
            ConfigResolver(_cfg()).get_provider("ghost")

    def test_accepts_provider_type(self):
        cfg = UnifiedConfig(providers={
            "openai": ProviderConfig(name=ProviderType.OPENAI, api_key="k",
                                     base_url="http://x", models=["m"])
        })
        pc = ConfigResolver(cfg).get_provider(ProviderType.OPENAI)
        assert pc.name == ProviderType.OPENAI


def test_no_agent_no_provider_key_falls_back_to_first():
    # agent 不在 config 且未给 provider_key → 取第一个 provider
    cfg = UnifiedConfig(providers=_cfg().providers, agents={})
    r = ConfigResolver(cfg).resolve("codex", preferred_format="openai")
    assert r.provider.name == ProviderType.OPENAI
    assert r.agent.provider == "p"
