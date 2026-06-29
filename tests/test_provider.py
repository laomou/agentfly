"""Provider 管理层测试."""

from __future__ import annotations

import pytest
from lmswitch.models.schema import ProviderConfig
from lmswitch.models.types import ProviderType
from lmswitch.providers.manager import ProviderManager


class TestProviderManager:
    """ProviderManager CRUD 测试."""

    def test_add_and_list(self, unified_config):
        mgr = ProviderManager(unified_config)
        providers = mgr.list()
        assert len(providers) == 1
        assert providers[0].name == ProviderType.ANTHROPIC

    def test_add_new(self, unified_config):
        mgr = ProviderManager(unified_config)
        openai_config = ProviderConfig(
            name=ProviderType.OPENAI,
            api_key="${OPENAI_API_KEY}",
            api_base="https://api.openai.com",
            models=["gpt-4o"],
            default_model="gpt-4o",
        )
        mgr.add(openai_config)
        assert len(mgr.list()) == 2

    def test_remove(self, unified_config):
        mgr = ProviderManager(unified_config)
        mgr.remove("anthropic")
        assert len(mgr.list()) == 0

    def test_remove_nonexistent(self, unified_config):
        mgr = ProviderManager(unified_config)
        with pytest.raises(KeyError):
            mgr.remove("nonexistent")

    def test_get(self, unified_config):
        mgr = ProviderManager(unified_config)
        provider = mgr.get("anthropic")
        assert provider is not None
        assert provider.name == ProviderType.ANTHROPIC
