"""DeepSeek Provider."""

from __future__ import annotations

from lmswitch.models.types import ProviderType
from lmswitch.providers.base import Provider


class DeepSeekProvider(Provider):
    """DeepSeek API Provider.

    DeepSeek tiers: pro(强) + flash(快)，只有 2 个 tier.
    用于 Claude 时: opus=sonnet=pro, haiku=flash.
    """

    name = ProviderType.DEEPSEEK
    display_name = "DeepSeek"

    def list_models(self) -> list[str]:
        return self.config.models

    def _test_endpoint(self) -> str:
        return "/v1/chat/completions"

    def _build_test_request(self, model: str) -> dict:
        return {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
