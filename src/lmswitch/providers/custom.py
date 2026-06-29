"""客制化 Provider — 支持任意 OpenAI/Anthropic 兼容 API."""

from __future__ import annotations

from lmswitch.models.types import ProviderType
from lmswitch.providers.base import Provider


class CustomProvider(Provider):
    """客制化 Provider.

    根据 endpoints 自动适配测试端点:
    - openai endpoint    → /v1/chat/completions
    - anthropic endpoint → /v1/messages
    """

    name = ProviderType.CUSTOM
    display_name = "Custom"

    def list_models(self) -> list[str]:
        return self.config.models

    def _test_endpoint(self) -> str:
        # 尝试用第一个可用格式
        if "openai" in self.config.endpoints:
            return "/v1/chat/completions"
        return "/v1/messages"

    def _build_test_request(self, model: str) -> dict:
        return {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
