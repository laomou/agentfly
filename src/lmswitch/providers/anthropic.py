"""Anthropic (Claude) Provider."""

from __future__ import annotations

import json

from lmswitch.models.types import ProviderType
from lmswitch.providers.base import Provider

ANTHROPIC_MODELS = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-fable-5",
]


class AnthropicProvider(Provider):
    """Anthropic Claude API Provider."""

    name = ProviderType.ANTHROPIC
    display_name = "Anthropic (Claude)"

    def list_models(self) -> list[str]:
        return ANTHROPIC_MODELS

    def _test_endpoint(self) -> str:
        return "/v1/messages"

    def _build_test_request(self, model: str) -> dict:
        return {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }

    def _parse_stream_chunk(self, line: str) -> str | None:
        """Anthropic SSE: data: {"type":"content_block_delta","delta":{"text":"..."}}"""
        if not line.startswith("data: "):
            return None
        try:
            event = json.loads(line[6:])
            if event.get("type") == "content_block_delta":
                return event.get("delta", {}).get("text", "")
            if event.get("type") == "message_delta":
                return None
        except json.JSONDecodeError:
            pass
        return None
