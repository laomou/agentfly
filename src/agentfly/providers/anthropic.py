"""Anthropic (Claude) Provider."""

from __future__ import annotations

from agentfly.models.types import ProviderType
from agentfly.providers.base import Provider

ANTHROPIC_MODELS = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-fable-5",
]

# Anthropic 官方端点 — 这些 URL 下 Claude Code 自身默认即指向此处,
# agentfly 无需覆盖 BASE_URL 或关闭 attribution header
ANTHROPIC_OFFICIAL_BASES = frozenset(
    {"https://api.anthropic.com", "http://api.anthropic.com"}
)


def is_official_anthropic_base(url: str) -> bool:
    """判断 api_base 是否 Anthropic 官方端点 (归一化后比对)."""
    return url.rstrip("/").lower() in ANTHROPIC_OFFICIAL_BASES


class AnthropicProvider(Provider):
    """Anthropic Claude API Provider."""

    name = ProviderType.ANTHROPIC
    display_name = "Anthropic (Claude)"

    def list_models(self) -> list[str]:
        return ANTHROPIC_MODELS
