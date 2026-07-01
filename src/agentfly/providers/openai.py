"""OpenAI Provider."""

from __future__ import annotations

from agentfly.models.types import ProviderType
from agentfly.providers.base import Provider

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "o3",
    "o4-mini",
    "codex",
]


class OpenAIProvider(Provider):
    """OpenAI API Provider."""

    name = ProviderType.OPENAI
    display_name = "OpenAI"

    def list_models(self) -> list[str]:
        return OPENAI_MODELS
