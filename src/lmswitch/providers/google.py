"""Google Gemini Provider."""

from __future__ import annotations

from lmswitch.models.types import ProviderType
from lmswitch.providers.base import Provider

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]


class GoogleProvider(Provider):
    """Google Gemini API Provider."""

    name = ProviderType.GOOGLE
    display_name = "Google (Gemini)"

    def list_models(self) -> list[str]:
        return GEMINI_MODELS

    def _test_endpoint(self) -> str:
        return "/v1beta/models/gemini-2.0-flash:generateContent"

    def _build_test_request(self, model: str) -> dict:
        return {
            "contents": [{"parts": [{"text": "hi"}]}],
        }
