"""客制化 Provider — 支持任意 OpenAI/Anthropic 兼容 API."""

from __future__ import annotations

import json

from agentfly.models.types import ProviderType
from agentfly.providers.base import Provider


class CustomProvider(Provider):
    """客制化 Provider — 支持任意 OpenAI/Anthropic 兼容 API.

    同时配了 openai + anthropic 端点时, 每个模型先试 anthropic 路径
    (/v1/messages), HTTP 400/404 自动回退到 openai 路径 (/v1/chat/completions).

    跑通的接口自动缓存到模型的 api_type 字段, 后续测试直接走缓存.
    """

    name = ProviderType.CUSTOM
    display_name = "Custom"

    def list_models(self) -> list[str]:
        return self.config.model_names

    def _build_test_request(self, model: str) -> dict:
        return {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }

    def _get_api_type(self, model: str) -> str:
        """查 models 列表中该模型的 api_type 缓存."""
        for me in self.config.models:
            if me.name == model and me.api_type:
                return me.api_type
        return ""

    def _set_api_type(self, model: str, api_type: str) -> None:
        """写入 models 列表中该模型的 api_type (原地修改)."""
        for me in self.config.models:
            if me.name == model:
                me.api_type = api_type
                return
        # 模型不在列表中也添加
        from agentfly.models.schema import ModelEntry
        self.config.models.append(ModelEntry(name=model, api_type=api_type))

    def _test_candidates(
        self, model: str, api_key: str, base: str,
    ) -> list[tuple[str, dict, str]]:
        """返回 (url, headers, endpoint_key) 列表.

        优先查模型的 api_type 缓存; 无缓存时 anthropic 优先, openai 回退.
        """
        base = base or self.config.base_url
        if not base:
            return []

        # ── 有 api_type 缓存 → 只试这个接口 ──
        proto = self._get_api_type(model)
        if proto:
            url = self._url_for(proto, base)
            return [(url, self._headers_for(proto, api_key), proto)]

        # ── 无缓存 → anthropic 优先, openai 回退 ──
        return [
            (self._url_for("anthropic", base), self._anthropic_headers(api_key), "anthropic"),
            (self._url_for("openai", base), self._openai_headers(api_key), "openai"),
        ]

    # ── 内部工具 ──

    @staticmethod
    def _url_for(ep_key: str, base: str) -> str:
        path = "/v1/messages" if ep_key == "anthropic" else "/v1/chat/completions"
        return f"{base.rstrip('/')}{path}"

    @staticmethod
    def _headers_for(ep_key: str, api_key: str) -> dict[str, str]:
        if ep_key == "anthropic":
            return {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _openai_headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _anthropic_headers(api_key: str) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _on_test_ok(self, model: str, ep_key: str, idx: int) -> None:
        """回退成功 → 缓存 api_type."""
        if idx > 0 and ep_key:
            self._set_api_type(model, ep_key)

    def _parse_stream_chunk(self, line: str) -> str | None:
        """兼容 OpenAI (choices) 与 Anthropic (content_block_delta) SSE."""
        if not line.startswith("data: "):
            return None
        data_str = line[6:]
        if data_str == "[DONE]":
            return None
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            return None

        # OpenAI
        choices = chunk.get("choices")
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content") or delta.get("reasoning_content") or ""

        # Anthropic
        if chunk.get("type") == "content_block_delta":
            return chunk.get("delta", {}).get("text", "")
        return None