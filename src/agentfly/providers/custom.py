"""客制化 Provider — 支持任意 OpenAI/Anthropic 兼容 API."""

from __future__ import annotations

import json
import threading

from agentfly.models.schema import ModelEntry, ProviderConfig
from agentfly.models.types import ProviderType
from agentfly.providers.base import Provider

# api_type → (path, header_builder)
_PATHS = {
    "anthropic": "/v1/messages",
    "openai": "/v1/chat/completions",
}


def _headers(api_type: str, api_key: str) -> dict[str, str]:
    if api_type == "anthropic":
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


class CustomProvider(Provider):
    """客制化 Provider — 支持任意 OpenAI/Anthropic 兼容 API.

    首次测试对每个模型: 先试 anthropic (/v1/messages), HTTP 400/404 自动
    回退 openai (/v1/chat/completions). 跑通的接口写入 ModelEntry.api_type,
    后续测试直接走缓存, 零额外探测.
    """

    name = ProviderType.CUSTOM
    display_name = "Custom"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._cache_dirty = False        # 本次是否改动了 api_type 缓存
        self._cache_lock = threading.Lock()  # 保护并发测试对 config.models 的写

    def list_models(self) -> list[str]:
        return self.config.model_names

    def _build_test_request(self, model: str) -> dict:
        return {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }

    # ── api_type 缓存 ──

    def _find_entry(self, model: str) -> ModelEntry | None:
        for me in self.config.models:
            if me.name == model:
                return me
        return None

    def _on_test_ok(self, model: str, ep_key: str, idx: int) -> None:
        """回退成功 (idx>0) → 缓存 api_type; 标记 dirty 供上层决定是否写盘."""
        if idx == 0 or not ep_key:
            return
        with self._cache_lock:
            me = self._find_entry(model)
            if me is not None:
                if me.api_type == ep_key:
                    return
                me.api_type = ep_key
            else:
                self.config.models.append(ModelEntry(name=model, api_type=ep_key))
            self._cache_dirty = True

    # ── 候选端点 ──

    def _candidate(self, api_type: str, base: str, api_key: str) -> tuple[str, dict, str]:
        return (
            f"{base.rstrip('/')}{_PATHS[api_type]}",
            _headers(api_type, api_key),
            api_type,
        )

    def _test_candidates(
        self, model: str, api_key: str, base: str,
    ) -> list[tuple[str, dict, str]]:
        """有 api_type 缓存 → 单一候选; 无缓存 → anthropic 优先, openai 回退."""
        base = base or self.config.base_url
        if not base:
            return []

        me = self._find_entry(model)
        if me and me.api_type:
            return [self._candidate(me.api_type, base, api_key)]

        return [
            self._candidate("anthropic", base, api_key),
            self._candidate("openai", base, api_key),
        ]

    # ── SSE 解析 ──

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

        choices = chunk.get("choices")
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content") or delta.get("reasoning_content") or ""

        if chunk.get("type") == "content_block_delta":
            return chunk.get("delta", {}).get("text", "")
        return None
