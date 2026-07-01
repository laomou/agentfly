"""客制化 Provider — 支持任意 OpenAI/Anthropic 兼容 API."""

from __future__ import annotations

import json
import threading

from agentfly.models.schema import ProviderConfig
from agentfly.models.types import ProviderType
from agentfly.providers.base import Provider


class CustomProvider(Provider):
    """客制化 Provider — 支持任意 OpenAI/Anthropic 兼容 API.

    候选/探测/取最快的骨架在 base.test_model; 这里只负责把跑通的接口写入
    api_type 缓存 (如 "anthropic,openai", 速度快的在前) 并解析两种 SSE.
    """

    name = ProviderType.CUSTOM
    display_name = "Custom"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._cache_dirty = False             # 本次是否改动了 api_type 缓存
        self._cache_lock = threading.Lock()   # 保护并发测试对 config.models 的写

    def list_models(self) -> list[str]:
        return self.config.model_names

    def _build_test_request(self, model: str) -> dict:
        return {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }

    def _on_results(self, model: str, api_types: list[str]) -> None:
        """缓存跑通的接口 (base 测试结束时回调)."""
        val = ",".join(api_types)
        with self._cache_lock:
            if self.config.models.get(model) == val:
                return
            self.config.models[model] = val
            self._cache_dirty = True

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

        # Anthropic: text_delta 或 thinking_delta (推理模型)
        if chunk.get("type") == "content_block_delta":
            delta = chunk.get("delta", {})
            return delta.get("text") or delta.get("thinking") or ""
        return None
