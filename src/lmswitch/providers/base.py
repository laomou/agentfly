"""Provider 抽象基类."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod

import httpx

from lmswitch.models.schema import ProviderConfig, TestResult
from lmswitch.models.types import ProviderType


class Provider(ABC):
    """服务提供商的抽象基类.

    每个 Provider 负责:
    - 列出可用模型
    - 测试模型连通性和延迟 (stream 模式测 TTFT + 吞吐)
    """

    name: ProviderType
    display_name: str = ""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def list_models(self) -> list[str]:
        """返回该 Provider 的已知模型列表."""
        ...

    @abstractmethod
    def _test_endpoint(self) -> str:
        """返回用于测试的 API endpoint."""
        ...

    @abstractmethod
    def _build_test_request(self, model: str):
        """构建测试请求 (需包含 stream: True 以测 TTFT)."""
        ...

    def _parse_stream_chunk(self, line: str) -> str | None:
        """解析 SSE 流中的一行，返回 delta content 或 None."""
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                return None
            try:
                chunk = json.loads(data_str)
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    return delta.get("content", "")
            except json.JSONDecodeError:
                pass
        return None

    def test_model(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> TestResult:
        """测试模型 (stream 模式): TTFT + 吞吐 + 总延迟."""
        key = api_key or self.config.api_key
        base = api_base or next(iter(self.config.endpoints.values()), "")

        url = f"{base.rstrip('/')}{self._test_endpoint()}"
        body = self._build_test_request(model)
        body["stream"] = True

        try:
            t_start = time.monotonic()
            ttft_ms = 0.0
            token_count = 0

            with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
                with client.stream(
                    "POST", url,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                ) as resp:
                    if resp.status_code in (401, 403):
                        return TestResult(
                            provider=self.name, model=model,
                            status="unauthorized",
                            error_message=f"HTTP {resp.status_code}",
                        )

                    if resp.status_code != 200:
                        return TestResult(
                            provider=self.name, model=model,
                            status="error",
                            error_message=f"HTTP {resp.status_code}",
                        )

                    first_token = True
                    for line in resp.iter_lines():
                        content = self._parse_stream_chunk(line)
                        if content:
                            if first_token:
                                ttft_ms = (time.monotonic() - t_start) * 1000
                                first_token = False
                            token_count += len(content) // 4 or 1  # rough estimate

            total_ms = (time.monotonic() - t_start) * 1000
            total_s = total_ms / 1000
            tps = token_count / total_s if total_s > 0 else 0

            return TestResult(
                provider=self.name,
                model=model,
                status="ok",
                latency_ms=round(total_ms, 1),
                ttft_ms=round(ttft_ms, 1),
                tokens_per_sec=round(tps, 1),
            )

        except httpx.TimeoutException:
            return TestResult(
                provider=self.name, model=model,
                status="timeout",
                error_message="请求超时 (30s)",
            )
        except httpx.ConnectError:
            return TestResult(
                provider=self.name, model=model,
                status="error",
                error_message="无法连接",
            )
        except Exception as e:
            return TestResult(
                provider=self.name, model=model,
                status="error",
                error_message=str(e)[:200],
            )
