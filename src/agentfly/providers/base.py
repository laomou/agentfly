"""Provider 抽象基类."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from agentfly.models.schema import ProviderConfig, TestResult
from agentfly.models.types import ProviderType


class Provider(ABC):
    """服务提供商的抽象基类.

    每个 Provider 负责:
    - 列出可用模型
    - 测试模型连通性和延迟 (stream 模式测 TTFT + 吞吐)
    - 为不同 Agent 提供 Provider 特定的环境变量
    """

    name: ProviderType
    display_name: str = ""

    def __init__(self, config: ProviderConfig):
        self.config = config

    _ENV_CACHE: dict[str, dict] = {}

    def env_for(self, agent_name: str) -> dict[str, str]:
        """返回该 Provider 给指定 Agent 的补充环境变量.

        读取顺序:
        1. ~/.config/agentfly/env/{name}.json (provider add/reload 缓存)
        2. 包内 providers/{name}.json (默认)

        Args:
            agent_name: Agent 名称 (如 "claude").

        Returns:
            环境变量字典，无配置时返回 {}.
        """
        name = self.name.value
        key = f"{name}:{agent_name}"
        if key in self._ENV_CACHE:
            return self._ENV_CACHE[key]

        # 用户缓存 → 包默认
        candidates = [
            Path.home() / ".config" / "agentfly" / "env" / f"{name}.json",
            Path(__file__).parent / f"{name}.json",
        ]
        for env_file in candidates:
            if env_file.exists():
                try:
                    data = json.loads(env_file.read_text())
                    result = data.get(agent_name, {})
                    self._ENV_CACHE[key] = result
                    return result
                except (json.JSONDecodeError, OSError):
                    pass

        self._ENV_CACHE[key] = {}
        return {}

    @abstractmethod
    def list_models(self) -> list[str]:
        """返回该 Provider 的已知模型列表."""
        ...

    def _test_endpoint(self, model: str) -> str:
        """返回默认识别的 API endpoint."""
        raise NotImplementedError  # pragma: no cover

    def _request_headers(self, model: str, api_key: str) -> dict[str, str]:
        """默认 Bearer 鉴权头."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @abstractmethod
    def _build_test_request(self, model: str):
        """构建测试请求 (需包含 stream: True 以测 TTFT)."""
        ...

    def _parse_stream_chunk(self, line: str) -> str | None:
        """解析 SSE 流中的一行 (OpenAI 格式), 返回 delta content 或 None.

        兼容 reasoning 模型（先出 reasoning_content 再出 content）.
        """
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                return None
            try:
                chunk = json.loads(data_str)
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    return delta.get("content") or delta.get("reasoning_content") or ""
            except json.JSONDecodeError:
                pass
        return None

    def _test_candidates(
        self, model: str, api_key: str, base: str,
    ) -> list[tuple[str, dict, str]]:
        """返回 (url, headers, endpoint_key) 候选列表.

        endpoint_key 用于缓存 (如 "openai" / "anthropic"), 单端点 Provider 传 "".
        """
        url = f"{base.rstrip('/')}{self._test_endpoint(model)}"
        headers = self._request_headers(model, api_key)
        return [(url, headers, "")]

    def test_model(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        provider_key: str = "",
    ) -> TestResult:
        """测试模型 (stream 模式): TTFT + 吞吐 + 总延迟.

        400/404 自动回退到下一个候选端点; 401/403/超时/连接不上直接返回.
        成功时若涉及回退则缓存 endpoint_key.
        """
        pkey = provider_key or self.config.name.value
        key = api_key or self.config.api_key
        base = api_base or self.config.base_url
        body = self._build_test_request(model)
        body["stream"] = True

        candidates = self._test_candidates(model, key, base)
        last_error: str = ""

        for idx, (url, headers, ep_key) in enumerate(candidates):
            result = self._do_test(pkey, model, url, body, headers)
            if result.status == "ok":
                self._on_test_ok(model, ep_key, idx)
                return result
            # 非 error → 401/403/超时/连接不上, 不回退
            if result.status != "error":
                return result
            # 400/404 才回退; 其他 error 码 (500 等) 也直接返回
            if "400" not in result.error_message and "404" not in result.error_message:
                return result
            last_error = result.error_message or last_error

        return TestResult(
            provider=pkey, model=model,
            status="error",
            error_message=last_error or "所有接口均失败",
        )

    # ── 单次测试 ──

    def _on_test_ok(self, model: str, ep_key: str, idx: int) -> None:
        """成功回调 — 子类可覆写以缓存 endpoint_key."""
        return

    def _do_test(
        self, pkey: str, model: str, url: str, body: dict, headers: dict,
    ) -> TestResult:
        """执行一次流式测试并返回结果."""
        try:
            t_start = time.monotonic()
            ttft_ms = 0.0
            token_count = 0

            with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
                with client.stream(
                    "POST", url, json=body, headers=headers,
                ) as resp:
                    if resp.status_code in (401, 403):
                        return TestResult(
                            provider=pkey, model=model,
                            status="unauthorized",
                            error_message=f"HTTP {resp.status_code}",
                        )
                    if resp.status_code != 200:
                        return TestResult(
                            provider=pkey, model=model,
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
                provider=pkey, model=model,
                status="ok",
                latency_ms=round(total_ms, 1),
                ttft_ms=round(ttft_ms, 1),
                tokens_per_sec=round(tps, 1),
            )

        except httpx.TimeoutException:
            return TestResult(
                provider=pkey, model=model,
                status="timeout", error_message="请求超时 (30s)",
            )
        except httpx.ConnectError:
            return TestResult(
                provider=pkey, model=model,
                status="error", error_message="无法连接",
            )
        except Exception as e:
            return TestResult(
                provider=pkey, model=model,
                status="error", error_message=str(e)[:200],
            )