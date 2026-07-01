"""Provider 抽象基类."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from agentfly.models.schema import ProviderConfig, TestResult
from agentfly.models.types import ProviderType

_DEFAULT_TIMEOUT_S = 30.0

# api_type → API 路径
_PATHS = {
    "anthropic": "/v1/messages",
    "openai": "/v1/chat/completions",
}
# 无缓存时的探测优先级
_PROBE_ORDER = ("anthropic", "openai")


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
        self._env_cache: dict[str, dict] = {}  # {agent_name: env}, 实例级避免跨实例泄漏

    # ── env ──

    def env_for(self, agent_name: str) -> dict[str, str]:
        """返回该 Provider 给指定 Agent 的补充环境变量.

        读取顺序: ~/.config/agentfly/env/{name}.json → 包内 providers/{name}.json.
        """
        if agent_name in self._env_cache:
            return self._env_cache[agent_name]

        name = self.name.value
        for env_file in (
            Path.home() / ".config" / "agentfly" / "env" / f"{name}.json",
            Path(__file__).parent / f"{name}.json",
        ):
            if not env_file.exists():
                continue
            try:
                data = json.loads(env_file.read_text())
                result = data.get(agent_name, {})
                self._env_cache[agent_name] = result
                return result
            except (json.JSONDecodeError, OSError):
                pass

        self._env_cache[agent_name] = {}
        return {}

    # ── 子类契约 ──

    @abstractmethod
    def list_models(self) -> list[str]:
        """返回该 Provider 的已知模型列表."""
        ...

    @abstractmethod
    def _build_test_request(self, model: str):
        """构建测试请求 (stream=True 由 test_model 自动添加)."""
        ...

    def _parse_stream_chunk(self, line: str) -> str | None:
        """解析 SSE 一行 (OpenAI 格式). 兼容 reasoning 模型."""
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
        if not choices:
            return None
        delta = choices[0].get("delta", {})
        return delta.get("content") or delta.get("reasoning_content") or ""

    # ── 候选端点 (基于 config.endpoints) ──

    def _api_types(self, model: str) -> list[str]:
        """待探测的接口顺序. 有 api_type 缓存则按缓存, 否则 anthropic 优先.

        只保留 config.endpoints 里实际配置了的接口.
        """
        eps = self.config.endpoints
        cached = [t for t in self.config.models.get(model, "").split(",")
                  if t.strip() in eps]
        if cached:
            return cached
        return [t for t in _PROBE_ORDER if t in eps]

    def _test_candidates(self, model: str, api_key: str) -> list[tuple[str, dict, str]]:
        """返回 (url, headers, api_type) 候选列表, 从 endpoints 逐接口构造."""
        out: list[tuple[str, dict, str]] = []
        for t in self._api_types(model):
            base = self.config.endpoints.get(t)
            if base:
                out.append((f"{base.rstrip('/')}{_PATHS[t]}", _headers(t, api_key), t))
        return out

    def _on_results(self, model: str, api_types: list[str]) -> None:
        """测试结束回调, api_types = 跑通的接口 (已按速度排序). 子类覆写以缓存."""

    # ── 主流程 ──

    def test_model(
        self,
        model: str,
        api_key: str | None = None,
        provider_key: str = "",
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> TestResult:
        """流式测试模型: 探测所有候选接口, 取最快的报告.

        - ok: 收集; 接口不匹配 (400/404/…): 试下一个候选; 其他错误: 立即返回.
        - 多个接口都通时 api_type 记成 "anthropic,openai" (速度快的在前),
          Total/TTFT/TPS 显示最快接口的指标.
        """
        pkey = provider_key or self.config.name.value
        key = api_key or self.config.api_key
        body = {**self._build_test_request(model), "stream": True}

        candidates = self._test_candidates(model, key)
        if not candidates:
            return TestResult(provider=pkey, model=model, status="error",
                              error_message="未配置 endpoints")

        successes: list[TestResult] = []  # 每个 result.api_type 已置为单接口 key
        for url, headers, ep_key in candidates:
            result = self._do_test(pkey, model, url, body, headers, timeout)
            if result.status == "ok":
                result.api_type = ep_key
                successes.append(result)
            elif not _should_fallback(result):
                return result  # 401/403/超时/连接不上 → 直接返回

        if not successes:
            return TestResult(provider=pkey, model=model, status="error",
                              error_message="所有接口均失败")

        successes.sort(key=lambda r: r.latency_ms)  # 最快的在前
        ordered = [r.api_type for r in successes if r.api_type]
        best = successes[0]
        best.api_type = ",".join(ordered)
        self._on_results(model, ordered)
        return best

    def _do_test(
        self, pkey: str, model: str, url: str, body: dict, headers: dict,
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> TestResult:
        """执行一次流式测试."""
        def result(**kw) -> TestResult:
            return TestResult(provider=pkey, model=model, **kw)

        try:
            t_start = time.monotonic()
            ttft_ms = 0.0
            token_count = 0

            with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
                with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code in (401, 403):
                        return result(
                            status="unauthorized",
                            status_code=resp.status_code,
                            error_message=f"HTTP {resp.status_code}",
                        )
                    if resp.status_code != 200:
                        return result(
                            status="error",
                            status_code=resp.status_code,
                            error_message=f"HTTP {resp.status_code}",
                        )

                    first_token = True
                    for line in resp.iter_lines():
                        content = self._parse_stream_chunk(line)
                        if not content:
                            continue
                        if first_token:
                            ttft_ms = (time.monotonic() - t_start) * 1000
                            first_token = False
                        token_count += len(content) // 4 or 1

            total_ms = (time.monotonic() - t_start) * 1000
            tps = token_count * 1000 / total_ms if total_ms > 0 else 0

            return result(
                status="ok",
                latency_ms=round(total_ms, 1),
                ttft_ms=round(ttft_ms, 1),
                tokens_per_sec=round(tps, 1),
            )

        except httpx.TimeoutException:
            return result(status="timeout", error_message=f"请求超时 ({int(timeout)}s)")
        except httpx.ConnectError:
            return result(status="error", error_message="无法连接")
        except Exception as e:
            return result(status="error", error_message=str(e)[:200])


# 这些状态码代表"该接口不支持此模型/请求" → 回退下一个候选端点
_FALLBACK_STATUS = frozenset({400, 404, 405, 415, 501})


def _should_fallback(result: TestResult) -> bool:
    """仅在明确的"接口不匹配"状态码上回退下一个候选."""
    return result.status == "error" and result.status_code in _FALLBACK_STATUS
