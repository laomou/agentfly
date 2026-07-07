"""[agentfly test] 测试模型可用性和速度 (stream 模式: TTFT + 吞吐)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatchcase
from typing import Any

import click

from agentfly.cli._completion import complete_providers, model_completer
from agentfly.core.config import ensure_config_exists, save_config
from agentfly.core.resolver import ConfigResolver
from agentfly.models.schema import ProviderConfig, TestResult
from agentfly.providers.base import _DEFAULT_TIMEOUT_S
from agentfly.providers.registry import get_provider

_STATUS_ICONS = {"ok": "✅", "timeout": "⏳", "error": "❌", "unauthorized": "❌"}
_COL_STATUS, _COL_LATENCY, _COL_TTFT, _COL_TPS = 14, 8, 8, 8
# API 列: api_type 取值 openai/anthropic/responses, 最长组合 "anthropic,openai,responses" = 24.
# 流式输出时还拿不到全部行, 用该上限保证不溢出; 非流式则按实际内容动态取宽.
_COL_API_MAX = len("anthropic,openai,responses")
_DEFAULT_PARALLEL = 4


# ── helpers ──

def _resolve(config, name: str) -> tuple[ProviderConfig, Any]:
    """解析 Provider 名称 → (ProviderConfig, Provider 实例). pc 已解析 env."""
    try:
        pc = ConfigResolver(config).get_provider(name)
    except KeyError:
        raise click.ClickException(f"Provider 未配置: {name}")
    p = get_provider(pc)
    if p is None:
        raise click.ClickException(f"不支持的 Provider: {pc.name.value}")
    return pc, p


def _clear_api_type(pc: ProviderConfig) -> None:
    """--refresh: 清空 api_type 缓存, 强制重新探测."""
    for name in pc.models:
        pc.models[name] = ""


def _expand_models(pc: ProviderConfig, p, pattern: str) -> list[str]:
    """展开模型名: 含 * / ? 时按 fnmatch 在候选清单里筛, 否则原样返回单元素.

    候选来自 pc.model_names (配置的模型), 兜底 p.list_models(). 大小写敏感.
    """
    if "*" not in pattern and "?" not in pattern:
        return [pattern]
    candidates = [m for m in (pc.model_names or p.list_models()) if m]
    return [m for m in candidates if fnmatchcase(m, pattern)]


def _save_cache(config, key: str, pc: ProviderConfig, p) -> None:
    """探测到的 api_type 从 (env 解析后的) 副本同步回原始 config, 再写盘.

    原始 config 保留 ${VAR} 形式的 api_key, 不会被解析后的明文覆盖.
    """
    if not getattr(p, "_cache_dirty", False):
        return
    orig = config.providers.get(key)
    if orig is not None:
        orig.models = dict(pc.models)
    save_config(config)


# ── 输出 ──

def _icon(status: str) -> str:
    return _STATUS_ICONS.get(status, "❓")


def _pad(val: float) -> str:
    """格式化时间数值: ms/s 或 `-`."""
    if val <= 0:
        return "-"
    if val < 1000:
        return f"{val:.0f}ms"
    return f"{val / 1000:.1f}s"


def _tps(val: float) -> str:
    """格式化吞吐量 (tokens/s) 或 `-`."""
    if val <= 0:
        return "-"
    return f"{val:.1f}"


def _header(model_w: int, api_w: int) -> str:
    return (
        f"{'Model':<{model_w}}  "
        f"{'Status':<{_COL_STATUS}}  "
        f"{'API':<{api_w}}  "
        f"{'Total':<{_COL_LATENCY}}  "
        f"{'TTFT':<{_COL_TTFT}}  "
        f"{'TPS':<{_COL_TPS}}\n"
        f"{'-'*model_w}  "
        f"{'-'*_COL_STATUS}  "
        f"{'-'*api_w}  "
        f"{'-'*_COL_LATENCY}  "
        f"{'-'*_COL_TTFT}  "
        f"{'-'*_COL_TPS}"
    )


def _row(r: TestResult, model_w: int, api_w: int) -> str:
    return (
        f"{r.model:<{model_w}}  "
        f"{_icon(r.status):<2}{r.status:<{_COL_STATUS - 2}} "
        f"{(r.api_type or '-'):<{api_w}}  "
        f"{_pad(r.latency_ms):<{_COL_LATENCY}}  "
        f"{_pad(r.ttft_ms):<{_COL_TTFT}}  "
        f"{_tps(r.tokens_per_sec):<{_COL_TPS}}"
    )


def _summary(results: list[TestResult]) -> str:
    """按状态汇总: '5 ok, 1 timeout, 2 error'."""
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    order = ["ok", "timeout", "unauthorized", "error"]
    parts = [f"{counts[s]} {s}" for s in order if s in counts]
    for s, n in counts.items():  # 兜底: 未知状态
        if s not in order:
            parts.append(f"{n} {s}")
    return ", ".join(parts)


def _print_table(results: list[TestResult]) -> None:
    if not results:
        return
    model_w = max(max(len(r.model) for r in results), 5)
    api_w = max(max(len(r.api_type or "-") for r in results), len("API"))
    click.echo(f"Provider: {results[0].provider}")
    click.echo(_header(model_w, api_w))
    for r in results:
        click.echo(_row(r, model_w, api_w))


def _print_json(results: list[TestResult]) -> None:
    click.echo(json.dumps(
        [r.model_dump(mode="json") for r in results], indent=2, ensure_ascii=False,
    ))


# ── 单 Provider 测试 ──

def _run_models(
    pc: ProviderConfig, p, provider_key: str, models: list[str],
    *, parallel: int, timeout: float, on_result=None,
) -> list[TestResult]:
    """并发测试模型列表, on_result 在每个结果就绪时回调 (线程主序不定)."""
    def run(model: str) -> TestResult:
        return p.test_model(model, pc.api_key, provider_key=provider_key, timeout=timeout)

    by_model: dict[str, TestResult] = {}
    if parallel <= 1:
        for model in models:
            r = run(model)
            by_model[model] = r
            if on_result:
                on_result(r)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {ex.submit(run, m): m for m in models}
            for fut in as_completed(futs):
                r = fut.result()
                by_model[futs[fut]] = r
                if on_result:
                    on_result(r)

    # 按模型原始顺序返回
    return [by_model[m] for m in models if m in by_model]


def _test_provider_models(
    config, pc: ProviderConfig, p, provider_key: str, models: list[str],
    *, parallel: int, timeout: float, stream: bool = True,
) -> list[TestResult]:
    """测试指定模型列表, 流式打印表格, 结束后回写 api_type 缓存."""
    if not models:
        return []

    model_w = max(len(m) for m in models)
    api_w = _COL_API_MAX
    on_result = None
    if stream:
        click.echo(f"Provider: {provider_key}")
        click.echo(_header(model_w, api_w))
        on_result = lambda r: click.echo(_row(r, model_w, api_w))  # noqa: E731

    results = _run_models(
        pc, p, provider_key, models, parallel=parallel, timeout=timeout, on_result=on_result,
    )
    if stream:
        click.echo(f"  → {_summary(results)}")
    _save_cache(config, provider_key, pc, p)
    return results


def _test_provider(
    config, pc: ProviderConfig, p, provider_key: str,
    *, parallel: int, timeout: float, stream: bool = True,
) -> list[TestResult]:
    models = [m for m in (pc.model_names or p.list_models()) if m]
    return _test_provider_models(
        config, pc, p, provider_key, models, parallel=parallel, timeout=timeout, stream=stream,
    )


# ── command ──

@click.command(name="test")
@click.argument("target", required=False, shell_complete=complete_providers)
@click.argument("model_name", required=False, shell_complete=model_completer("target"))
@click.option("--format", "-f", "fmt", type=click.Choice(["text", "json"]), default="text", help="输出格式")
@click.option("--parallel", "-p", default=_DEFAULT_PARALLEL, type=int, help=f"并发数 (默认 {_DEFAULT_PARALLEL})")
@click.option("--timeout", "-t", default=_DEFAULT_TIMEOUT_S, type=float, help=f"单模型超时秒数 (默认 {int(_DEFAULT_TIMEOUT_S)})")
@click.option("--refresh", "-r", is_flag=True, default=False, help="清空 api_type 缓存, 强制重新探测接口")
def test(
    target: str | None, model_name: str | None, fmt: str,
    parallel: int, timeout: float, refresh: bool,
) -> None:
    """测试模型可用性和延迟 (stream 模式).

    \b
    示例:
      agentfly test                           # 全部 Provider 所有模型
      agentfly test deepseek                  # DeepSeek 所有模型
      agentfly test deepseek deepseek-chat    # 指定模型
      agentfly test deepseek 'deepseek-*'     # 通配符匹配一组模型
      agentfly test -p 8 -t 15                # 8 并发, 15s 超时
      agentfly test stepcode --refresh        # 强制重探接口
      agentfly test --format json             # JSON 输出
    """
    config, _ = ensure_config_exists()
    parallel = max(1, parallel)

    if target and model_name:
        # 单模型 (或通配符匹配的一组模型)
        pc, p = _resolve(config, target)
        if refresh:
            _clear_api_type(pc)
        models = _expand_models(pc, p, model_name)
        if not models:
            raise click.ClickException(f"未找到匹配模型: {model_name}")
        stream = fmt == "text"
        results = _test_provider_models(
            config, pc, p, target, models, parallel=parallel, timeout=timeout, stream=stream,
        )
        if fmt == "json":
            _print_json(results)
        return

    if target:
        # 单个 Provider
        pc, p = _resolve(config, target)
        if refresh:
            _clear_api_type(pc)
        results = _test_provider(
            config, pc, p, target, parallel=parallel, timeout=timeout, stream=(fmt == "text"),
        )
        if fmt == "json":
            _print_json(results)
        return

    # 全部 Provider
    for pk in list(config.providers):
        pc, p = _resolve(config, pk)
        if p is None:
            continue
        if refresh:
            _clear_api_type(pc)
        _test_provider(config, pc, p, pk, parallel=parallel, timeout=timeout)
        click.echo()
