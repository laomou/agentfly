"""[agentfly test] 测试模型可用性和速度 (stream 模式: TTFT + 吞吐)."""

from __future__ import annotations

import json

import click

from agentfly.core.config import ensure_config_exists
from agentfly.core.resolver import ConfigResolver
from agentfly.models.schema import ProviderConfig, TestResult
from agentfly.providers.registry import get_provider


# ── helpers ──

def _resolve(config, name: str):
    """解析 Provider 名称 → (ProviderConfig, Provider 实例)."""
    try:
        pc = ConfigResolver(config).get_provider(name)
    except KeyError:
        raise click.ClickException(f"Provider 未配置: {name}")
    p = get_provider(pc)
    if p is None:
        raise click.ClickException(f"不支持的 Provider: {pc.name.value}")
    return pc, p


def _test_models(pc, p, provider_key: str, stream: bool = False) -> list[TestResult]:
    """测试 provider 所有模型，stream=True 时逐个输出."""
    results: list[TestResult] = []
    models = [m for m in (pc.models or p.list_models()) if m]
    mw = max((len(m) for m in models), default=5)
    sw, lw, tw, pw = 14, 8, 8, 8  # status, latency, ttft, tps
    if stream and models:
        click.echo(f"Provider: {provider_key}")
        click.echo(f"{'Model':<{mw}}  {'Status':<{sw}}  {'Total':<{lw}}  {'TTFT':<{tw}}  {'TPS':<{pw}}")
        click.echo(f"{'-'*mw}  {'-'*sw}  {'-'*lw}  {'-'*tw}  {'-'*pw}")
    for model in models:
        r = p.test_model(model, pc.api_key, _base(pc), provider_key=provider_key)
        results.append(r)
        if stream:
            click.echo(f"{r.model:<{mw}}  {_icon(r.status):<2}{r.status:<{sw-2}} {_pad(r.latency_ms):<{lw}}  {_pad(r.ttft_ms):<{tw}}  {_pad(r.tokens_per_sec):<{pw}}")
    return results


# ── output ──

def _base(pc: ProviderConfig) -> str:
    return next(iter(pc.endpoints.values()), "") if pc.endpoints else ""


def _icon(status: str) -> str:
    return {"ok": "✅", "timeout": "⏳", "error": "❌", "unauthorized": "❌"}.get(status, "❓")


def _pad(val: float) -> str:
    """格式化数值: ms/s 或 `-`."""
    if val <= 0:
        return "-"
    if val < 1000:
        return f"{val:.0f}ms"
    return f"{val/1000:.1f}s"


def _print_table(results: list[TestResult]) -> None:
    """表格打印测试结果."""
    if not results:
        return

    mw = max(max(len(r.model) for r in results), 5)
    sw, lw, tw, pw = 14, 8, 8, 8  # status, latency, ttft, tps

    click.echo(f"Provider: {results[0].provider}")
    click.echo(f"{'Model':<{mw}}  {'Status':<{sw}}  {'Total':<{lw}}  {'TTFT':<{tw}}  {'TPS':<{pw}}")
    click.echo(f"{'-'*mw}  {'-'*sw}  {'-'*lw}  {'-'*tw}  {'-'*pw}")

    for r in results:
        click.echo(
            f"{r.model:<{mw}}  "
            f"{_icon(r.status):<2}{r.status:<{sw - 2}} "
            f"{_pad(r.latency_ms):<{lw}}  "
            f"{_pad(r.ttft_ms):<{tw}}  "
            f"{_pad(r.tokens_per_sec):<{pw}}"
        )


def _print_json(results: list[TestResult]) -> None:
    click.echo(json.dumps([r.model_dump(mode="json") for r in results], indent=2, ensure_ascii=False))


# ── command ──

@click.command(name="test")
@click.argument("target", required=False)
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text", help="输出格式")
def test(target: str | None, fmt: str) -> None:
    """测试模型可用性和延迟 (stream 模式).

    \b
    示例:
      agentfly test                           # 全部 Provider 所有模型
      agentfly test deepseek                  # DeepSeek 所有模型
      agentfly test deepseek:deepseek-chat    # 指定模型
      agentfly test --format json             # JSON 输出
    """
    config, _ = ensure_config_exists()

    if target and ":" in target:
        # 单模型
        pn, model = target.split(":", 1)
        pc, p = _resolve(config, pn)
        r = p.test_model(model, pc.api_key, _base(pc), provider_key=pn)
        _print_table([r]) if fmt == "text" else _print_json([r])

    elif target:
        # 单个 Provider → 流式输出
        pc, p = _resolve(config, target)
        results = _test_models(pc, p, target, stream=True)
        if fmt == "json":
            _print_json(results)

    else:
        # 全部 Provider → 逐个流式输出
        for pk, pc in config.providers.items():
            p = get_provider(pc)
            if p is None:
                continue
            _test_models(pc, p, pk, stream=True)
            click.echo()
