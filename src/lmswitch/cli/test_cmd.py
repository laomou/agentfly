"""[lmswitch test] 测试模型可用性和速度 (stream 模式: TTFT + 吞吐)."""

from __future__ import annotations

import json
import sys

import click

from lmswitch.core.config import ensure_config_exists
from lmswitch.core.resolver import ConfigResolver
from lmswitch.providers.registry import get_provider
from lmswitch.models.schema import ProviderConfig, TestResult


def _make_provider(pc):
    return get_provider(pc)


def _first_endpoint(pc: ProviderConfig) -> str:
    return next(iter(pc.endpoints.values()), "") if pc.endpoints else ""


@click.command(name="test")
@click.argument("target", required=False)
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text", help="输出格式")
def test(target: str | None, output_format: str) -> None:
    """测试模型可用性和延迟 (stream 模式).

    \b
    示例:
      lmswitch test                           # 所有 Provider 默认模型
      lmswitch test deepseek                  # DeepSeek 所有模型
      lmswitch test deepseek:deepseek-chat    # 指定 Provider:Model
      lmswitch test --format json             # JSON 输出
    """
    config, _ = ensure_config_exists()
    results: list[TestResult] = []

    if target and ":" in target:
        pn, model = target.split(":", 1)
        results = _test_model(config, pn, model)
    elif target:
        results = _test_provider_models(config, target)
    else:
        for pk, pc in config.providers.items():
            p = _make_provider(pc)
            if p is None:
                continue
            model = pc.default_model or (pc.models[0] if pc.models else "")
            if model:
                results.append(p.test_model(model, pc.api_key, api_base=_first_endpoint(pc)))

    if output_format == "json":
        click.echo(json.dumps([r.model_dump(mode="json") for r in results], indent=2, ensure_ascii=False))
    else:
        _print_results_table(results)


def _test_model(config, provider_name: str, model: str) -> list[TestResult]:
    resolver = ConfigResolver(config)
    try:
        pc = resolver.get_provider(provider_name)
    except KeyError as e:
        click.secho(f"Provider 未配置: {e}", fg="red")
        sys.exit(1)
    p = _make_provider(pc)
    if p is None:
        click.secho(f"不支持的 Provider: {pc.name.value}", fg="red")
        sys.exit(1)
    return [p.test_model(model, pc.api_key, api_base=_first_endpoint(pc))]


def _test_provider_models(config, provider_name: str) -> list[TestResult]:
    try:
        pc = ConfigResolver(config).get_provider(provider_name)
    except KeyError as e:
        click.secho(f"Provider 未配置: {e}", fg="red")
        return []
    except ValueError as e:
        click.secho(f"配置错误: {e}", fg="red")
        return []
    p = _make_provider(pc)
    if p is None:
        return []
    results = []
    for model in (pc.models or p.list_models()):
        results.append(p.test_model(model, pc.api_key, api_base=_first_endpoint(pc)))
    return results


def _print_results_table(results: list[TestResult]) -> None:
    """以表格形式打印测试结果."""
    if not results:
        click.echo("  无结果")
        return
    name_w = max(max(len(r.provider.value) for r in results), 8)
    model_w = max(max(len(r.model) for r in results), 5)
    status_w = 12
    latency_w = 7
    ttft_w = 6
    tps_w = 8

    click.echo(
        f"  {'Provider':<{name_w}}  {'Model':<{model_w}}  "
        f"{'Status':<{status_w}}  {'Total':>{latency_w}}  {'TTFT':>{ttft_w}}  {'TPS':>{tps_w}}"
    )
    click.echo(f"  {'-' * name_w}  {'-' * model_w}  {'-' * status_w}  {'-' * latency_w}  {'-' * ttft_w}  {'-' * tps_w}")

    for r in results:
        icon = _status_icon(r.status)
        click.echo(
            f"  {r.provider.value:<{name_w}}  {r.model:<{model_w}}  "
            f"{icon} {r.status:<{status_w - 2}}  "
            f"{_fmt_ms(r.latency_ms):>{latency_w}}  "
            f"{_fmt_ms(r.ttft_ms):>{ttft_w}}  "
            f"{_fmt_tps(r.tokens_per_sec):>{tps_w}}"
        )


def _status_icon(status: str) -> str:
    return {"ok": "✅", "timeout": "⏱️", "error": "❌", "unauthorized": "🔒"}.get(status, "❓")


def _fmt_ms(ms: float) -> str:
    if ms <= 0:
        return "  -"
    if ms < 1000:
        return f"{ms:4.0f}ms"
    return f"{ms/1000:.1f}s"


def _fmt_tps(tps: float) -> str:
    if tps <= 0:
        return "  -"
    return f"{tps:5.0f}/s"
