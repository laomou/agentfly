"""[agentfly provider] 管理服务提供商."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click
import httpx

from agentfly.core.config import ensure_config_exists
from agentfly.models.schema import ProviderConfig
from agentfly.models.types import ProviderType
from agentfly.providers.manager import ProviderManager

# ── 远端厂商配置 ──

_GITHUB_RAW = "https://raw.githubusercontent.com/laomou/agentfly/main/src/agentfly"
_KNOWN_CACHE: dict[str, dict] | None = None


def _get_known_providers() -> dict[str, dict]:
    """从 GitHub 拉取已知厂商列表，失败降级本地 providers.json.

    每进程缓存一次，不重复拉取.
    """
    global _KNOWN_CACHE
    if _KNOWN_CACHE is not None:
        return _KNOWN_CACHE

    url = f"{_GITHUB_RAW}/providers/providers.json"
    try:
        resp = httpx.get(url, timeout=3)
        if resp.status_code == 200:
            _KNOWN_CACHE = resp.json()
            # 缓存到磁盘，离线可用
            cache_dir = Path.home() / ".config" / "agentfly" / "env"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "providers.json").write_text(resp.text, encoding="utf-8")
            click.echo(f"  ✓ 获取厂商列表")
            return _KNOWN_CACHE
    except Exception:
        pass
    click.echo(f"  使用本地厂商列表")

    # 用户缓存 → 包默认
    for src in [
        Path.home() / ".config" / "agentfly" / "env" / "providers.json",
        Path(__file__).resolve().parent.parent / "providers" / "providers.json",
    ]:
        if src.exists():
            _KNOWN_CACHE = json.loads(src.read_text())
            return _KNOWN_CACHE
    _KNOWN_CACHE = {}
    return {}


def _refresh_env_cache(provider_key: str) -> None:
    """从 GitHub 拉取 env 配置并缓存到 ~/.config/agentfly/env/."""
    click.echo(f"  获取远程环境配置...")
    url = f"{_GITHUB_RAW}/providers/{provider_key}.json"
    try:
        resp = httpx.get(url, timeout=5)
        if resp.status_code != 200:
            return
        cache_dir = Path.home() / ".config" / "agentfly" / "env"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{provider_key}.json").write_text(resp.text, encoding="utf-8")
        click.echo(f"  ✓ 环境配置已缓存")
    except Exception:
        click.secho(f"  ⚠ 无法获取远程配置，使用本地默认", fg="yellow")


def _mask_key(key: str) -> str:
    """脱敏 API Key 显示. sk-abc...xyz"""
    if key.startswith("${") and key.endswith("}"):
        return key  # env var 引用，不脱敏
    if len(key) <= 8:
        return key[:2] + "***" + key[-2:]
    return key[:5] + "..." + key[-4:]


def _is_plaintext_key(key: str) -> bool:
    """检查是否为明文 API Key（非 env var 引用）."""
    return not (key.startswith("${") and key.endswith("}"))


def _probe_endpoint(api_base: str, api_key: str, fmt: str) -> bool:
    """探测指定格式的 endpoint 是否可用."""
    base = api_base.rstrip("/")
    endpoint = f"{base}/v1/chat/completions" if fmt == "openai" else f"{base}/v1/messages"
    body = {"model": "ping", "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}

    try:
        with httpx.Client(timeout=httpx.Timeout(8.0)) as client:
            resp = client.post(
                endpoint, json=body,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            return resp.status_code != 404
    except (httpx.TimeoutException, httpx.ConnectError):
        return False


def _detect_endpoints(api_base: str, api_key: str) -> dict[str, str]:
    """自动探测可用的 API endpoints.

    同时探测 OpenAI (/v1/chat/completions) 和 Anthropic (/v1/messages).
    返回: {"openai": "http://...", ...} - 同 base URL 不同格式.
    """
    base = api_base.rstrip("/")
    endpoints: dict[str, str] = {}
    for fmt in ("openai", "anthropic"):
        if _probe_endpoint(api_base, api_key, fmt):
            endpoints[fmt] = base
    return endpoints


def _fetch_models(api_base: str, api_key: str, api_format: str = "openai") -> list[str]:
    """从 API 拉取可用模型列表 (OpenAI 兼容 GET /v1/models)."""
    base = api_base.rstrip("/")

    with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
        if api_format == "openai":
            try:
                resp = client.get(
                    f"{base}/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m["id"] for m in data.get("data", [])]
                    # 过滤：只保留 chat/language 模型
                    return sorted(
                        [m for m in models if not any(
                            x in m for x in ["embedding", "moderation", "dall-e", "whisper", "tts"]
                        )]
                    )
            except Exception:
                pass

        # Anthropic 或拉取失败：返回空，用户手动指定
        return []


# ──────────────────────────────────────────
# provider group
# ──────────────────────────────────────────


@click.group(name="provider")
def provider_group():
    """管理服务提供商配置."""
    pass


# ── provider list ──


@provider_group.command(name="list")
def list_providers():
    """列出所有已配置的 Provider."""
    config, _ = ensure_config_exists()
    if not config.providers:
        click.echo("暂无已配置的 Provider.")
        click.echo("使用 'agentfly provider add [name] --api-base <url> --api-key <key>' 添加.")
        return

    click.echo("已配置的 Provider:")
    click.echo()
    for key, p in config.providers.items():
        formats = ", ".join(p.endpoints.keys()) if p.endpoints else "(无)"
        n_models = len(p.models)
        click.echo(f"  {key:<20} [{formats}]")
        for fmt, url in p.endpoints.items():
            click.echo(f"    {fmt}: {url}")
        click.echo(f"    默认: {p.default_model or '(未设置)'}  |  {n_models} 个可用模型")
        click.echo()


# ── provider add ──


def _auto_name(api_base: str) -> str:
    """从 API Base URL 自动生成 Provider 名称.

    http://host:3000  →  host
    host:3000         →  host
    https://api.openai.com     →  api.openai.com
    """
    # 补全协议头
    if not re.match(r'https?://', api_base):
        api_base = f"http://{api_base}"
    m = re.search(r'https?://([^/]+)', api_base)
    if not m:
        return "custom"
    host = m.group(1).split(":")[0]
    # localhost 太通用，加上端口区分
    if host == "localhost":
        port = m.group(1).split(":")[1] if ":" in m.group(1) else "8000"
        return f"localhost:{port}"
    return host


@provider_group.command(name="add")
@click.argument("name", required=False)
@click.option("--api-base", default=None, help="API Base URL (已知厂商自动填充)")
@click.option("--api-key", default=None, help="API Key (支持 ${ENV_VAR} 引用，留空则交互式输入)")
@click.option("--models", default=None, help="手动指定模型列表，逗号分隔")
def add_provider(name: str | None, api_base: str | None, api_key: str | None,
                  models: str | None):
    """添加 Provider 配置.

    \b
    名称从 --api-base 自动生成，也可手动指定:
      agentfly provider add --api-base http://your-host:3000 --api-key '$MY_KEY'
      → 自动生成名称: your-host

    \b
    手动指定名称:
      agentfly provider add my-proxy --api-base http://... --api-key '${MY_KEY}'

    \b
    内置 Provider (只需 --api-key):
      agentfly provider add deepseek --api-key '$DEEPSEEK_API_KEY'
      agentfly provider add anthropic --api-key '$ANTHROPIC_API_KEY'
    """
    known_providers = _get_known_providers()  # 先拉取，避免网络卡交互

    # 1. 确定 api_base: 已知厂商 > --api-base > 交互输入
    known = known_providers.get(name.lower()) if name else None
    if known and not api_base:
        api_base = known["api_base"]
    if not api_base:
        api_base = click.prompt("API Base URL", default="http://localhost:8000")

    # 2. 确定名称: 参数 > 自动提取
    if name:
        provider_key = name.lower().replace(" ", "-")
    else:
        provider_key = _auto_name(api_base)
        click.echo(f"  自动生成名称: {provider_key}")
        # 自动名称也可能命中已知厂商
        if not known:
            known = known_providers.get(provider_key)

    if known:
        click.echo(f"  已知厂商: {provider_key} → {api_base}")

    # ── 加载配置 ──
    config, cfg_path = ensure_config_exists()
    mgr = ProviderManager(config)

    if mgr.get(provider_key):
        if not click.confirm(f"Provider '{provider_key}' 已存在，是否覆盖?"):
            return

    # ── API Key ──
    if not api_key:
        api_key = click.prompt("API Key (${ENV} 或直接输入)")

    if _is_plaintext_key(api_key):
        click.echo(f"  ✓ Key: {_mask_key(api_key)}")
        click.secho("  ⚠ 明文存储不安全，建议用 ${MY_KEY}", fg="yellow")
        if not click.confirm("  仍然存储?", default=False):
            return
    else:
        click.echo(f"  ✓ Key: {api_key}")

    # ── 自动探测 endpoints ──
    model_list: list[str] = []
    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]

    if _is_plaintext_key(api_key):
        click.echo(f"  探测 API 端点...")
        endpoints = _detect_endpoints(api_base, api_key)
    else:
        # env var 引用无法探测，直接用已知配置或手动选择
        endpoints: dict[str, str] = {}
        click.echo(f"  ${api_key[2:-1]} 环境变量引用，跳过探测")
    if known:
        # 已知厂商: 补全探测没识别到的格式
        for fmt, url in known["endpoints"].items():
            if fmt not in endpoints:
                endpoints[fmt] = url
                click.echo(f"  + {fmt} (已知默认)")
        click.echo(f"  ✓ 格式: {', '.join(endpoints.keys())}")
    elif endpoints:
        click.echo(f"  ✓ 可用格式: {', '.join(endpoints.keys())}")
    else:
        click.echo(f"  ⚠ 无法自动探测，手动选择格式")
        fmt_str = click.prompt("API 格式 (openai/anthropic, 逗号分隔)", default="openai")
        endpoints = {}
        for f in fmt_str.split(","):
            f = f.strip()
            if f in ("openai", "anthropic"):
                endpoints[f] = api_base
        if not endpoints:
            endpoints = {"openai": api_base}

    # ── 拉取模型 ──
    if not model_list and "openai" in endpoints and _is_plaintext_key(api_key):
        click.echo(f"  拉取可用模型...")
        model_list = _fetch_models(endpoints["openai"], api_key)
        if model_list:
            click.echo(f"  ✓ 发现 {len(model_list)} 个模型")
    if not model_list:
        models_str = click.prompt("模型列表 (逗号分隔)", default="")
        model_list = [m.strip() for m in models_str.split(",") if m.strip()] if models_str else []

    default_model = model_list[0] if model_list else ""

    provider_type = ProviderType(known["type"]) if known else ProviderType.CUSTOM
    provider_config = ProviderConfig(
        name=provider_type,
        api_key=api_key,
        endpoints=endpoints,
        models=model_list,
        default_model=default_model,
    )

    mgr.add(provider_config, key=provider_key)
    mgr.save()
    _refresh_env_cache(provider_key)
    click.echo(f"  ✓ Provider '{provider_key}' ({provider_type.value}) → {cfg_path}")


# ── provider reload ──


@provider_group.command(name="reload")
@click.argument("name")
def reload_models(name: str):
    """从 API 重新拉取 Provider 模型列表.

    \b
    示例:
      agentfly provider reload deepseek
    """
    config, cfg_path = ensure_config_exists()
    mgr = ProviderManager(config)

    provider = mgr.get(name)
    if provider is None:
        click.secho(f"Provider '{name}' 未配置", fg="red")
        sys.exit(1)

    if "openai" not in provider.endpoints:
        click.secho("无 OpenAI endpoint，无法拉取模型", fg="yellow")
        return

    click.echo(f"  从 API 拉取模型...")
    new_models = _fetch_models(provider.endpoints["openai"], provider.api_key)
    if new_models:
        provider.models = new_models
        provider.default_model = new_models[0]
        mgr.add(provider, key=name)
        mgr.save()
        click.echo(f"  ✓ 更新 {len(new_models)} 个模型")
        for m in new_models:
            click.echo(f"    {m}")
    else:
        click.secho(f"  ⚠ 拉取失败，保留原有 {len(provider.models)} 个模型", fg="yellow")
    _refresh_env_cache(name)


# ── provider rm ──


@provider_group.command(name="remove")
@click.argument("name")
def remove_provider(name: str):
    """删除 Provider 配置."""
    config, cfg_path = ensure_config_exists()
    mgr = ProviderManager(config)
    try:
        mgr.remove(name)
    except KeyError as e:
        click.echo(f"错误: {e}")
        sys.exit(1)
    mgr.save()
    click.echo(f"  ✓ Provider '{name}' 已移除")


# ── provider show ──


@provider_group.command(name="show")
@click.argument("name")
def show_provider(name: str):
    """查看 Provider 详情."""
    config, _ = ensure_config_exists()
    mgr = ProviderManager(config)
    provider = mgr.get(name)
    if provider is None:
        click.echo(f"错误: Provider '{name}' 未配置")
        sys.exit(1)

    click.echo(f"  Key:        {name}")
    click.echo(f"  Type:       {provider.name.value}")
    click.echo(f"  API Key:    {_mask_key(provider.api_key)}")
    click.echo(f"  Endpoints:")
    if provider.endpoints:
        for fmt, url in provider.endpoints.items():
            click.echo(f"    {fmt}: {url}")
    else:
        click.echo(f"    (无)")
    click.echo(f"  Models:     {', '.join(provider.models) if provider.models else '(无)'}")
    click.echo(f"  Default:    {provider.default_model or '(未设置)'}")
