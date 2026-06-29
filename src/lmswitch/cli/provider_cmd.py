"""[lmswitch provider] 管理服务提供商."""

from __future__ import annotations

import re
import sys

import click
import httpx

from lmswitch.core.config import ensure_config_exists
from lmswitch.models.schema import ProviderConfig
from lmswitch.models.types import ProviderType
from lmswitch.providers.manager import ProviderManager

# ── 已知 Provider 厂商 ──
# 格式: { "vendor": {"endpoints": {"openai": "url", ...}, "models": [...] } }
_KNOWN_DEFAULTS: dict[str, dict] = {
    "openai": {
        "endpoints": {"openai": "https://api.openai.com"},
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o3", "o4-mini"],
    },
    "deepseek": {
        "endpoints": {
            "openai": "https://api.deepseek.com",
            "anthropic": "https://api.deepseek.com/anthropic",
        },
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "moonshot": {
        "endpoints": {"openai": "https://api.moonshot.cn"},
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
    },
    "zhipu": {
        "endpoints": {"openai": "https://open.bigmodel.cn/api/paas/v4"},
        "models": ["glm-4-plus", "glm-4-flash", "glm-4-air"],
    },
    "qwen": {
        "endpoints": {"openai": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
        "models": ["qwen-max", "qwen-plus", "qwen-turbo"],
    },
    "siliconflow": {
        "endpoints": {"openai": "https://api.siliconflow.cn"},
        "models": ["Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3"],
    },
    "together": {
        "endpoints": {"openai": "https://api.together.xyz"},
        "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
    },
    "groq": {
        "endpoints": {"openai": "https://api.groq.com/openai/v1"},
        "models": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    },
    "openrouter": {
        "endpoints": {"openai": "https://openrouter.ai/api/v1"},
        "models": ["openai/gpt-4o", "anthropic/claude-sonnet-4-6"],
    },
    "perplexity": {
        "endpoints": {"openai": "https://api.perplexity.com"},
        "models": ["sonar-pro", "sonar"],
    },
    "cerebras": {
        "endpoints": {"openai": "https://api.cerebras.ai/v1"},
        "models": ["llama3.1-8b", "llama3.1-70b"],
    },
    "mistral": {
        "endpoints": {"openai": "https://api.mistral.ai/v1"},
        "models": ["mistral-large-latest", "mistral-small-latest"],
    },
    "xai": {
        "endpoints": {"openai": "https://api.x.ai/v1"},
        "models": ["grok-2", "grok-2-mini"],
    },
    "minimax": {
        "endpoints": {"openai": "https://api.minimax.chat/v1"},
        "models": ["abab6.5s-chat", "abab7-chat"],
    },
    "fireworks": {
        "endpoints": {"openai": "https://api.fireworks.ai/inference/v1"},
        "models": ["accounts/fireworks/models/llama-v3p3-70b-instruct"],
    },
    "anthropic": {
        "endpoints": {"anthropic": "https://api.anthropic.com"},
        "models": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
    },
    "google": {
        "endpoints": {},  # 不使用标准 OpenAI/Anthropic 协议
        "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    },
}

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
        click.echo("使用 'lmswitch provider add <name> --api-base <url> --api-key <key>' 添加.")
        return

    click.echo("已配置的 Provider:")
    click.echo()
    for key, p in config.providers.items():
        default_mark = " ★" if key == config.default_provider else ""
        formats = ", ".join(p.endpoints.keys()) if p.endpoints else "(无)"
        n_models = len(p.models)
        click.echo(f"  {key:<20} [{formats}]")
        for fmt, url in p.endpoints.items():
            click.echo(f"    {fmt}: {url}")
        click.echo(f"    默认: {p.default_model or '(未设置)'}  |  "
                    f"{n_models} 个可用模型{default_mark}")
        click.echo()


# ── provider add ──


def _auto_name(api_base: str) -> str:
    """从 API Base URL 自动生成 Provider 名称.

    http://10.235.115.58:3000  →  10.235.115.58
    https://api.openai.com    →  api.openai.com
    """
    m = re.search(r'https?://([^/]+)', api_base)
    if not m:
        return "custom"
    return m.group(1).split(":")[0]


@provider_group.command(name="add")
@click.argument("name", required=False)
@click.option("--api-base", default=None, help="API Base URL (已知厂商自动填充)")
@click.option("--api-key", default=None, required=True, help="API Key (支持 ${ENV_VAR} 引用)")
@click.option("--models", default=None, help="手动指定模型列表，逗号分隔")
def add_provider(name: str | None, api_base: str | None, api_key: str,
                  models: str | None):
    """添加 Provider 配置.

    \b
    名称从 --api-base 自动生成，也可手动指定:
      lmswitch provider add --api-base http://10.235.115.58:3000 --api-key '$MY_KEY'
      → 自动生成名称: 10-235-115-58-3000

    \b
    手动指定名称:
      lmswitch provider add my-proxy --api-base http://... --api-key '${MY_KEY}'

    \b
    内置 Provider (只需 --api-key):
      lmswitch provider add anthropic --api-key '$ANTHROPIC_API_KEY'
      lmswitch provider add openai --api-key '$OPENAI_API_KEY'
    """
    # ── 判断 provider 类型 & 名称 ──
    defaults = _KNOWN_DEFAULTS.get((name or "").lower(), {})

    if name:
        try:
            provider_type = ProviderType(name.lower())
            provider_key = provider_type.value
            is_custom = False
        except ValueError:
            provider_type = ProviderType.CUSTOM
            provider_key = name.lower().replace(" ", "-")
            is_custom = True
    else:
        provider_key = _auto_name(api_base or "")
        provider_type = ProviderType.CUSTOM
        is_custom = True
        click.echo(f"  自动生成名称: {provider_key}")

    # ── api_base: 已知厂商自动填充 ──
    if not api_base:
        preset = defaults.get("endpoints", {})
        if preset:
            api_base = next(iter(preset.values()))
        elif is_custom:
            api_base = click.prompt("API Base URL", default="http://localhost:8000")

    # ── 加载配置 ──
    config, cfg_path = ensure_config_exists()
    mgr = ProviderManager(config)

    if mgr.get(provider_key):
        if not click.confirm(f"Provider '{provider_key}' 已存在，是否覆盖?"):
            return

    # ── API Key 安全处理 ──
    if _is_plaintext_key(api_key):
        env_var = f"LMSWITCH_{provider_key.upper().replace('-', '_').replace('.', '_')}_KEY"
        click.echo()
        click.secho("  ⚠ API Key 不能明文存储在配置文件中", fg="yellow")
        click.secho(f"  → 已自动转为环境变量引用: ${{{env_var}}}", fg="green")
        click.secho(f"  → 请将以下内容添加到 ~/.bashrc 或 ~/.zshrc:", fg="green")
        click.secho(f"", fg="green")
        click.secho(f"      export {env_var}=\"{api_key}\"", fg="bright_white")
        click.secho(f"")
        click.secho(f"  然后执行: source ~/.bashrc  (或 ~/.zshrc)", fg="green")
        click.echo()
        api_key = f"${{{env_var}}}"

    # ── 自动探测 endpoints ──
    endpoints: dict[str, str] = {}
    model_list: list[str] = []

    preset_endpoints = defaults.get("endpoints", {})
    preset_models = defaults.get("models", [])

    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]

    if is_custom:
        click.echo(f"  探测 API 端点...")
        endpoints = _detect_endpoints(api_base, api_key)
        if endpoints:
            click.echo(f"  ✓ 可用格式: {', '.join(endpoints.keys())}")
        else:
            click.echo(f"  ⚠ 无法自动探测，手动选择格式")
            fmt = click.prompt("API 格式", type=click.Choice(["openai", "anthropic"]), default="openai")
            endpoints = {fmt: api_base}
    else:
        endpoints = preset_endpoints

    # ── 拉取模型 ──
    if not model_list and "openai" in endpoints:
        click.echo(f"  拉取可用模型...")
        model_list = _fetch_models(endpoints["openai"], api_key)
        if model_list:
            click.echo(f"  ✓ 发现 {len(model_list)} 个模型")
    if not model_list and preset_models:
        model_list = preset_models
        click.echo(f"  使用预设模型 ({len(model_list)} 个): {', '.join(model_list[:6])}"
                   f"{'...' if len(model_list) > 6 else ''}")
    if not model_list:
        models_str = click.prompt("模型列表 (逗号分隔)", default="")
        model_list = [m.strip() for m in models_str.split(",") if m.strip()] if models_str else []

    default_model = model_list[0] if model_list else ""

    # ── 保存 ──
    provider_config = ProviderConfig(
        name=provider_type,
        api_key=api_key,
        endpoints=endpoints,
        models=model_list,
        default_model=default_model,
    )

    mgr.add(provider_config, key=provider_key)

    if len(config.providers) == 1:
        mgr.set_default(provider_key)
        click.echo(f"  已设为默认 Provider")

    mgr.save()
    click.echo(f"  ✓ Provider '{provider_key}' → {cfg_path}")


# ── provider models ──


@provider_group.command(name="models")
@click.argument("name")
@click.option("--refresh", "-r", is_flag=True, default=False, help="重新从 API 拉取模型列表")
def list_models(name: str, refresh: bool):
    """查看 / 刷新 Provider 的可用模型.

    \b
    示例:
      lmswitch provider models my-proxy
      lmswitch provider models my-proxy --refresh
    """
    config, cfg_path = ensure_config_exists()
    mgr = ProviderManager(config)

    provider = mgr.get(name)
    if provider is None:
        click.echo(f"错误: Provider '{name}' 未配置")
        sys.exit(1)

    if refresh:
        if "openai" not in provider.endpoints:
            click.echo(f"  ⚠ 无 OpenAI endpoint，无法拉取模型")
            return
        click.echo(f"  从 API 拉取模型列表...")
        new_models = _fetch_models(provider.endpoints["openai"], provider.api_key)
        if new_models:
            provider.models = new_models
            provider.default_model = new_models[0]
            mgr.add(provider, key=name)
            mgr.save()
            click.echo(f"  ✓ 更新了 {len(new_models)} 个模型 → {cfg_path}")
        else:
            click.echo(f"  ⚠ 无法拉取模型列表，保留现有 {len(provider.models)} 个模型")
            return

    if not provider.models:
        click.echo("(无)")
        return

    click.echo(f"  {name} — 可用模型 ({len(provider.models)}):")
    click.echo()
    for m in provider.models:
        mark = " ★" if m == provider.default_model else ""
        click.echo(f"    {m}{mark}")


# ── provider rm ──


@provider_group.command(name="rm")
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


# ── provider set-default ──


@provider_group.command(name="set-default")
@click.argument("name")
def set_default_provider(name: str):
    """设置默认 Provider."""
    config, cfg_path = ensure_config_exists()
    mgr = ProviderManager(config)
    try:
        mgr.set_default(name)
    except KeyError as e:
        click.echo(f"错误: {e}")
        sys.exit(1)
    mgr.save()
    click.echo(f"  ✓ 默认 Provider → {name}")


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
