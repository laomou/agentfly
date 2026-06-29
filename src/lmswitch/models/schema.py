"""Pydantic 数据模型 — 统一配置 Schema."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from lmswitch.models.types import AgentType, ProviderType

# ──────────────────────────────────────────
# Provider
# ──────────────────────────────────────────


class ProviderConfig(BaseModel):
    """单个服务提供商的配置."""

    name: ProviderType = Field(description="Provider 类型")
    api_key: str = Field(description="API Key，支持 ${ENV_VAR} 引用")
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="API 格式 → Base URL 映射，如 {'openai': 'https://api.deepseek.com'}",
    )
    models: list[str] = Field(default_factory=list, description="可用模型列表")
    default_model: str = Field(default="", description="默认模型")
    extra_env: dict[str, str] = Field(
        default_factory=dict, description="额外环境变量"
    )

    @field_validator("default_model", mode="before")
    @classmethod
    def set_default_model(cls, v: str, info) -> str:
        if not v and info.data.get("models"):
            return info.data["models"][0]
        return v


# ──────────────────────────────────────────
# Agent
# ──────────────────────────────────────────


class AgentConfig(BaseModel):
    """单个 Agent 的配置."""

    name: AgentType = Field(description="Agent 名称")
    provider: str = Field(description="绑定的 Provider 键名 (如 'anthropic' / 'my-proxy')")
    model: Optional[str] = Field(default=None, description="覆盖默认模型")
    extra_args: list[str] = Field(
        default_factory=list, description="启动时的额外 CLI 参数"
    )
    env_overrides: dict[str, str] = Field(
        default_factory=dict, description="环境变量覆盖"
    )


# ──────────────────────────────────────────
# Unified Config
# ──────────────────────────────────────────


class UnifiedConfig(BaseModel):
    """用户统一配置文件 (~/.config/lmswitch/config.yaml) 的完整模型."""

    version: str = Field(default="1", description="配置版本")
    default_provider: str = Field(
        default="", description="默认 Provider 键名"
    )
    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict, description="Provider 配置映射"
    )
    agents: dict[str, AgentConfig] = Field(
        default_factory=dict, description="Agent 配置映射"
    )


# ──────────────────────────────────────────
# Resolved Config (内部使用)
# ──────────────────────────────────────────


class ResolvedConfig(BaseModel):
    """解析后的完整配置 — 供 AgentAdapter 使用.

    所有 ${ENV_VAR} 引用已被解析为真实值.
    """

    agent: AgentConfig
    provider: ProviderConfig
    effective_api_base: str = ""   # Agent 匹配到的 API Base URL
    effective_api_format: str = ""  # Agent 匹配到的 API 格式


# ──────────────────────────────────────────
# Test
# ──────────────────────────────────────────


class TestResult(BaseModel):
    """模型测试结果."""

    provider: ProviderType
    model: str
    status: str  # "ok" | "timeout" | "error" | "unauthorized"
    latency_ms: float = 0.0     # 总响应时间
    ttft_ms: float = 0.0         # Time To First Token
    tokens_per_sec: float = 0.0  # 吞吐量 (tokens/s)
    error_message: str = ""
