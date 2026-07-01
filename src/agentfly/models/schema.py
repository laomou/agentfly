"""Pydantic 数据模型 — 统一配置 Schema."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from agentfly.models.types import AgentType, ProviderType

# ──────────────────────────────────────────
# Provider
# ──────────────────────────────────────────


class ModelEntry(BaseModel):
    """单个模型的信息, 配置 yaml 中 models 的成员."""

    name: str = Field(description="模型名称")
    api_type: str = Field(default="", description="跑通接口: openai / anthropic, test 自动填充")

    model_config = ConfigDict(extra="ignore")


class ProviderConfig(BaseModel):
    """单个服务提供商的配置."""

    model_config = ConfigDict(extra="ignore")

    name: ProviderType = Field(description="Provider 类型")
    api_key: str = Field(description="API Key，支持 ${ENV_VAR} 引用")
    base_url: str = Field(default="", description="API Base URL")
    models: list[ModelEntry] = Field(default_factory=list, description="可用模型列表")
    default_model: str = Field(default="", description="默认模型")

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.models]

    @field_validator("default_model", mode="before")
    @classmethod
    def set_default_model(cls, v: str, info) -> str:
        if not v and info.data.get("models"):
            models = info.data["models"]
            if models:
                first = models[0]
                return first.name if isinstance(first, ModelEntry) else first
        return v

    @field_validator("models", mode="before")
    @classmethod
    def coerce_models(cls, v):
        """兼容旧配置: models 是 list[str]"""
        if isinstance(v, list):
            return [
                ModelEntry(name=s) if isinstance(s, str) else s
                for s in v
            ]
        return v

    @field_serializer("models")
    def serialize_models(self, v):
        """无 api_type → 序列化为字符串, 有 api_type → 序列化为 dict."""
        return [
            m.name if not m.api_type else m.model_dump(mode="json")
            for m in v
        ]


# ──────────────────────────────────────────
# Agent
# ──────────────────────────────────────────


class AgentConfig(BaseModel):
    """单个 Agent 的配置."""

    model_config = ConfigDict(extra="ignore")

    name: AgentType = Field(description="Agent 名称")
    provider: str = Field(description="绑定的 Provider 键名 (如 'anthropic' / 'my-proxy')")
    model: Optional[str] = Field(default=None, description="覆盖默认模型")


# ──────────────────────────────────────────
# Unified Config
# ──────────────────────────────────────────


class UnifiedConfig(BaseModel):
    """用户统一配置文件 (~/.config/agentfly/config.yaml) 的完整模型."""

    model_config = ConfigDict(extra="ignore")

    version: str = Field(default="1", description="配置版本")
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
    """解析后的完整配置 — 供 Agent 使用.

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

    provider: str
    model: str
    status: str  # "ok" | "timeout" | "error" | "unauthorized"
    status_code: int = 0         # HTTP 状态码 (0 = 非 HTTP 错误)
    latency_ms: float = 0.0     # 总响应时间
    ttft_ms: float = 0.0         # Time To First Token
    tokens_per_sec: float = 0.0  # 吞吐量 (tokens/s)
    error_message: str = ""
