"""Pydantic 数据模型 — 统一配置 Schema."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentfly.models.types import AgentType, ProviderType

# ──────────────────────────────────────────
# Provider
# ──────────────────────────────────────────


class ProviderConfig(BaseModel):
    """单个服务提供商的配置."""

    model_config = ConfigDict(extra="ignore")

    name: ProviderType = Field(description="Provider 类型")
    api_key: str = Field(description="API Key，支持 ${ENV_VAR} 引用")
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="api_type → Base URL, 如 {openai: url1, anthropic: url2}",
    )
    models: dict[str, str] = Field(
        default_factory=dict,
        description="模型名 → api_type (openai/anthropic; 空=未探测, test 自动填充)",
    )
    default_model: str = Field(default="", description="默认模型")

    @property
    def model_names(self) -> list[str]:
        return list(self.models.keys())

    @field_validator("models", mode="before")
    @classmethod
    def coerce_models(cls, v):
        """归一到 dict[name, api_type]. 接受 list[str] / list[{name,api_type}] / dict."""
        if isinstance(v, dict):
            return {k: (val or "") for k, val in v.items()}
        if isinstance(v, list):
            out: dict[str, str] = {}
            for item in v:
                if isinstance(item, str):
                    out[item] = ""
                elif isinstance(item, dict) and item.get("name"):
                    out[item["name"]] = item.get("api_type") or ""
            return out
        return v

    @model_validator(mode="after")
    def _fill_default_model(self):
        # 空则取第一个模型作为默认
        if not self.default_model and self.models:
            self.default_model = next(iter(self.models))
        return self


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
    api_type: str = ""           # 跑通的接口类型 (openai / anthropic)
    latency_ms: float = 0.0     # 总响应时间
    ttft_ms: float = 0.0         # Time To First Token
    tokens_per_sec: float = 0.0  # 吞吐量 (tokens/s)
    error_message: str = ""
