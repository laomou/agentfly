"""枚举类型定义."""

from enum import Enum


class ProviderType(str, Enum):
    """服务提供商类型."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"


class AgentType(str, Enum):
    """AI Agent 类型."""

    CLAUDE = "claude"
    CLAUDE_CODE = "claude-code"
    CLINE = "cline"
    CODEX = "codex"
    DROID = "droid"
    OPENCODE = "opencode"
    OPENCLAW = "openclaw"
    PI = "pi"


class TestStatus(str, Enum):
    """模型测试状态."""

    OK = "ok"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNAUTHORIZED = "unauthorized"
