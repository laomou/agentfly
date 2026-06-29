"""Claude (Anthropic 官方 CLI) Agent 适配器 — claude-code 的别名."""

from lmswitch.agents.claude_code import ClaudeCode
from lmswitch.models.types import AgentType


class Claude(ClaudeCode):
    """Claude — claude-code 的短别名."""

    name = AgentType.CLAUDE
    display_name = "Claude"
