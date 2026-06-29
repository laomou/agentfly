"""Claude (Anthropic 官方 CLI) Agent 适配器 — claude-code 的别名."""

from lmswitch.agents.claude_code import ClaudeCodeAdapter
from lmswitch.models.types import AgentType


class ClaudeAdapter(ClaudeCodeAdapter):
    """Claude — claude-code 的短别名."""

    name = AgentType.CLAUDE
    display_name = "Claude"
