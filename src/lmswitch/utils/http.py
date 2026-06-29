"""HTTP 工具 — 创建预配置的 httpx Client."""

from __future__ import annotations

import httpx


def create_client(timeout: float = 15.0) -> httpx.Client:
    """创建预配置的 HTTP 客户端.

    Args:
        timeout: 请求超时秒数.

    Returns:
        配置好的 httpx.Client.
    """
    return httpx.Client(
        timeout=httpx.Timeout(timeout),
        headers={"User-Agent": "lmswitch/0.1.0"},
    )
