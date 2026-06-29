"""统一配置文件的加载与保存."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from lmswitch.models.schema import UnifiedConfig

# XDG 规范：默认配置路径
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "lmswitch"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"


def _default_config_dir() -> Path:
    """获取配置目录，优先使用 XDG_CONFIG_HOME."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "lmswitch"
    return DEFAULT_CONFIG_DIR


def config_path(explicit: Optional[str | Path] = None) -> Path:
    """返回配置文件路径.

    优先级:
    1. 显式传入的路径
    2. 环境变量 LMSWITCH_CONFIG
    3. XDG_CONFIG_HOME/lmswitch/config.yaml
    4. ~/.config/lmswitch/config.yaml
    """
    if explicit:
        return Path(explicit)
    env = os.environ.get("LMSWITCH_CONFIG")
    if env:
        return Path(env)
    return _default_config_dir() / "config.yaml"


def load_config(path: Optional[str | Path] = None) -> UnifiedConfig:
    """加载统一配置.

    Args:
        path: 配置文件路径，为 None 时使用默认路径.

    Returns:
        UnifiedConfig 实例.

    Raises:
        FileNotFoundError: 配置文件不存在.
    """
    p = config_path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {p}\n"
            f"请运行 'lmswitch provider add' 初始化配置"
        )

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return UnifiedConfig.model_validate(data)


def save_config(config: UnifiedConfig, path: Optional[str | Path] = None) -> Path:
    """保存统一配置到文件.

    Args:
        config: UnifiedConfig 实例.
        path: 目标路径，为 None 时使用默认路径.

    Returns:
        实际写入的文件路径.
    """
    p = config_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # mode="json" 确保 Enum 序列化为字符串而非 Python 对象引用
    # exclude_defaults 去掉 null / [] / {} 等默认值
    data = config.model_dump(mode="json", exclude_defaults=True)
    content = yaml.dump(
        data,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )

    # 写入临时文件后原子替换
    tmp = p.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)

    return p


def ensure_config_exists(path: Optional[str | Path] = None) -> tuple[UnifiedConfig, Path]:
    """确保配置文件存在，不存在则创建默认空配置.

    Returns:
        (config, path) 元组.
    """
    p = config_path(path)
    if p.exists():
        return load_config(p), p

    config = UnifiedConfig()
    save_config(config, p)
    return config, p
