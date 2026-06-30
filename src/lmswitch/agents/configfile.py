"""Agent 配置文件的「会话级补丁」工具.

部分 Agent (cline/droid/pi 等) 不读环境变量，只能从磁盘配置文件读取
provider/model。直接写用户配置文件有副作用：覆盖已有配置、明文 key 长期
落盘、launch 结束后仍残留。

ScopedConfigFile 把这类修改限制在本次 launch 内，做法参考 ollama
cmd/launch（merge 不覆盖 + 写前备份），并更进一步：post_launch 时还原
原始内容，等价于环境变量「退出即焚」的语义。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def read_json(path: Path) -> dict:
    """读取 JSON 配置，文件不存在/损坏时返回空 dict."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


class ScopedConfigFile:
    """对单个 JSON 配置文件做「会话级」修改.

    用法::

        patch = ScopedConfigFile(path)
        patch.apply(lambda d: mutate(d))   # 备份原文件 + merge 写入
        ... 运行 agent ...
        patch.restore()                    # 还原成原样（新建的则删除）
    """

    _BAK_SUFFIX = ".lmswitch.bak"

    def __init__(self, path: Path):
        self.path = path
        self._original: bytes | None = path.read_bytes() if path.exists() else None

    @property
    def _backup_path(self) -> Path:
        return self.path.with_name(self.path.name + self._BAK_SUFFIX)

    def apply(self, mutate: Callable[[dict], None]) -> None:
        """读出现有内容 → 调 mutate 就地修改 → 备份原文件并原子写回."""
        data = read_json(self.path)
        mutate(data)
        if self._original is not None:
            self._backup_path.write_bytes(self._original)
        _write_atomic(self.path, data)

    def restore(self) -> None:
        """还原到 apply 之前：原样写回，或删除我们新建的文件；并清掉备份."""
        if self._original is not None:
            self.path.write_bytes(self._original)
        elif self.path.exists():
            self.path.unlink()
        if self._backup_path.exists():
            self._backup_path.unlink()
