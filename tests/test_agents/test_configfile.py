"""ScopedConfigFile 工具 + droid/pi 配置注入的测试.

重点验证「把伤害降到最低」：merge 不覆盖用户其它条目、写前备份、launch 结束
还原（新建的则删除）。用 monkeypatch 把 Path.home 指到 tmp，不碰真实家目录。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmswitch.agents.configfile import ScopedConfigFile, read_json
from lmswitch.agents.droid import Droid
from lmswitch.agents.pi import Pi
from lmswitch.models.schema import AgentConfig, ProviderConfig, ResolvedConfig
from lmswitch.models.types import AgentType, ProviderType


def _rc(name: AgentType, model: str = "deepseek-v4-pro") -> ResolvedConfig:
    return ResolvedConfig(
        agent=AgentConfig(name=name, provider="deepseek", model=model),
        provider=ProviderConfig(
            name=ProviderType.DEEPSEEK, api_key="sk-secret",
            endpoints={"openai": "https://api.deepseek.com"},
        ),
        effective_api_base="https://api.deepseek.com",
        effective_api_format="openai",
    )


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


class TestScopedConfigFile:
    def test_new_file_created_then_removed_on_restore(self, tmp_path):
        p = tmp_path / "sub" / "cfg.json"
        patch = ScopedConfigFile(p)
        patch.apply(lambda d: d.update({"x": 1}))
        assert read_json(p) == {"x": 1}
        patch.restore()
        assert not p.exists()  # 我们新建的 → 还原即删除

    def test_existing_file_merged_backed_up_then_restored(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"keep": "me"}), encoding="utf-8")

        patch = ScopedConfigFile(p)
        patch.apply(lambda d: d.update({"added": True}))

        merged = read_json(p)
        assert merged == {"keep": "me", "added": True}      # merge，不覆盖
        assert (tmp_path / "cfg.json.lmswitch.bak").exists()  # 写前备份

        patch.restore()
        assert read_json(p) == {"keep": "me"}                # 完整还原
        assert not (tmp_path / "cfg.json.lmswitch.bak").exists()  # 备份清掉

    def test_corrupt_file_treated_as_empty(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text("not json", encoding="utf-8")
        assert read_json(p) == {}


class TestDroidInjection:
    def test_writes_custom_model_and_restores(self, home):
        settings = home / ".factory" / "settings.json"
        # 用户已有一个自定义模型 + 其它字段，应被保留
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({
            "customModels": [{"id": "mine", "model": "x"}],
            "theme": "dark",
        }), encoding="utf-8")

        d = Droid()
        d.pre_launch(_rc(AgentType.DROID))

        data = read_json(settings)
        ours = data["customModels"][0]
        assert ours["id"] == "lmswitch"
        assert ours["baseUrl"] == "https://api.deepseek.com/v1"
        assert ours["apiKey"] == "sk-secret"
        assert ours["provider"] == "generic-chat-completion-api"
        assert data["sessionDefaultSettings"]["model"] == "lmswitch"
        # 用户原有的模型与字段保留
        assert {"id": "mine", "model": "x"} in data["customModels"]
        assert data["theme"] == "dark"

        d.post_launch()
        assert read_json(settings) == {
            "customModels": [{"id": "mine", "model": "x"}], "theme": "dark",
        }

    def test_no_plaintext_key_left_after_restore(self, home):
        settings = home / ".factory" / "settings.json"
        d = Droid()
        d.pre_launch(_rc(AgentType.DROID))
        assert "sk-secret" in settings.read_text()   # 会话期间在
        d.post_launch()
        assert not settings.exists()                 # 新建的 → 还原即删除，明文 key 不残留


class TestPiInjection:
    def test_writes_provider_and_launch_flags(self, home):
        p = Pi()
        rc = _rc(AgentType.PI)
        cmd = p.launch_command(rc)
        assert cmd == ["pi", "--provider", "lmswitch", "--model", "deepseek-v4-pro"]

        p.pre_launch(rc)
        models = home / ".pi" / "agent" / "models.json"
        prov = read_json(models)["providers"]["lmswitch"]
        assert prov["baseUrl"] == "https://api.deepseek.com/v1"
        assert prov["api"] == "openai-completions"
        assert prov["apiKey"] == "sk-secret"
        assert prov["models"] == [{"id": "deepseek-v4-pro"}]

        p.post_launch()
        assert not models.exists()
