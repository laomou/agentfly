"""配置加载/保存测试."""

from __future__ import annotations

import pytest

from lmswitch.core.config import (
    config_path,
    ensure_config_exists,
    load_config,
    save_config,
)
from lmswitch.models.schema import UnifiedConfig


class TestConfigPath:
    """配置路径解析测试."""

    def test_default_path(self):
        path = config_path()
        assert str(path).endswith(".config/lmswitch/config.yaml")

    def test_explicit_path(self):
        path = config_path("/tmp/my-config.yaml")
        assert path == path  # Path comparison

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LMSWITCH_CONFIG", "/tmp/env-config.yaml")
        path = config_path()
        assert str(path) == "/tmp/env-config.yaml"

    def test_xdg_config_home(self, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/xdg")
        path = config_path()
        assert str(path) == "/custom/xdg/lmswitch/config.yaml"


class TestLoadSave:
    """配置加载保存测试."""

    def test_save_and_load(self, temp_config_file):
        config = load_config(temp_config_file)
        assert config.version == "1"
        assert "anthropic" in config.providers

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_config("/tmp/nonexistent-lmswitch-config.yaml")

    def test_save_creates_dir(self, tmp_path):
        cfg_path = tmp_path / "sub" / "dir" / "config.yaml"
        config = UnifiedConfig()
        save_config(config, cfg_path)
        assert cfg_path.exists()


class TestEnsureConfig:
    """ensure_config_exists 测试."""

    def test_creates_new(self, tmp_path):
        cfg_path = tmp_path / "new-config.yaml"
        config, path = ensure_config_exists(cfg_path)
        assert config.version == "1"
        assert path.exists()

    def test_loads_existing(self, temp_config_file):
        config, path = ensure_config_exists(temp_config_file)
        assert "anthropic" in config.providers
