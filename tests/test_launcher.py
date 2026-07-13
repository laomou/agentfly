"""AgentLauncher 测试 — adapter env + 实际 launch() 流程."""

from __future__ import annotations

import pytest

from agentfly.agents.base import Agent
from agentfly.agents.claude import Claude
from agentfly.core import launcher as launcher_mod
from agentfly.core.launcher import AgentLauncher, LaunchError
from agentfly.models.schema import AgentConfig, ProviderConfig, ResolvedConfig
from agentfly.models.types import AgentType, ProviderType


class TestAdapterEnvVars:
    """adapter 产出的环境变量（launcher 的输入）."""

    def test_claude_env(self, resolved_config):
        env = Claude().env_vars(resolved_config)
        assert "ANTHROPIC_AUTH_TOKEN" in env
        # 官方端点 fixture: 不覆盖 BASE_URL (保持 Claude Code 默认)
        assert "ANTHROPIC_BASE_URL" not in env


class _FakeAdapter(Agent):
    name = AgentType.CODEX
    display_name = "Fake"
    preferred_format = "openai"

    def env_vars(self, config):
        return {"BASE_ENV": "1"}

    def launch_command(self, config):
        return ["fakebin"]


class _FakeProvider:
    def env_for(self, agent_name):
        return {"PROVIDER_OVERRIDE": "yes"}


def _resolved():
    return ResolvedConfig(
        agent=AgentConfig(name=AgentType.CODEX, provider="openai", model="m1"),
        provider=ProviderConfig(type=ProviderType.OPENAI, api_key="sk-x",
                                endpoints={"openai": "http://x"}, models=["m1"]),
        effective_api_base="http://x", effective_api_format="openai",
    )


class TestLaunch:
    """AgentLauncher.launch 子进程流程（不起真进程）."""

    def test_missing_binary_raises(self, monkeypatch):
        monkeypatch.setattr(launcher_mod.shutil, "which", lambda c: None)
        with pytest.raises(LaunchError):
            AgentLauncher(_FakeAdapter()).launch(_resolved())

    def test_popen_env_merged_and_returncode(self, monkeypatch):
        captured = {}

        class FakePopen:
            def __init__(self, cmd, env=None, cwd=None, stdin=None, stdout=None, stderr=None):
                captured["cmd"] = cmd
                captured["env"] = env
                captured["cwd"] = cwd
                self.returncode = 7

            def wait(self):
                return 0

        monkeypatch.setattr(launcher_mod.shutil, "which", lambda c: "/usr/bin/fakebin")
        monkeypatch.setattr(launcher_mod, "get_provider", lambda pc: _FakeProvider())
        monkeypatch.setattr(launcher_mod.subprocess, "Popen", FakePopen)

        code = AgentLauncher(_FakeAdapter()).launch(_resolved(), cwd="/tmp")

        assert code == 7  # 透传 returncode
        assert captured["cmd"] == ["fakebin"]
        assert captured["cwd"] == "/tmp"
        assert captured["env"]["BASE_ENV"] == "1"          # adapter env
        assert captured["env"]["PROVIDER_OVERRIDE"] == "yes"  # provider 覆盖合并

    def test_popen_failure_raises_launch_error(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("nope")

        monkeypatch.setattr(launcher_mod.shutil, "which", lambda c: "/usr/bin/fakebin")
        monkeypatch.setattr(launcher_mod, "get_provider", lambda pc: _FakeProvider())
        monkeypatch.setattr(launcher_mod.subprocess, "Popen", boom)
        with pytest.raises(LaunchError):
            AgentLauncher(_FakeAdapter()).launch(_resolved())

    def test_post_launch_runs_even_on_popen_failure(self, monkeypatch):
        # 保证 ScopedConfigFile.restore 这类清理在启动失败时也会执行
        calls = []

        class _RestoringAdapter(_FakeAdapter):
            def post_launch(self):
                calls.append(1)

        def boom(*a, **k):
            raise OSError("nope")

        monkeypatch.setattr(launcher_mod.shutil, "which", lambda c: "/usr/bin/fakebin")
        monkeypatch.setattr(launcher_mod, "get_provider", lambda pc: _FakeProvider())
        monkeypatch.setattr(launcher_mod.subprocess, "Popen", boom)
        with pytest.raises(LaunchError):
            AgentLauncher(_RestoringAdapter()).launch(_resolved())
        assert calls == [1]
