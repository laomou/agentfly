# AgentFly — AI Agent 客制化配置中心

统一管理多 AI Agent 配置，一键启动。

```bash
pip install agentfly

# 一键安装 Tab 补全
agentfly completion --install
source ~/.bashrc
```

## 快速开始

```bash
# 1. 添加 Provider（自动探测端点 + 拉取模型列表）
agentfly provider add deepseek --api-key '${DEEPSEEK_API_KEY}'

# 2. 测试所有模型（流式输出: 延迟 + TTFT + 吞吐）
agentfly test deepseek                              # Provider 所有模型
agentfly test deepseek deepseek-v4-pro              # 指定模型
agentfly test                                       # 全部 Provider

# 3. 启动 AI Agent
agentfly launch claude                              # 默认配置
agentfly launch claude --provider deepseek          # 指定 Provider
agentfly launch claude --model deepseek-v4-pro      # 指定模型
agentfly launch claude --provider deepseek -- --continue  # 透传 Agent 参数
```

## 命令

| 命令 | 作用 |
|------|------|
| `agentfly launch` | 启动 AI Agent，自动注入环境变量 |
| `agentfly test` | 测试模型延迟 / TTFT / 吞吐 |
| `agentfly provider` | 管理服务提供商 (add/list/show/reload/remove) |
| `agentfly doctor` | 配置健康检查 |
| `agentfly completion` | 安装命令补全 (`--install` 一键安装) |

## 支持的 Agent

`claude` · `cline` · `codex` · `droid` · `opencode` · `openclaw` · `pi`

## 支持的 Provider

内置厂商（自动填 BASE_URL）:

- `deepseek` — https://api.deepseek.com (openai + anthropic 双格式)
- `anthropic` — https://api.anthropic.com
- `openai` — https://api.openai.com

任意 OpenAI / Anthropic 兼容端点:

```bash
agentfly provider add --api-base http://10.0.0.1:3000 --api-key '${MY_KEY}'
agentfly provider add my-proxy --api-base http://10.0.0.1:3000 --models "model-a,model-b"
```

## 配置

```yaml
# ~/.config/agentfly/config.yaml
providers:
  deepseek:
    name: deepseek
    api_key: ${DEEPSEEK_API_KEY}
    endpoints:
      openai: https://api.deepseek.com
      anthropic: https://api.deepseek.com
    models: [deepseek-v4-pro, deepseek-v4-flash]
    default_model: deepseek-v4-pro
```

Agent 环境变量从 GitHub 远程配置自动获取，无需手动配置。
离线时使用包内默认配置降级。

## 安全

API Key 支持环境变量引用 `${VAR}`，禁止明文存储：

```bash
agentfly provider add deepseek --api-key '${DEEPSEEK_API_KEY}'
```
