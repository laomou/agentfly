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
# 1. 添加 Provider（自动探测接口类型 + 拉取模型列表）
agentfly provider add deepseek --api-key '${DEEPSEEK_API_KEY}'

# 2. 测试所有模型（流式输出: 延迟 + TTFT + 吞吐）
agentfly test deepseek                              # Provider 所有模型
agentfly test deepseek deepseek-v4-pro              # 指定模型
agentfly test                                       # 全部 Provider
agentfly test deepseek -p 8 -t 15                   # 8 并发 + 15s 超时
agentfly test deepseek --refresh                    # 强制重探接口类型

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

- `deepseek` — https://api.deepseek.com
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
    base_url: https://api.deepseek.com
    models:
      deepseek-v4-pro: ''
      deepseek-v4-flash: ''
    default_model: deepseek-v4-pro

  my-proxy:
    name: custom
    api_key: ${MY_KEY}
    base_url: https://my-gateway.com
    models:
      claude-opus-4-6: ''       # 空 = 未探测, 首次 test 自动识别接口
      gpt-5.5: openai           # 已缓存接口类型, 后续直接走 openai
    default_model: claude-opus-4-6
```

`models` 是「模型名 → api_type」映射。Custom Provider 同时兼容 OpenAI
(`/v1/chat/completions`) 和 Anthropic (`/v1/messages`)：`agentfly test` 首次自动
探测每个模型走哪个接口 (anthropic 优先，`400/404` 回退 openai)，把结果写回
`api_type` 缓存，后续测试零额外探测。`--refresh` 清空缓存强制重探。

Agent 环境变量从 GitHub 远程配置自动获取，无需手动配置。
离线时使用包内默认配置降级。

## 安全

API Key 支持环境变量引用 `${VAR}`，禁止明文存储：

```bash
agentfly provider add deepseek --api-key '${DEEPSEEK_API_KEY}'
```
