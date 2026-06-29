# LMSwitch — AI Agent 客制化配置中心

一键切换 AI Coding Agent，统一管理 18+ 服务提供商配置。

```bash
pip install lmswitch
```

## 为什么需要 LMSwitch？

每个 AI Agent（Claude Code、Codex、Cline 等）使用不同的环境变量和配置格式。
LMSwitch 让你用一套统一配置管理所有 Agent 和 Provider，无需关心底层差异。

## 快速开始

```bash
# 1. 添加 Provider（自动探测 API 格式 + 拉取模型）
lmswitch provider add deepseek --api-key '${DEEPSEEK_API_KEY}'

# 2. 测试模型（stream 模式: 延迟 + TTFT + 吞吐）
lmswitch test deepseek

# 3. 启动 Agent（自动注入环境变量）
lmswitch launch codex --provider deepseek
```

## 三个核心命令

| 命令 | 作用 |
|------|------|
| `lmswitch launch` | 启动 AI Agent，自动注入配置 |
| `lmswitch test` | 测试模型延迟 / TTFT / 吞吐 |
| `lmswitch provider` | 管理服务提供商 (add/rm/list/models) |

## 支持的 Agent

`claude` · `claude-code` · `cline` · `codex` · `droid` · `opencode` · `openclaw` · `pi`

## 支持的 Provider

`openai` · `anthropic` · `deepseek` · `google` · `moonshot` · `zhipu` · `qwen` · `siliconflow` · `together` · `groq` · `openrouter` · `perplexity` · `cerebras` · `mistral` · `xai` · `minimax` · `fireworks` + 任意 OpenAI/Anthropic 兼容端点

## 配置示例

```yaml
# ~/.config/lmswitch/config.yaml
providers:
  deepseek:
    endpoints:
      openai: https://api.deepseek.com
      anthropic: https://api.deepseek.com/anthropic
    models: [deepseek-chat, deepseek-reasoner]
    default_model: deepseek-chat
```

## 安全

API Key 自动转为环境变量引用，禁止明文存储。

```bash
lmswitch provider add --api-base http://10.0.0.1:3000 --api-key '${MY_KEY}'
```

## License

MIT
