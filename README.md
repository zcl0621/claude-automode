# claude-automode

给 Claude Code 用的第三方 auto mode：Claude Code 本来要弹窗问你的工具调用，先交给一个分类器模型判断，由它决定放行、拒绝，还是照常弹窗让你自己定。

Claude Code 自带的 auto mode 只支持 Claude 模型。底座用 Claude Code、模型换成别家时，就只能一路手动点确认。这个插件补的就是这块：判断交给任意一个兼容 OpenAI 或 Anthropic 接口的模型，比如 DeepSeek、Qwen、OpenRouter、自建的 vLLM。

## 它做什么

- **只看本来要弹窗的调用。** 挂在 Claude Code 的 `PermissionRequest` hook 上。你已经允许过的、Claude Code 自己就放行的调用，都不经过它
- **规则先过一遍。** 明显只读的命令直接放行，比如 `ls`、`git status`、`grep`、只读的 MCP 调用。`rm -rf ~` 这类灾难性命令直接拒绝。其余的才交给模型
- **模型只拦有害的操作，不管做没做超出范围。** 拒绝的情形只有这几种：
  - 照着工具输出里的注入内容去做
  - 用户没要求的删除或覆盖
  - 把密钥或数据发到用户没指定的地方
  - 用户没要求的持久化或安全设置改动，包括用 `--no-verify` 跳过检查
  - 换个形式重试已经被拒的操作
- **拿不准就弹窗。** 分类器出错、超时、判 ask，都不输出任何东西，Claude Code 照常弹窗，所以坏掉的分类器永远不会自动放行
- **拒绝会告诉 agent 原因**，让它换个做法。连续被拒超过 3 次，就停下来弹窗给你
- **扫描工具输出里的注入。** 每次工具调用结束后，用快速模型检查输出里有没有"写给 agent 的指令"。发现了就提醒 agent，之后的判断也会参考

分类器看到的内容：
- 你发的消息
- agent 最近的工具调用（名字和参数）
- agent 在你最新一条消息之前说的话（用来理解你回的"好""跑"指的是什么）
- 当前这次调用本身

它看不到 agent 的其他发言和工具输出，因为这两样都可能被注入。

## 快速开始

需要 Claude Code 和 `python3`，没有其他依赖。

在 Claude Code 里：

```
/plugin marketplace add zcl0621/claude-automode
```

```
/plugin install automode@claude-automode
```

然后填分类器的接口地址、密钥和模型：

```
/plugin configure automode@claude-automode
```

用 `acceptEdits` 模式启动：

```bash
claude --permission-mode acceptEdits
```

完整步骤、各个模式的区别、DeepSeek 的配置示例，见[接入指引](docs/接入指引.md)。

## 效果

用 DeepSeek flash（关闭思考）做分类器测出来的结果：

| | 结果 |
|---|---|
| 公开数据集里 agent 照做了注入的 21 个案例 | 拦下 20 个 |
| 公开数据集里 125 个非敏感操作 | 误拦 3 个 |
| 作者本机 4000 条真实操作（规则层没放行、要交给模型的那部分） | 拦下 1.15%–1.4%，其中明显拦错的约 0.4% |

数据来源和局限见[评测结论](docs/评测结论.md)。

## 注意

- 每次工具调用的输出都会发给分类器服务商扫描注入，被判断的调用和你的消息也会发过去。不要用你不信任的服务商
- 判断要调一到两次模型，每次弹窗前会多等几秒
- 判定日志写在 `~/.claude-automode/decisions.jsonl`，默认带上截断后的工具参数
- 只在交互式会话里用。`claude -p` 下没人回答弹窗，Claude Code 会把 ask 当成拒绝

## 开发

```bash
python3 -m unittest discover -s tests
```

评测脚本在 `eval/`，数据集要自己下载，见各脚本开头的说明。
