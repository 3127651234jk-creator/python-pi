# pi-python

pi（TypeScript 编码智能体）核心架构的 Python 移植版。目标不是 1:1 复刻整个 monorepo，而是用 Python 忠实还原 pi 的三层核心：Provider 流式协议、agent loop、以及四个编码工具，可直接运行。

## 与 TS 版的对应关系

| Python 模块 | TS 源 | 内容 |
| --- | --- | --- |
| `pi.ai.types` | `packages/ai/src/types.ts` | 四角色消息模型（system/user/assistant/toolResult）、内容块、Usage、事件协议 |
| `pi.ai.providers` | `packages/ai/src/api/*`、`providers/*` | `anthropic-messages` 与 `openai-completions`（覆盖所有 OpenAI 兼容端点）的 SSE 流式适配 + `faux` 脚本化测试 provider |
| `pi.ai.utils` | `packages/ai/src/utils` | `normalize_context`（系统消息重放折叠）、`get_tool_state_changes`（工具集增量声明）、JSON Schema 参数校验 |
| `pi.agent.agent_loop` | `packages/agent/src/agent-loop.ts` | 主循环：steering/follow-up 队列、顺序/并行工具执行、`beforeToolCall`/`afterToolCall` 钩子、截断消息失败、错误/中止处理 |
| `pi.agent.agent` | `packages/agent/src/agent.ts` | 有状态 Agent：transcript、队列、事件订阅、prompt/abort |
| `pi.coding_agent.tools` | `coding-agent/src/core/tools/*` | `read`（截断/offset/limit/图片）、`write`（父目录自动创建）、`edit`（唯一匹配 + 重叠校验 + CRLF/BOM 保留 + 模糊匹配）、`bash`（流式输出、尾截断、超时、进程树终止） |
| `pi.coding_agent.system_prompt` | `coding-agent/src/core/system-prompt.ts` | 分节系统提示词 + AGENTS.md 项目上下文 |
| `pi.coding_agent.session` | `coding-agent/src/core/session-manager.ts` | JSONL 会话树、分支、切换、命名、导出、`--continue` 恢复 |
| `pi.coding_agent.cli` | `coding-agent/src/cli` | 单次模式 + 交互 REPL + 斜杠命令 + Ctrl+C 中止 |
| `pi.evals.harness` | `packages/evals/src/harness.ts` | 隔离评测 harness：临时工作区、凭据注入、会话录制、用量统计、POSIX 工具沙箱、文档变体 |
| `pi.evals.plan` | `packages/evals/src/plan.ts` | 发现用例解析与 (case, variant, run) 任务计划 |
| `pi.evals.report` | `packages/evals/src/report.ts` | 观察读取、成对配对、lift/标记/运营总量与文本报告 |
| `pi.evals.docker` | `packages/evals/src/docker.ts` | 变体镜像构建、auth.json 校验、容器发现与单任务运行 |
| `pi.evals.cli` | `packages/evals/src/cli.ts` | 评测编排 CLI：参数解析、发现对齐、协议落盘、报告产出 |
| `pi.evals.framework` | `vitest-evals`（npm 依赖） | 本地替代：Harness 适配器、运行规范化、结构化输出裁判、用例发现与运行器 |

## 安装

```bash
cd pi-python
pip install -e .        # 或直接 python -m pi（依赖 httpx）
```

要求 Python 3.9+，依赖 `httpx`、`PyYAML`、`prompt_toolkit` 和 `wcwidth`。

## 运行

```bash
# OpenAI 兼容端点（OpenAI/DeepSeek/Groq/OpenRouter/vLLM/llama.cpp...）
set OPENAI_API_KEY=sk-...
set OPENAI_BASE_URL=https://api.openai.com/v1   # 可选
python -m pi "list the python files in this project"

# Anthropic
set ANTHROPIC_API_KEY=sk-ant-...
python -m pi --provider anthropic "explain pyproject.toml"

# 离线/测试：脚本化假模型
python -m pi --provider faux "hello"
```

常用参数：`--provider anthropic|openai|faux`、`--model <id>`、`--base-url`、`--api-key`、`--thinking off|low|medium|high`、`-c` 继续上次会话、`--no-session` 不落盘。

CLI 还内置 `openrouter`，并可从 `models.json` 注册其他兼容 provider。

环境变量也可用 `PI_PY_PROVIDER` / `PI_PY_MODEL` / `PI_PY_API_KEY` / `PI_PY_BASE_URL`。

交互模式（不带 prompt 参数）支持斜杠命令：

```text
/model <id>   切换模型
/clear        清空对话（保留系统提示词）
/new          新建会话
/resume       列出会话（显示 ID、名称和消息数）
/resume <id>  切换会话，也可提供完整文件路径
/session      查看当前会话路径、名称、活动节点和消息数
/name <name>  命名当前会话
/tree         显示全部分支与消息 ID，* 标记活动节点
/tree <id>    回到指定消息之后；下一次输入创建新分支
/fork <id>    将指定消息及其祖先复制为独立会话
/clone        将当前活动分支复制为独立会话
/export <path.jsonl|path.html|path.md>  导出当前活动分支
/usage        说明
/help /exit
Ctrl+C        中止当前运行
```

## 原生模型协议

除原有 Chat Completions 和 Anthropic Messages 外，已接入三个专用 API：

| provider | API | 认证 |
| --- | --- | --- |
| `openai-responses` | `openai-responses` | `OPENAI_API_KEY`、`--api-key` 或该 provider 的已保存凭据 |
| `google` | `google-generative-ai` | `GEMINI_API_KEY`、`--api-key` 或已保存凭据 |
| `amazon-bedrock` | `bedrock-converse-stream` | boto3 的 AWS 凭据链，支持环境变量、profile 和角色凭据 |

```bash
python -m pi --provider openai-responses --model gpt-4o-mini "hello"
python -m pi --provider google --model gemini-2.5-flash "hello"
pip install -e ".[bedrock]"
python -m pi --provider amazon-bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0 "hello"
```

示例型号用于展示参数，不保证账号、地区或上游当前开放访问。Bedrock 使用 `AWS_REGION` / `AWS_DEFAULT_REGION`，未设置时默认为 `us-east-1`；`--base-url` 可覆盖端点。AWS 密钥不写入 Pi 的 `auth.json`。Bedrock 的可选 boto3 版本固定为兼容 Python 3.9 的 `1.40.75`。

三种适配均支持流式文本、工具调用、输入图片、工具结果、停止原因及用量映射。Responses 支持 reasoning summary 和加密 reasoning 条目回放；Gemini 保存工具调用的 thought signature，以便恢复会话后回传。Bedrock 支持 ConverseStream reasoning 内容回放，Claude 模型配置 `reasoning=true` 后可使用 thinking budget；其他模型专用推理参数尚未统一。

`models.json` 的 `api` 现在还可指定上述三个新 API，适用于自定义端点；内置 `openai` 默认协议仍为 Chat Completions。未知的原生 provider 模型 ID 可以使用，但需要在 `models.json` 中补准确的上下文窗口、输出限制和能力参数。

实现参考 [OpenAI Responses 流式事件](https://developers.openai.com/api/reference/resources/responses/streaming-events)、[Google generateContent](https://ai.google.dev/api/generate-content) 和 [AWS ConverseStream](https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock-runtime/client/converse_stream.html)。验证使用模拟 SSE 和 AWS 事件流，未执行真实付费请求。当前不包含 Vertex AI、Azure 专用认证、Responses 内置托管工具、音视频输入或自动模型目录同步。

## 终端界面

在真实终端中、不带 prompt 参数启动时，默认使用增强编辑器。`--tui on` 强制启用，`--tui off` 使用原始输入循环；管道输入默认关闭增强界面。单次执行、JSON 和 RPC 模式保持原输出协议。

编辑器支持多行输入、中文、粘贴、当前进程内输入历史、命令和文件路径补全；流式输出显示在编辑器上方。底部显示当前 provider/model、thinking 和运行状态。运行时仍可输入消息，提交后进入 steering 队列。

| 默认快捷键 | 行为 |
| --- | --- |
| Enter | 提交输入 |
| Alt+Enter | 插入换行；终端也可先按 Escape 再按 Enter |
| Tab | 打开补全，继续按 Tab 切换候选 |
| Ctrl+C | 空闲时清空输入，运行时请求中止 |
| Ctrl+L | 打开模型选择器 |
| Ctrl+O | 打开会话选择器 |
| Ctrl+D | 空闲时退出，运行时请求中止 |

`/model`、`/resume`、`/tree` 不带参数时打开模型、会话或会话节点选择器。选择器用方向键移动，Tab 切换确认/取消按钮，Enter 确认。带参数命令仍可直接使用。`/theme` 打开主题选择器，也可输入 `/theme light`；支持 `dark`、`light`、`mono`，保存到用户设置。`/keys` 显示有效快捷键。

用户/项目 `settings.json` 支持：

```json
{
  "theme": "dark",
  "keybindings": {
    "submit": ["c-s"],
    "newline": ["escape enter"],
    "abort": ["c-c"],
    "models": ["c-l"],
    "sessions": ["c-o"],
    "exit": ["c-d"],
    "complete": ["tab"]
  }
}
```

快捷键使用 prompt_toolkit 的键名，空格分隔组合序列；每项覆盖对应默认值，重启后生效。重复绑定会报错。当前界面使用终端滚动记录配合多行编辑器和模态选择器，尚未实现 TS 版的全屏固定对话区、Markdown 富文本、图片内联显示或工具输出折叠。已用受控输入验证编辑、选择器和中止操作，尚未完成所有实际终端的人工验证。

## Python 扩展

扩展从项目 `.pi/extensions/` 和用户目录 `extensions/` 自动加载，支持直接 `.py` 文件或包含 `__init__.py` 的目录包。也可在用户/项目 `settings.json` 中设置 `extensions` 路径数组，相对路径以设置文件目录为基准；`include_default_extensions: false` 关闭默认目录发现。

每个扩展定义同步或异步 `setup(api)`。例如 `.pi/extensions/greeting.py`：

```python
from pi.agent.types import AgentTool, AgentToolResult, BeforeToolCallResult
from pi.ai.types import TextContent

def setup(api):
    api.register_command("greeting", lambda args, ctx: "Greet " + (args or "world"), "Generate a greeting")

    async def echo(call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=params["text"])])

    api.register_tool(AgentTool(
        name="echo", description="Repeat text",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        execute=echo,
    ))

    def guard(event, ctx):
        if event.tool_name == "write":
            return BeforeToolCallResult(block=True, reason="Writes disabled by extension")
    api.on("tool_call", guard)
    api.on("session_start", lambda event, ctx: api.log("started"))
    api.on("session_shutdown", lambda event, ctx: api.log("stopped"))
```

`/greeting Alice` 返回的字符串会作为 prompt 发送给模型；返回 `None` 表示命令已处理、不启动模型。扩展命令优先于同名模板，不能覆盖内置命令。注册只允许在 `setup` 内完成，重名工具/命令或失败 factory 的注册整体丢弃，并给出 stderr 诊断。

事件回调接受 `(event, ctx)`；`ctx` 提供 `agent`、`cwd` 和可选 `session`。支持 `session_start`、`session_shutdown`，以及 Agent 的 `agent_start/end`、`turn_start/end`、`message_start/update/end`、`tool_execution_start/update/end` 通知。`api.on` 返回取消订阅函数。通知异常仅记录诊断；回调按加载与注册顺序执行。

`input` 回调接受 `{"text": ...}`，可返回 `pi.coding_agent.extensions.InputResult(text=...)` 修改后续输入，或 `InputResult(handled=True)` 截止处理。多个输入变换依次组合。`tool_call` 可返回 `BeforeToolCallResult` 阻止工具执行；处理器异常也会使该工具调用失败。`tool_result` 可返回 `AfterToolCallResult` 修改输出，多次返回以最后一个为准。普通通知的返回值忽略。输入/命令处理适用于普通 prompt；RPC 的 steer/follow_up 保持原队列行为。

`/extensions` 查看来源，`/help` 列出扩展命令。`/reload` 执行旧扩展的 `session_shutdown`、撤销工具与事件订阅，然后重新执行 factory 并发送 `session_start`。退出同样执行清理。长期资源应在 `session_start` 创建，在幂等的 `session_shutdown` 中关闭，不应在 factory 中启动后台任务。

扩展同时作用于文本、JSON 和 RPC 模式。RPC `get_commands` 列出扩展命令，空闲时可调用 `reload_extensions`；通过 `prompt` 发送 `/命令`，纯本地处理返回 `disposition=handled`，此时不产生新的 `agent_settled`。

扩展是本机 Python 代码，以 Pi 进程权限运行，没有沙箱或项目授权界面。扩展 factory、回调和注册工具内的普通 `print` 重定向到 stderr，推荐使用 `api.log`；自行创建的后台任务仍须自行管理输出和生命周期。TS/JS 扩展可通过 Node 桥接（`pi.coding_agent.ts_extension`，需 PATH 中的 Node 22.6+）加载：`registerTool/registerCommand/registerProvider/registerKeybinding/on/ui/appendEntry/sendMessage/log` 跨进程桥接，其余 TS 专用 API（TUI 渲染器、虚拟模型、MCP 注册、会话管理、`pi.events`、状态 getter 等）在调用时抛出明确错误而非静默缺失。Python `ExtensionAPI` 本身不包含 TUI 渲染器或 npm/git 包安装。

## JSON 输出与 RPC

`python -m pi --mode json --provider faux --no-session "hello"` 输出 JSONL 事件。每行是一个 JSON 对象，包含 Agent、消息和工具执行事件；结束时输出 `agent_settled`。诊断写入 stderr。JSON 模式要求提供 prompt，不进入交互输入循环。

`python -m pi --mode rpc --provider faux --no-session` 启动长期运行的 stdio RPC 服务。通过 stdin 发送 LF 分隔的 JSON 命令，从 stdout 持续读取响应和事件。例如：

```json
{"id":"1","type":"get_state"}
{"id":"2","type":"prompt","message":"hello"}
{"id":"3","type":"abort"}
```

响应格式为 `{"type":"response","id":"1","command":"get_state","success":true,"data":{...}}`。失败时 `success=false` 并携带 `error`；非法 JSON 返回 `command=parse`，服务继续处理下一行。`prompt` 响应只表示接受请求，完成要等待 `agent_settled`。运行中可读取状态、发送 `steer` / `follow_up` 或 `abort`；第二个普通 prompt 会被拒绝。关闭 stdin 请求退出，服务中止活动运行并清理 MCP 连接。

当前支持的命令：`prompt`、`steer`、`follow_up`、`abort`、`get_state`、`get_messages`、`get_tools`、`get_available_models`、`set_model`、`set_thinking_level`、`new_session`、`switch_session`、`fork`、`clone`、`get_tree`、`get_entries`、`set_session_name`、`export`。会话命令要求启用持久化；修改模型和会话要求 Agent 空闲。参数分别使用 `message`、`model`/`provider`、`level`、`path`、`entry_id` 或 `name`。`export` 根据路径扩展名导出 JSONL、HTML 或 Markdown。

事件和消息字段采用 Python 的 snake_case，例如 `assistant_message_event`、`tool_call_id`。这是 Python 版协议的核心子集，不保证与 TS 全部命令及 camelCase 字段兼容，也不提供网络监听服务或扩展 UI 协议。

异步客户端封装：

```python
from pi.coding_agent.rpc_client import RpcClient

async def example():
    client = await RpcClient().start(["--provider", "faux", "--no-session"])
    try:
        events = await client.prompt_and_wait("hello")
        state = await client.request("get_state")
    finally:
        await client.close()
```

客户端按 ID 对应并发命令响应；`events` 队列接收事件。`prompt_and_wait` 应由单一消费者调用，避免其他消费者抢走完成事件。调用超时不会自动取消远程任务，需要显式发送 `abort`。stdout 必须持续读取，以免阻塞进程；stderr 保留为诊断通道。RPC 启动参数不能带 prompt、登录或模型列表命令。

## MCP 服务

Python CLI 自动读取用户目录的 `mcp.json` 和项目 `.pi/mcp.json`。项目同名字段覆盖用户字段，可用 `enabled: false` 禁用服务。

```json
{
  "mcpServers": {
    "local-tools": {
      "command": "python",
      "args": ["server.py"],
      "timeout": 60
    },
    "remote-tools": {
      "url": "https://example.com/mcp",
      "headers": {"Authorization": "Bearer ${MCP_TOKEN}"}
    }
  }
}
```

支持 stdio 和 Streamable HTTP 的 JSON/SSE 响应；协议参考 [MCP transports](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)。stdio 的 `command` 是可执行文件，`args` 是参数列表，不通过 shell 拼接。可设置 `env`、相对工作目录的 `cwd` 和请求超时 `timeout`。`env`、`headers` 支持环境变量插值，不执行凭据命令。

服务工具以 `mcp__服务__工具_哈希` 名称直接注册给模型，支持工具列表分页、JSON Schema 参数、文本/图片、`structuredContent` 和 `isError`。配置或连接失败会输出诊断，其他服务仍可使用。`/mcp` 显示状态，`/mcp reconnect` 或 `/reload` 重连并重新发现工具；退出时关闭连接并终止本地进程树。工具调用不自动重试，避免重复副作用。

当前为核心工具接入：尚不包含 MCP OAuth、资源列表/读取、codemode/deferred exposure、后台工具变更通知或自动断线重连。HTTP 服务可通过 headers 配置现有 Bearer token。本地配置中的服务会在启动时执行，不包含 TS 版的项目信任交互。

## Skills 与提示词模板

启动时自动扫描以下位置：

| 资源 | 默认位置 |
| --- | --- |
| Skills | 项目 `.pi/skills/`、用户目录 `skills/`、`~/.agents/skills/`，以及工作目录向上到 Git 仓库根目录的 `.agents/skills/` |
| 提示词模板 | 项目 `.pi/prompts/`、用户目录 `prompts/` 的直接 `.md` 子文件 |

用户目录仍由 `--agent-dir` / `PI_PY_AGENT_DIR` 决定，默认 `~/.pi-python`。Skill 目录递归查找 `SKILL.md`，不递归跟随目录符号链接。没有 Git 根目录时，祖先发现到文件系统根目录为止。发现过程不执行 skill 脚本。

示例 `.pi/skills/review/SKILL.md`：

```markdown
---
name: review
description: Review code changes for correctness and missing tests.
---
Read the changed files in full. Identify regressions and missing tests.
Resolve supporting files relative to this skill directory.
```

系统提示词只包含 skill 名称、描述和文件路径，模型按需通过 `read` 加载完整指令；是否自动选中取决于模型。输入 `/skill:review focus on concurrency` 会明确加载完整指令，并附上用户请求。`disable-model-invocation: true` 隐藏自动发现提示，但保留显式命令。

示例 `.pi/prompts/review.md`：

```markdown
---
description: Review code changes
argument-hint: "[focus]"
---
Review the changes. Focus on ${1:-correctness and tests}.
```

输入 `/review "API compatibility"` 会将模板展开为用户消息。支持 `$1`、`$2`、`$@`、`$ARGUMENTS`、`${1:-default}`、`${@:-default}`、`${ARGUMENTS:-default}`、`${@:N}` 和 `${@:N:L}`。参数支持单双引号分组，反斜杠保留为普通字符；未闭合引号报错。替换只进行一次，不递归展开参数中的 `$` 占位符，也不执行 shell 命令。单次模式同样支持，例如 `python -m pi --provider faux "/review tests"`。

`/skills`、`/prompts` 列出资源与来源；`/help` 列出动态命令；`/reload` 重新发现资源，并更新当前会话中的 skill 目录。恢复旧会话时也会刷新目录。模板内容修改后需要 `/reload`；显式 skill 调用每次读取文件。

用户或项目 `settings.json` 可以添加 `skills`、`prompts` 路径数组，接受目录或单独 Markdown 文件；相对路径以该设置文件所在目录为基准。设置 `include_default_resources: false` 可关闭默认目录发现，仅加载显式路径。`enable_skill_commands: false` 只隐藏 `/help` 中的 skill 命令，手工输入仍可调用。

同名资源保留首次发现的条目并输出警告。优先次序为显式项目路径、显式用户路径、项目默认目录、用户默认目录及 Agent Skills 目录。无效 YAML、缺少描述的 skill、空正文和模板占用内置命令等情况会跳过并输出诊断。当前不包含 npm/git 资源包安装或 TUI 补全。

## 内置搜索与 PowerShell 工具

默认工具集现为 `read`、`bash`、`edit`、`write`、`grep`、`find`、`ls`、`powershell`，CLI 会将实际注册的工具加入系统提示词。

| 工具 | 参数与行为 |
| --- | --- |
| `ls` | 可选 `path`、`limit`（默认 500）；包含隐藏文件，按名称排序，目录带 `/` |
| `find` | 必填 glob `pattern`；可选 `path`、`limit`（默认 1000）；返回相对路径，只查找文件 |
| `grep` | 必填 `pattern`；可选 `path`、`glob`、`ignoreCase`、`literal`、`context`、`limit`（默认 100 个匹配行）；返回路径、行号与文本 |
| `powershell` | 必填 `command`；可选 `timeout`（秒）；流式输出、尾部截断、超时与中止时终止进程树 |

`grep`/`find` 需要 PATH 中的 `rg`（ripgrep），遵循其忽略规则，包含隐藏文件，排除 `.git` 和 `node_modules`；不跟随目录符号链接。搜索结果上限为 50KB，超出时返回提示。`grep` 的长行使用 ripgrep 的 500 列预览；默认是正则，精确文本请用 `literal=true`。`find` 支持 `*.py`、`src/**/*.py` 等模式。工具使用参数列表启动 rg，不通过 shell 拼接查询。

`powershell` 优先使用 `pwsh`，其次使用 `powershell`，缺失时返回明确错误。通过 UTF-16LE 编码命令传递脚本，输出设为 UTF-8。原有 `bash` 的平台行为保持：Windows 使用 PowerShell，其他平台使用 bash。PowerShell 和 bash 的文本输出保留最后 2000 行或 50KB，完整溢出日志保存到临时文件。

## 配置、认证与模型

用户目录默认为 `~/.pi-python`，可通过 `--agent-dir` 或 `PI_PY_AGENT_DIR` 更改：

| 文件 | 内容 |
| --- | --- |
| `settings.json` | 默认 provider、model、thinking、session_dir、compaction、cache_retention |
| `models.json` | 自定义 provider、模型元数据与覆盖 |
| `auth.json` | API key、OAuth access/refresh token 和过期时间 |

项目 `.pi/settings.json` 覆盖用户设置，对象字段递归合并。模型选择、thinking 和端点的优先级为命令行 > `PI_PY_*` 环境变量 > 项目设置 > 用户设置 > 内置默认值。端点从 `models.json` 加载，也支持 provider 的 `*_BASE_URL` 环境变量；设置文件中不存 API key。

用户或项目的 `settings.json` 示例：

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "thinking": "off",
  "cache_retention": "5m",
  "compaction": {
    "enabled": true,
    "reserve_tokens": 16384,
    "keep_recent_tokens": 20000,
    "summary_tokens": 2048
  }
}
```

Prompt caching 默认开启（`cache_retention: "5m"`，设为 `"1h"` 或 `false` 可调整/关闭，仅影响支持显式缓存标记的 API）：`anthropic-messages` 在 system 提示、最后一个工具与最后一条消息上写 `cache_control` 断点（1h 需追加 `extended-cache-ttl-2025-04-11` beta 头）；Bedrock 在 system 与最后一条用户消息后插入 `cachePoint`（仅 Claude 模型，其他型号自动跳过）。OpenAI 系为自动缓存，无请求侧标记；`prompt_cache_key` 等参数可经 `models.json` 的 `params` 传入（Responses 适配器同样生效）。用量统计区分 `cache_read`、`cache_write` 与 `cache_write_1h`（1h 写入按目录价或 2 倍基础输入价计）。

保存和移除凭据：

```bash
python -m pi --login openai
python -m pi --login anthropic
python -m pi --login openrouter --oauth
python -m pi --logout openrouter
python -m pi --list-models
```

API key 通过隐藏输入提示录入。凭据优先级为 `--api-key` > 已保存凭据 > `models.json` 的 `api_key` > `PI_PY_API_KEY` > provider 的环境变量。凭据在每次模型请求时重新读取；设备码 OAuth 的令牌接近过期时自动刷新，刷新失败不会覆盖旧记录。`auth.json` 是明文 JSON，采用临时文件加原子替换写入；POSIX 文件权限为 `0600`，Windows 依赖用户目录 ACL，并非系统钥匙串存储。`/logout` 只移除本地记录，不撤销服务端授权。

OpenRouter 登录会打印授权链接，需在本机浏览器打开；程序等待随机本地端口上的回调，五分钟后超时。实现使用 [OpenRouter PKCE 协议](https://openrouter.ai/docs/guides/overview/auth/oauth)。其他兼容服务可配置 [RFC 8628 设备码流程](https://www.rfc-editor.org/rfc/rfc8628)，专用订阅登录使用独立适配，命令如下：

```bash
python -m pi --login claude --oauth
python -m pi --login codex
python -m pi --login copilot
python -m pi --provider anthropic "hello"
python -m pi --provider openai-codex "hello"
python -m pi --provider github-copilot "hello"
```

Claude 使用 PKCE 授权链接并要求粘贴授权码；Codex 和 Copilot 使用设备码。适配包含令牌自动刷新、Claude OAuth 请求头与工具名转换、Codex 账户标识及 Responses 请求、Copilot 服务端点及请求来源头。以上流程已通过离线模拟测试，未验证真实账号授权或订阅可用性。

`models.json` 示例（字段使用 Python 的 snake_case）：

```json
{
  "providers": {
    "local": {
      "api": "openai-completions",
      "base_url": "http://localhost:8000/v1",
      "auth_required": false,
      "models": [
        {
          "id": "coder",
          "name": "Local coder",
          "reasoning": false,
          "context_window": 32768,
          "max_tokens": 4096,
          "input": ["text"],
          "costs": {"input": 0, "output": 0}
        }
      ]
    }
  }
}
```

启动：`python -m pi --provider local --model coder`。支持的 API 为 `openai-completions`、`anthropic-messages` 和测试用 `faux`。同名 provider/model 条目会覆盖对应元数据；其他内置模型仍保留。模型输入类型支持 `text`、`image`，价格单位为美元/百万 token；元数据应与上游实际能力一致。

provider 还可设置 `api_key`、`api_key_env`、`headers`。`api_key` 和 header 值支持 `$NAME` / `${NAME}` 环境变量插值，不执行 shell 命令。标准设备码服务需添加 `oauth` 对象，包含 HTTPS `device_url`、`token_url`、`client_id` 和可选 `scope`；生成的 access token 必须能直接用于该兼容端点的 Bearer 认证。

交互命令：

```text
/models                         列出模型与能力参数
/model openai/gpt-4o             切换 provider 和模型
/model gpt-4o-mini               在当前 provider 内切换
/login openai                   保存 API key
/login openrouter oauth         OAuth 登录
/logout openrouter              移除本地凭据
/settings                       查看合并后的默认设置
/settings thinking "high"       保存用户默认值
/reload                         重载配置、目录及当前模型
```

`/settings` 保存用户默认值，下次启动生效；`/reload` 可立即应用 thinking 和压缩配置。项目同名设置仍优先。模型目录目前为内置小型目录加用户定义，不含 TS 版的在线目录更新或所有专用 provider。

会话文件写在 `./.pi/sessions/*.jsonl`。启动时可指定 `--session <路径或ID>`、`--session-dir <目录>`、`--name <名称>`；`-c` 恢复最近会话。

新记录通过 `id`、`parent_id` 连接成树，另以追加记录保存活动节点和名称。切换分支不会删除原分支，重启后保留最后选择的节点；只有活动分支会送入模型。节点 ID 支持唯一前缀。为避免不完整的工具协议，不能切换或复制到尚缺工具结果的节点。

`/fork` 不带 ID 时复制当前分支；指定 ID 时复制至该消息之后，可在新会话中继续提问。当前为命令行文本操作，不含 TS 版的交互选择器、回填编辑器或离开分支时的自动摘要。JSONL 为 Python 自有格式，可重新加载，不保证与 TS 会话文件互通。HTML 为转义后的静态文本。导出不会覆盖已有文件。

## 评测系统（documentation-lift evals）

`packages/evals` 的 Python 移植版位于 `pi/evals/`（运行器代码）与两个仓库目录：`evals/`（评测套件，`*.docs.eval.py`）和 `docker/`（`Dockerfile` + `entrypoint.py`，容器侧）。TS 版依赖 Vitest 与 `vitest-evals`；Python 版用 `pi.evals.framework` 本地实现同等的最小契约（Harness 适配器、运行规范化、结构化输出裁判、`describe_eval` 注册、单用例运行器），报告仍写 Vitest JSON 结构（`testResults[].assertionResults[].meta`），因此 `report.ts` 的读取与配对逻辑按原样移植。

运行一次文档对比：

```bash
python -m pi.evals.cli --provider openai --model gpt-4o-mini
# 等价环境变量：PI_PROVIDER / PI_MODEL；重复次数：--runs-per-variant N 或 PI_EVAL_RUNS_PER_VARIANT
# 用例过滤：-t "adds the model" 或 --testNamePattern=...
```

流程与 TS 版一致：构建 `without_docs` / `with_docs` 两个镜像 → 在两个容器内发现用例并要求完全一致 → 按 (case, variant, run) 计划任务（奇数轮先 without、偶数轮先 with，降低顺序偏差）→ 每个任务独立容器运行 → 写入 `protocol.json`、`expected-runs.json`、`observations.jsonl` → 读取观察、成对配对、产出 `report.json` / `report.txt`。成对 lift 只在两臂都恰好产生一个分数时计入；缺失、重复、跳过、未计分或出错都会阻塞该对，任一对被阻塞时隐藏标题通过率并退出码 1。结果目录为忽略的 `.eval/<时间戳>_<id>/`，其中 `evals/<variant>/sessions/<sha256>/session.jsonl` 保存原生会话快照。

评测 harness（`create_pi_coding_agent_harness`）在临时根目录内组装与 CLI 相同的 Agent：隔离 HOME 与 agent 目录、注入宿主 `auth.json` 凭据、按名单选工具（文档评测固定为 `read/write/edit/grep/find/ls`，不暴露 shell 与联网搜索）、录制 JSONL 会话、统计 token/工具调用/成本，并在 POSIX 上按 `PI_EVAL_SANDBOX_UID/GID` 永久降权后再提示模型。评测失败时把部分运行数据（事件、用量、错误）附在异常上，报告据此保留部分指标。

系统提示词现在包含 `<docs>` 一节（PI 文档路由：README/docs/examples 路径与阅读指引），`SystemPromptOptions.include_pi_documentation` 控制其出现：默认自动探测包内文档是否存在；评测 harness 强制包含并在 `without_docs` 变体中剥离该节（`exclude_pi_documentation`），配合镜像内删除文档文件构成对照实验。`<cwd>` 一节同步改为 `<cwd>\n路径\n</cwd>` 的分节格式。

与 TS 版的差异（有意省略）：容器内不再运行 Vitest，而是 `docker/entrypoint.py` 直接发现/运行用例并写 Vitest JSON；`reload` 输入步骤不受支持（Python 会话无扩展重载）；TS 的 Vite transformed-module 保护无对应物，改由镜像内 `chmod go-rwx` 保证评测源码仅 root 可读；宿主侧 host evals（非 `*.docs.eval` 的普通套件）与 acme/configured-runtime 夹具未移植；npm tarball 打包换为容器内 `pip install /repo/pi-python`，并把 README 复制到 site-packages 作为 `with_docs` 的文档。凭据来源探测在降权时按环境变量名近似（`PROVIDER_API_KEY` / `PI_PY_API_KEY`），TS 版使用 ModelRuntime 的精确 source。

## 作为库使用

```python
import asyncio
from pi.agent import Agent, AgentTool, AgentToolResult
from pi.ai.models import resolve_model
from pi.ai.types import TextContent

model = resolve_model("faux")

echo = AgentTool(
    name="echo",
    label="echo",
    description="repeat text back",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    execute=lambda call_id, params, signal=None, on_update=None: asyncio.sleep(
        0, result=AgentToolResult(content=[TextContent(text=params["text"])])
    ),
)

agent = Agent(model=model, tools=[echo], system_prompt="You are helpful.")

async def main():
    async for event in agent.stream_events("hello"):
        print(event.type if hasattr(event, "type") else event)

asyncio.run(main())
```

## 测试

```bash
pip install pytest
python -m pytest tests -q
```

测试全部离线：provider 层用 canned SSE 通过 `httpx` mock 传输验证事件映射；agent loop 用 `faux` 脚本驱动；工具层直接操作临时目录。

## 未移植（有意省略）

### Python 上下文压缩

Agent 和 CLI 默认启用自动上下文压缩：接近模型上下文窗口时，摘要旧消息并保留近期消息、系统指令和工具声明。切分不会拆开工具调用与其结果。明确的上下文溢出错误会触发强制压缩并重试一次；其他错误不会触发重试。摘要失败、为空或达到输出上限时不会替换已有摘要。

原始会话 JSONL 保持完整。摘要与压缩切点（checkpoint：已消费消息数 + 末条消息指纹）一起持久化为会话 `compaction` 条目；恢复会话时按 checkpoint 从加载的 transcript 重建切点，校验通过则直接续用，不再重复投影或重新产生摘要调用费用。checkpoint 与 transcript 不匹配（例如切到更早分支）时清空摘要与切点，与运行中的自愈行为一致；旧版本会话文件只有摘要、没有 checkpoint，恢复摘要但切点未知，行为同修复前。单条过大的最新输入、系统提示词或不可拆分工具结果仍可能超过窗口，需要缩短输入。

库调用可通过 `Agent(..., compaction=CompactionSettings(...))` 配置；从 `pi.agent.compaction` 导入 `CompactionSettings`。默认 `reserve_tokens=16384`、`keep_recent_tokens=20000`、`summary_tokens=2048`，可设置 `enabled=False` 关闭。未知模型应准确设置 `model.context_window`。Token 数使用 UTF-8 长度估算，并非模型的精确分词结果。

- TS TUI 的全屏对话布局、富文本/图片渲染与工具输出折叠；Python 已有多行编辑、补全、选择器、主题及快捷键
- codemode、协议/服务器等包；扩展系统、MCP、durable、telemetry 已有部分 Python 实现
- 其他 provider 的专属认证；已实现 Claude、Codex、Copilot 订阅登录、OpenRouter PKCE 及可配置标准设备码 OAuth
- 模型目录只内置少量常见型号，任意 id 会以默认参数运行
