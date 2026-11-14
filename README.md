# pi-python

pi（TypeScript monorepo）的 Python 移植版。目标是 1:1 复刻整个 monorepo 的全部 package——`ai`、`agent`、`coding-agent`（含 codemode/MCP/TUI）、`evals`、`durable`、`telemetry`，以及 `protocol`/`server`/`client` 路由传输与 `chord` 服务层——用 Python 忠实还原对应实现并保持可直接运行。与 TS 版尚有差距的部分见文末「尚未移植」。

## 复刻版本基线

- 源仓库：[earendil-works/pi](https://github.com/earendil-works/pi)（TypeScript monorepo，13 个 package）。
- 基线版本：`v1.0.3`（tag 提交 `d78dc83d6`，2026-10-05），全部 package 的 `package.json` 版本一致为 `1.0.3`。
- 移植基点：`analysis-notes` 分支提交 `293fb03cb`（2026-11-07），即 `v1.0.3` 之后 8 个仅更新 pi-python 子模块的提交，TS 源码与 `v1.0.3` 相同。
- 路由协议版本：两侧一致为 `PROTOCOL_VERSION = 8`（`packages/protocol/src/protocol.ts` 与 `pi/protocol.py`）。
- 后续同步：`main` 分支与更新的 release（当前最新为 `v1.5.0`，2026-11-02）不在本基线内；TS 侧版本推进时更新本节记录的 tag/提交。

## 与 TS 版的对应关系

| Python 模块 | TS 源 | 内容 |
| --- | --- | --- |
| `pi.ai.types` | `packages/ai/src/types.ts` | 四角色消息模型（system/user/assistant/toolResult）、内容块、Usage、事件协议 |
| `pi.ai.providers` | `packages/ai/src/api/*`、`providers/*` | `anthropic-messages`、`openai-completions`、`openai-responses`、`google-generative-ai`、`google-vertex`、`azure-openai`、`azure-openai-responses`、`bedrock-converse-stream`、`pi-messages`（Radius 网关）的流式适配 + `faux` 脚本化测试 provider |
| `pi.ai.utils` | `packages/ai/src/utils` | `normalize_context`（系统消息重放折叠）、`get_tool_state_changes`（工具集增量声明）、JSON Schema 参数校验 |
| `pi.ai.retry` | `packages/ai/src/utils/retry.ts` | `is_retryable_assistant_error` 瞬时错误分类（含 #10379 的 HTTP/2 `pending stream has been canceled`）+ `retry_assistant_call` 有界重试循环（指数退避、abort 归一化、回调） |
| `pi.agent.agent_loop` | `packages/agent/src/agent-loop.ts` | 主循环：steering/follow-up 队列、顺序/并行工具执行、`beforeToolCall`/`afterToolCall` 钩子、截断消息失败、错误/中止处理 |
| `pi.agent.agent` | `packages/agent/src/agent.ts` | 有状态 Agent：transcript、队列、事件订阅、prompt/abort |
| `pi.coding_agent.tools` | `coding-agent/src/core/tools/*` | `read`（截断/offset/limit/图片）、`write`（父目录自动创建）、`edit`（唯一匹配 + 重叠校验 + CRLF/BOM 保留 + 模糊匹配）、`bash`（流式输出、尾截断、超时、进程树终止） |
| `pi.coding_agent.system_prompt` | `coding-agent/src/core/system-prompt.ts` | 分节系统提示词 + AGENTS.md 项目上下文 |
| `pi.coding_agent.session` | `coding-agent/src/core/session-manager.ts` | JSONL 会话树、分支、切换、命名、导出、`--continue` 恢复 |
| `pi.coding_agent.cli` | `coding-agent/src/cli` | 单次模式 + 交互 REPL + 斜杠命令 + Ctrl+C 中止 |
| `pi.codemode` | `packages/codemode` | codemode 工具：模型编写的 Python 在隔离子进程中经 Agent 管线组合调用其他工具，含按 token 预算的延迟暴露 |
| `pi.coding_agent.catalog_sync` / `pi.coding_agent.radius_catalog` | models.dev 目录 + `packages/ai/src/providers/radius-config.ts` | 在线模型目录同步与 Radius 网关发现，缓存叠加进模型注册表 |
| `pi.evals.harness` | `packages/evals/src/harness.ts` | 隔离评测 harness：临时工作区、凭据注入、会话录制、用量统计、POSIX 工具沙箱、文档变体 |
| `pi.evals.plan` | `packages/evals/src/plan.ts` | 发现用例解析与 (case, variant, run) 任务计划 |
| `pi.evals.report` | `packages/evals/src/report.ts` | 观察读取、成对配对、lift/标记/运营总量与文本报告 |
| `pi.evals.docker` | `packages/evals/src/docker.ts` | 变体镜像构建、auth.json 校验、容器发现与单任务运行 |
| `pi.evals.cli` | `packages/evals/src/cli.ts` | 评测编排 CLI：参数解析、发现对齐、协议落盘、报告产出 |
| `pi.evals.framework` | `vitest-evals`（npm 依赖） | 本地替代：Harness 适配器、运行规范化、结构化输出裁判、用例发现与运行器、`tool_calls` 提取 |
| `pi.evals.host` | `packages/evals/vitest.evals.config.ts`（host project） | host 评测运行器：发现非 docs 套件、顺序执行、聚合 Vitest JSON 报告 |
| `pi.durable` | `packages/durable` | 持久化运行时：会话/任务/文档先提交后可见、`ExecutionEnv`、持久化工具与一致性套件（见下文「持久化运行时」） |
| `pi.telemetry` | `packages/telemetry` | 遥测契约：span/上下文类型、OTLP/HTTP(JSON) 导出、内存与空实现后端 |
| `pi.protocol` / `pi.server` / `pi.client` | `packages/protocol`、`packages/server`、`packages/client` | Pi v8 路由封包传输：严格 JSON 信封与定长 CBOR 帧、握手与身份校验、并发关联请求、TCP/Unix 传输、连接作用域端点与会话路由（见 ROUTED_PROTOCOL.md） |
| `pi.chord` | `packages/chord` | Chord 服务层：契约绑定、`$chord.service` 控制调用、delta wire 编解码（`pi.delta`）、provider/consumer、状态编解码注册表；与 TS 的互通由 `tests/test_chord_interop.py` 以 Node 直接加载 TS 源码验证（见 ROUTED_PROTOCOL.md） |
| `pi.coding_agent.mcp` | `packages/mcp` | MCP 客户端：stdio / Streamable HTTP（JSON/SSE）传输、工具发现与注册 |
| `pi.coding_agent.tui` | `packages/tui` | 终端 UI：全屏对话布局、多行编辑、补全、选择器、主题、快捷键、ANSI Markdown、终端图片 |

## 安装

要求 Python 3.9+。以下命令从仓库根目录执行，建议使用独立虚拟环境。

Windows PowerShell：

```powershell
cd pi-python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

macOS / Linux：

```bash
cd pi-python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

安装会自动安装 `httpx`、`Pillow`、`PyYAML`、`prompt_toolkit` 和 `wcwidth`。`-e` 表示可编辑安装，修改 Python 源码后无需重新安装。

如果 PowerShell 禁止运行激活脚本，可直接使用 `.\.venv\Scripts\python.exe -m pip install -e .` 安装，后续将命令中的 `python` 替换为 `.\.venv\Scripts\python.exe`。

每次打开新终端后，需要重新激活虚拟环境。退出虚拟环境使用 `deactivate`。

## 运行

先验证安装（无需 API key，不调用真实模型）：

```bash
python -m pi --help
python -m pi --provider faux --no-session "hello"
```

配置 API key 后启动。以下以 OpenAI 为例，将占位密钥替换为自己的密钥。

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY = "sk-..."
python -m pi --provider openai
```

macOS / Linux：

```bash
export OPENAI_API_KEY="sk-..."
python -m pi --provider openai
```

不带 prompt 参数进入交互模式，输入问题后按 Enter 提交，输入 `/exit` 退出。也可直接执行一次任务，或恢复上次会话：

```bash
python -m pi --provider openai "列出当前项目中的 Python 文件"
python -m pi --provider openai -c
python -m pi --provider openai --tui fullscreen
```

安装后也可以用 `pi` 代替 `python -m pi`。如果机器同时安装了 TypeScript 版 pi，建议使用 `python -m pi` 明确启动 Python 版。可先切换到要处理的项目目录，再启动；工具将以该目录作为工作目录。

使用 OpenAI 兼容端点时，可添加 `--base-url <服务地址>` 和 `--model <模型ID>`；服务地址应包含服务要求的 API 路径（例如 `/v1`）。使用 Anthropic 时，在当前终端设置 `ANTHROPIC_API_KEY`（PowerShell 使用 `$env:ANTHROPIC_API_KEY = "sk-ant-..."`，macOS / Linux 使用 `export ANTHROPIC_API_KEY="sk-ant-..."`），然后运行：

```bash
python -m pi --provider anthropic "解释 pyproject.toml"
```

常用参数：`--provider <id>`、`--model <id>`、`--base-url`、`--api-key`、`--thinking off|minimal|low|medium|high|xhigh|max`、`-c` 继续上次会话、`--no-session` 不落盘、`--sync-models` 同步在线模型目录。

工具选择参数：`--tools <列表>`（逗号分隔的允许名单，支持 `*` 通配，如 `--tools read,codemode,'mcp__radius__*'`；名单不含 `mcp__` 开头的条目时保留 MCP 工具但只供 codemode 调用、不声明给模型，点名 `mcp__` 条目时匹配的 MCP 工具正常声明）、`--exclude-tools <列表>`（排除名单，含 MCP 工具，同样支持 `*`）、`--no-tools` 移除全部工具（含 MCP）、`--no-mcp` 单次运行禁用内置 MCP 支持（不连接服务器、无 MCP 工具）。

CLI 还内置 `openai`、`anthropic`、`openrouter`、`google`、`openai-responses`、`google-vertex`、`azure-openai`、`azure-openai-responses`、`amazon-bedrock`、`openai-codex`、`github-copilot`、`kimi-coding`、`xai`、`radius`、`faux`，并可从 `models.json` 注册其他兼容 provider。

环境变量也可用 `PI_PY_PROVIDER` / `PI_PY_MODEL` / `PI_PY_API_KEY` / `PI_PY_BASE_URL`。

交互模式（不带 prompt 参数）支持斜杠命令（`/help` 显示完整列表）：

```text
/model <id>    切换模型
/models        列出模型与能力参数
/thinking [level]  查看或设置 thinking 级别
/clear         清空对话（保留系统提示词）
/compact [instructions]  立即摘要较早的上下文
/image [model] <prompt>  经 openrouter-images 生成图片
/sync-models   刷新在线模型目录缓存
/evals <file>  对当前 agent 运行评测用例
/package [install <source>|update <name>|remove <name>|list]  管理资源包
/new           新建会话
/resume        列出会话（显示 ID、名称和消息数）
/resume <id>   切换会话，也可提供完整文件路径
/session       查看当前会话路径、名称、活动节点和消息数
/name <name>   命名当前会话
/tree          显示全部分支与消息 ID，* 标记活动节点
/tree <id>     回到指定消息之后；下一次输入创建新分支
/fork <id>     将指定消息及其祖先复制为独立会话
/clone         将当前活动分支复制为独立会话
/export <path.jsonl|path.html|path.md>  导出当前活动分支
/export-ts <path.jsonl>  按 TS pi 会话格式导出活动分支
/usage         累计 token 用量与成本
/trust         保存当前目录的项目信任决策
/bug [hint]    上传或导出 bug 报告（环境元数据 + 失败诊断，脱敏）
/help /exit
Ctrl+C         中止当前运行
```

## 原生模型协议

除原有 Chat Completions 和 Anthropic Messages 外，已接入以下专用 API：

| provider | API | 认证 |
| --- | --- | --- |
| `openai-responses` | `openai-responses` | `OPENAI_API_KEY`、`--api-key` 或该 provider 的已保存凭据 |
| `google` | `google-generative-ai` | `GEMINI_API_KEY`、`--api-key` 或已保存凭据 |
| `google-vertex` | `google-vertex` | Google Cloud 服务账号：`GOOGLE_APPLICATION_CREDENTIALS` 指向 JSON 密钥；`GOOGLE_OAUTH_ACCESS_TOKEN` 可直接提供已签发的 token |
| `amazon-bedrock` | `bedrock-converse-stream` | boto3 的 AWS 凭据链，支持环境变量、profile 和角色凭据 |
| `azure-openai` | `azure-openai`（Chat Completions 传输） | `AZURE_OPENAI_API_KEY`（`api-key` 头）或 Entra ID（`AZURE_ACCESS_TOKEN`，Bearer） |
| `azure-openai-responses` | `azure-openai-responses` | 同上 |
| `radius` | `pi-messages`（Radius 网关协议） | `python -m pi --login radius`（浏览器或设备码 OAuth）或 `RADIUS_API_KEY` |

```bash
python -m pi --provider openai-responses --model gpt-4o-mini "hello"
python -m pi --provider google --model gemini-2.5-flash "hello"
pip install -e ".[bedrock]"
python -m pi --provider amazon-bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0 "hello"
python -m pi --provider google-vertex --model gemini-2.5-flash "hello"
python -m pi --provider azure-openai --model gpt-4o "hello"
```

示例型号用于展示参数，不保证账号、地区或上游当前开放访问。Bedrock 使用 `AWS_REGION` / `AWS_DEFAULT_REGION`，未设置时默认为 `us-east-1`；`--base-url` 可覆盖端点。AWS 密钥不写入 Pi 的 `auth.json`。Bedrock 的可选 boto3 版本固定为兼容 Python 3.9 的 `1.40.75`。

各协议的差异：

- Vertex 复用 Gemini `generateContent` 的请求构造与 SSE 转换，改用服务账号签名 RS256 JWT 换取 OAuth token（纯标准库实现，无额外依赖）；项目与区域取 `GOOGLE_CLOUD_PROJECT` / `VERTEX_PROJECT` / `GCP_PROJECT` 与 `GOOGLE_CLOUD_REGION` / `VERTEX_REGION`（默认 `us-central1`），端点在请求时构造。
- Azure 端点取 `AZURE_OPENAI_ENDPOINT` 或 `AZURE_OPENAI_RESOURCE_NAME`（拼为 `https://<资源名>.openai.azure.com`），部署名走 URL 路径；`api-version` 来自 models.json `params` 的 `api_version` 或 `AZURE_OPENAI_API_VERSION`（Chat Completions 默认 `2024-10-21`，Responses 默认 `v1`）。Entra 认证使用 `AZURE_ACCESS_TOKEN`。
- Radius（`pi-messages`）是 camelCase 转录与 SSE 事件的网关协议；网关默认 `https://radius.pi.dev`，可经 `RADIUS_GATEWAY` 或凭据记录覆盖，令牌过期自动刷新。
- 三种 Responses 适配（openai-responses、openai-codex-responses、azure-openai-responses）支持 reasoning summary 和加密 reasoning 条目回放，并支持内置托管工具（`Tool.hosted`，如 `{"type": "web_search"}`）：调用输出项原文保存在 `ToolCall.raw_item` 中，后续请求原样回放；Gemini 保存工具调用的 thought signature，以便恢复会话后回传。Bedrock 支持 ConverseStream reasoning 内容回放，Claude 模型配置 `reasoning=true` 后可使用 thinking budget；其他模型专用推理参数经 `models.json` 的 `params` 逐键注入 Converse 请求（与内建键一层合并，显式选项优先），`top_k` 按家族自动路由（Nova 为 `additionalModelRequestFields.inferenceConfig.topK`，其余为扁平 `additionalModelRequestFields.top_k`）。

全部适配均支持流式文本、工具调用、输入图片、工具结果、停止原因及用量映射。`models.json` 的 `api` 可指定上述任一原生 API，适用于自定义端点；内置 `openai` 默认协议仍为 Chat Completions。未知的原生 provider 模型 ID 可以使用，但需要在 `models.json` 中补准确的上下文窗口、输出限制和能力参数。

实现参考 [OpenAI Responses 流式事件](https://developers.openai.com/api/reference/resources/responses/streaming-events)、[Google generateContent](https://ai.google.dev/api/generate-content) 和 [AWS ConverseStream](https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock-runtime/client/converse_stream.html)。验证使用模拟 SSE 和 AWS 事件流，未执行真实付费请求。音视频输入（`AudioContent`/`VideoContent`，`@file` 附件按扩展名识别）按 provider 支持情况转换：Gemini 音频与视频均走 `inlineData`；OpenAI 系仅音频且只接受 wav/mp3；Anthropic 与 Bedrock Converse 不支持并会报错。

## 模型目录同步

模型目录由三部分合并：内置目录、用户/扩展定义（`models.json` 优先于扩展注册）、在线目录缓存。在线同步有两个来源：

- models.dev 公共目录：覆盖 `anthropic`、`openai`、`google`、`google-vertex`、`azure-openai`、`amazon-bedrock`、`openrouter` 七个 provider，为用户与扩展都未显式配置的 provider 补充型号（上下文窗口、输出限制、价格）。触发方式为 `--sync-models` 参数、交互 `/sync-models` 命令或 RPC `sync_models`；结果缓存在 agent 目录的 `models-cache.json`，有效期 7 天，过期前 reload 直接读缓存。
- Radius 网关发现：仅在已配置 Radius 凭据（`/login radius` 保存的凭据或 `RADIUS_API_KEY`）时于启动刷新，与 TS 一致；未配置时不发起请求。`PI_OFFLINE=1` 或 `sync_catalog: false` 跳过；`/login radius` 成功后也会刷新一次。通过网关 `/v1/config` 读取模型与 `baseUrl`，缓存在 `radius-cache.json`；发现的型号叠加到 `radius` provider，用户在 `models.json` 声明的同 id 条目仍优先。

settings 的 `sync_catalog: false` 同时关闭两个缓存叠加（缓存文件保留）。与 TS 版的差异：TS 在每次启动时刷新 models.dev 目录，Python 版仅在显式触发或缓存有效期内叠加。

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

快捷键使用 prompt_toolkit 的键名，空格分隔组合序列；每项覆盖对应默认值，重启后生效。重复绑定会报错。`--tui fullscreen` 使用固定对话区，支持 ANSI Markdown、工具输出折叠和图片显示。

图片来自工具结果（例如 `read` 读取图片文件）、助手消息和随 prompt 附加的图片。Kitty、Ghostty、WezTerm、Warp 使用 Kitty 图形协议，iTerm2 使用 OSC 1337；Kitty 收到非 PNG 数据时先用 Pillow 转码。全屏对话区由 prompt_toolkit 重绘、无法透传图形协议序列，因此改用 24 位色半块字符渲染（每个单元格上下两个像素）；协议路径用于不经过 prompt_toolkit 重绘的输出，即 `--tui off`、单次执行，以及 `--tui on` 时编辑器上方的流式输出。终端不支持图片、真彩色或图像解码失败时回退为 `[Image: ...]` 文本。`PI_IMAGE_PROTOCOL=kitty|iterm2|none`、`PI_TRUE_COLOR=0|1`、`PI_HYPERLINKS=0|1` 可覆盖自动检测。已用受控输入验证编辑、选择器和中止操作，尚未完成所有实际终端的人工验证。

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

`/greeting Alice` 返回的字符串会作为 prompt 发送给模型；返回 `None` 表示命令已处理、不启动模型。扩展命令优先于同名模板，不能覆盖内置命令。扩展加载后仍可动态注册工具、Provider、虚拟模型和 MCP 服务；`refreshTools()` 发布新增工具。重名工具/命令或失败 factory 会给出 stderr 诊断。

事件回调接受 `(event, ctx)`；`ctx` 提供 `agent`、`cwd` 和可选 `session`。支持 `session_start`、`session_shutdown`，以及 Agent 的 `agent_start/end`、`turn_start/end`、`message_start/update/end`、`tool_execution_start/update/end` 通知。`api.on` 返回取消订阅函数。通知异常仅记录诊断；回调按加载与注册顺序执行。

`input` 回调接受 `{"text": ...}`，可返回 `pi.coding_agent.extensions.InputResult(text=...)` 修改后续输入，或 `InputResult(handled=True)` 截止处理。多个输入变换依次组合。`tool_call` 可返回 `BeforeToolCallResult` 阻止工具执行；处理器异常也会使该工具调用失败。`tool_result` 可返回 `AfterToolCallResult` 修改输出，多次返回以最后一个为准。普通通知的返回值忽略。输入/命令处理适用于普通 prompt；RPC 的 steer/follow_up 保持原队列行为。

`/extensions` 查看来源，`/help` 列出扩展命令。`/reload` 执行旧扩展的 `session_shutdown`、撤销工具与事件订阅，然后重新执行 factory 并发送 `session_start`。退出同样执行清理。长期资源应在 `session_start` 创建，在幂等的 `session_shutdown` 中关闭，不应在 factory 中启动后台任务。

扩展同时作用于文本、JSON 和 RPC 模式。RPC `get_commands` 列出扩展命令，空闲时可调用 `reload_extensions`；通过 `prompt` 发送 `/命令`，纯本地处理返回 `disposition=handled`，此时不产生新的 `agent_settled`。

扩展是本机 Python 代码，以 Pi 进程权限运行，没有沙箱或项目授权界面。扩展 factory、回调和注册工具内的普通 `print` 重定向到 stderr，推荐使用 `api.log`；自行创建的后台任务仍须自行管理输出和生命周期。TS/JS 扩展可通过 Node 桥接（`pi.coding_agent.ts_extension`，需 PATH 中的 Node 22.6+）加载。桥接支持工具、消息和会话条目渲染器、Markdown transformer、虚拟模型、动态 MCP、Provider 注销、`refreshTools`、事件总线、状态 getter 和扩展 CLI flag。渲染函数在 Node 侧执行，组件的 `render(width)` 结果作为终端行返回。

`ctx.ui.custom(factory, options)` 的交互组件保留在 Node 进程中：Python 侧显示组件每次 `render(width)` 的行，`tui.requestRender()` 触发重绘，键盘输入原样转发给 `handleInput`（同一批到达的 ESC+字母会合并为 alt 组合），SGR/X10 鼠标事件转换为 TS 的 `TuiMouseEvent` 后转发给 `handleMouse`（含 press/release/click/drag/wheel）。`keybindings.matches(data, action)` 支持 `tui.select.*`、`tui.input.*` 和常用的 `app.*` 默认按键；组件调用 `done(result)` 后视图关闭并让 `ui.custom` 的 promise 取得结果。

`options.overlay` 为真时按 `overlayOptions`（`width`/`minWidth`/`maxHeight`/`anchor`/`offsetX`/`offsetY`/`row`/`col`/`margin`/`visible`）把组件放进浮动框；`onHandle` 提供 `hide`/`setHidden`/`isHidden`/`focus`/`unfocus`/`isFocused`/`getBounds`。限制：浮动框背景是空白而不是实时 transcript；`hide()` 会关闭视图并让 `ui.custom` 以 `undefined` 结束（TS 中 promise 保持 pending）；`setHidden(true)` 只隐藏浮动框，按键仍由组件接收。

`ctx.ui.setWidget(key, lines | factory, options)`、`setHeader(factory)` 和 `setFooter(factory)` 也走桥接：工厂在 Node 中按终端宽度渲染，`tui.requestRender()` 重新渲染并推送新行。widget 按 key 覆盖，`placement` 为 `aboveEditor`/`belowEditor`；这些区域只在 `--tui fullscreen` 中显示。

`ctx.ui.setEditorComponent(factory)` 把输入编辑器换成扩展组件：每次读取输入时显示 transcript 和编辑器行，按键与鼠标转发给组件，`onSubmit` 的文本作为本次输入返回，`Escape`/`Ctrl+D` 在组件未自行定义时映射为中止/退出。该模式使用独立的输入视图，`--tui on` 下也会临时占用全屏。

RPC/JSON 模式没有交互宿主，`ui.custom` 立即返回 `undefined`（与 TS RPC 一致），widget/header/footer 和编辑器接管被忽略。代理运行中打开自定义视图会暂停编辑器输入循环，关闭后恢复。

扩展 flag 与 TS CLI 一样使用未知长参数的两阶段解析。`--flag` 传入布尔值，`--flag=value` 或 `--flag value` 传入字符串；扩展加载后校验名称和类型，未注册参数会报错。

## JSON 输出与 RPC

`python -m pi --mode json --provider faux --no-session "hello"` 输出 JSONL 事件。每行是一个 JSON 对象，包含 Agent、消息和工具执行事件；结束时输出 `agent_settled`。诊断写入 stderr。JSON 模式要求提供 prompt，不进入交互输入循环。

`python -m pi --mode rpc --provider faux --no-session` 启动长期运行的 stdio RPC 服务。通过 stdin 发送 LF 分隔的 JSON 命令，从 stdout 持续读取响应和事件。例如：

```json
{"id":"1","type":"get_state"}
{"id":"2","type":"prompt","message":"hello"}
{"id":"3","type":"abort"}
```

响应格式为 `{"type":"response","id":"1","command":"get_state","success":true,"data":{...}}`。失败时 `success=false` 并携带 `error`；非法 JSON 返回 `command=parse`，服务继续处理下一行。`prompt` 响应只表示接受请求，完成要等待 `agent_settled`。运行中可读取状态、发送 `steer` / `follow_up` 或 `abort`；第二个普通 prompt 会被拒绝。关闭 stdin 请求退出，服务中止活动运行并清理 MCP 连接。

当前支持的命令：`prompt`、`steer`、`follow_up`、`abort`、`get_state`、`get_messages`、`get_tools`、`get_available_models`、`set_model`、`set_thinking_level`、`new_session`、`switch_session`、`fork`、`clone`、`get_tree`、`get_entries`、`set_session_name`、`export`、`sync_models`、`reload_extensions`。会话命令要求启用持久化；修改模型、会话和 `sync_models` 要求 Agent 空闲。参数分别使用 `message`、`model`/`provider`、`level`、`path`、`entry_id` 或 `name`。`export` 根据路径扩展名导出 JSONL、HTML 或 Markdown。

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

扩展可用 `registerMcpServer`、`unregisterMcpServer` 和 `getMcpServers` 动态管理同一运行时的服务；变更会重建连接并刷新 Agent 工具。文件配置优先于同名扩展注册。

## Codemode

`packages/codemode` 的移植版（`pi.codemode` 子包：`tool` 适配 Agent 工具、`host` 管理子进程、`worker_source` 内嵌 worker 程序）。settings 中配置后启用：

```json
{
  "codemode": {
    "enabled": true,
    "timeout_ms": 120000
  }
}
```

启用后 Agent 多出一个 `codemode` 工具：模型写一段 Python，代码在隔离的 `python -I` 子进程中运行，只能通过 JSON 行协议调用宿主；宿主再经 Agent 正常管线执行工具调用，因此校验、钩子与权限检查全部生效。子进程不提供网络、进程或定时器原语。工具目录覆盖启用时已注册的全部工具（含 MCP 工具）。

脚本契约：`tools.<name>(args)` 阻塞调用单个工具；`tools.<name>.start(args)` / `pi.call(...)` 启动调用返回句柄，`wait(handle)` / `wait_all(...)` / `wait_any(...)` 组合并发；`ALL_TOOLS` 列出全部工具；`text(value)` / `image(dataUri)` / `print(...)` / `console.*` 追加输出项；`exit(value)` 立即结束；`store(key, value)` / `load(key)` 跨调用保存小体积 JSON 值（成功脚本以 `codemode-store` 会话条目持久化，失败脚本保留输出但不落盘）；可选首行 `# @options: {"timeout_ms": .., "max_output_tokens": ..}` 控制超时与输出预算。

延迟暴露（deferred exposure）：工具目录按 MCP 命名空间分组、按 token 预算截取（默认 3000 token，超出部分不内联列出）；未列出的工具经 `search_tools(query, {limit, namespace})`、`describe_tool(name)`、`describe_namespace(name)` 按需查找。输出超过 `max_output_tokens`（默认 10000）时保留首尾并截断，完整输出落盘到临时文件。

嵌套调用记录：脚本内的每次工具调用都经统一入口执行，获得稳定 ID（顶层调用 A 的子调用为 A/1，更深为 A/1/1）和直接父 ID；`tool_execution_*` 事件带 `parent_tool_call_id`。顶层工具结果消息只保存一份结果，并附带 `nested_calls` 摘要（参数、状态、耗时）。每个顶层调用的记录上限为 256 条、单条参数 8 KiB（UTF-8 JSON 字节）、保留参数合计 32 KiB；超限时省略参数或丢弃调用并置 `complete=false`，不影响执行和用量累计。

用量口径：`AgentToolResult.usage` 只表示该工具自身产生的用量，不包含子工具。顶层结果落盘前合并一次：`ToolResultMessage.usage = 顶层工具自身 usage + 全部嵌套工具自身 usage`；`take_record` 具有消费语义，不会在多个消息事件中重复合并。`/usage` 与 `session_stats` 只统计 assistant 消息、顶层工具结果消息和显式额外用量（如压缩摘要），不遍历 `nested_calls`；缺少模型归属的工具用量归入 `Tools/summaries` 桶。`tool_calls` 仍是顶层模型发起的调用数，`nested_tool_calls` 是已保留的子调用记录数，`nested_tool_calls_complete=false` 时该数是下界。

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

同名资源保留首次发现的条目并输出警告。优先次序为显式项目路径、显式用户路径、项目默认目录、用户默认目录及 Agent Skills 目录。无效 YAML、缺少描述的 skill、空正文和模板占用内置命令等情况会跳过并输出诊断。

扩展、skills、prompts 和 themes 可从 npm tarball（`npm:<pkg>[@version]`）或 git 仓库（`git:<url>[#ref]`）安装到 agent 目录的默认发现位置：`/package install <source>`、`/package update <name>`、`/package remove <name>`、`/package list`。安装从不执行生命周期脚本（npm 包按普通归档解包，git 包仅 clone/update），安装记录保存在 `packages.json`；安装、更新和移除会即时刷新当前会话的资源目录。

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
| `settings.json` | 默认 provider、model、thinking、session_dir、compaction、cache_retention、default_project_trust、enable_install_telemetry、sync_catalog、codemode、retry |
| `models.json` | 自定义 provider、模型元数据与覆盖 |
| `auth.json` | API key、OAuth access/refresh token 和过期时间 |
| `trust.json` | 项目信任决策（`{规范化路径: true/false}`，沿父目录就近生效） |
| `crashes.json` | 最近 5 条未捕获异常记录（7 天有效） |

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
  },
  "retry": {
    "enabled": true,
    "max_retries": 3,
    "base_delay_ms": 2000,
    "max_agent_delay_ms": 60000
  }
}
```

Prompt caching 默认开启（`cache_retention: "5m"`，设为 `"1h"` 或 `false` 可调整/关闭，仅影响支持显式缓存标记的 API）：`anthropic-messages` 在 system 提示、最后一个工具与最后一条消息上写 `cache_control` 断点（1h 需追加 `extended-cache-ttl-2025-04-11` beta 头）；Bedrock 在 system 与最后一条用户消息后插入 `cachePoint`（仅 Claude 模型，其他型号自动跳过）。OpenAI 系为自动缓存，无请求侧标记；`prompt_cache_key` 等参数可经 `models.json` 的 `params` 传入（Responses 适配器同样生效）。用量统计区分 `cache_read`、`cache_write` 与 `cache_write_1h`（1h 写入按目录价或 2 倍基础输入价计）。

自动重试默认开启（`retry.enabled`），对瞬时 provider/传输错误（overloaded、rate limit、连接中断、HTTP/2 停滞等；配额/订阅类错误不重试，context overflow 由 compaction 处理）按 `base_delay_ms * 2^(attempt-1)` 指数退避重试，单次延迟上限 `max_agent_delay_ms`。CLI/RPC 的 agent 运行在可重试错误后自动继续并发出 `auto_retry_start` / `auto_retry_end` 事件；被重试替换的失败轮次保留在原始 transcript 和 session 中（`AssistantMessage.retried`），但不进入模型请求，resume 后投影一致。compaction 摘要调用共用同一重试预算；durable harness 的 `pi.generation` 与 `pi.compact` 在 checkpoint 中记录 attempt 并以 `retry` phase 退避（`Harness(session, model, retry=RetryPolicy(...))` 传入，默认不重试；`pi durable --retry N` 可为 CLI 打开）。库调用通过 `Agent(..., retry=RetryPolicy(...))` 配置，默认 `None` 不重试。

保存和移除凭据：

```bash
python -m pi --login openai
python -m pi --login anthropic
python -m pi --login openrouter --oauth
python -m pi --login custom     # 自定义 API 模型（保存后退出）
python -m pi --logout openrouter
python -m pi --list-models
```

API key 通过隐藏输入提示录入。凭据优先级为 `--api-key` > 已保存凭据 > `models.json` 的 `api_key` > `PI_PY_API_KEY` > provider 的环境变量。凭据在每次模型请求时重新读取；设备码 OAuth 的令牌接近过期时自动刷新，刷新失败不会覆盖旧记录。`auth.json` 是明文 JSON，采用临时文件加原子替换写入；POSIX 文件权限为 `0600`，Windows 依赖用户目录 ACL，并非系统钥匙串存储。`/logout` 只移除本地记录，不撤销服务端授权。

OpenRouter 登录会打印授权链接，需在本机浏览器打开；程序等待随机本地端口上的回调，五分钟后超时。实现使用 [OpenRouter PKCE 协议](https://openrouter.ai/docs/guides/overview/auth/oauth)。其他兼容服务可配置 [RFC 8628 设备码流程](https://www.rfc-editor.org/rfc/rfc8628)，专用订阅登录使用独立适配，命令如下：

Claude 使用 PKCE 授权链接并要求粘贴授权码；Codex 和 Copilot 使用设备码；`--login openai --oauth` 走 Sign in with ChatGPT（不带 `--oauth` 时保存 API key；`chatgpt` / `openai-chatgpt` 为同一目标的别名）；Kimi 与 xAI 使用各自的设备码流程；Radius 可选浏览器回调或设备码，登录成功后立即刷新一次网关模型目录。适配包含令牌自动刷新、Claude OAuth 请求头与工具名转换、Codex 账户标识及 Responses 请求、Copilot 服务端点及请求来源头。以上流程已通过离线模拟测试，未验证真实账号授权或订阅可用性。

### 添加自定义 API 模型（Add custom API model）

`/login` 菜单第三项「Add custom API model」（或 `/login custom`）在增强终端中打开单页表单：API 格式、Provider ID、Base URL、API Key、模型 ID、显示名称、上下文窗口、最大输出 token、图片输入、reasoning 以及「立即使用」「设为未来启动的默认模型」同页可见。Tab/Shift+Tab 切换焦点，Esc 取消，Ctrl+S 或「Save model」写入用户目录的 `models.json` 与 `auth.json`。字段错误在失焦或保存时显示；完整端点建议需要手动应用；非本机 HTTP 需要勾选确认。终端至少 60×18 时表单可滚动完成，更小时保留草稿并提示调整尺寸。纯文本 REPL 与 `python -m pi --login custom` 仍使用逐项向导。保存后模型立即出现在 `/model` 选择器中，重启后可用；项目 `.pi/settings.json` 或 `--provider`/`--model` 参数仍会覆盖用户默认值。非交互 stdin 下 `--login custom` 直接报错，不读取管道内容。

三种格式的地址规则（向导只追加路径后缀，不自动补 `/v1`）：

| 格式 | Base URL 示例 | 最终请求地址 |
| --- | --- | --- |
| OpenAI Chat Completions | `https://gateway.example.com/v1` | `https://gateway.example.com/v1/chat/completions` |
| OpenAI Responses | `https://gateway.example.com/v1` | `https://gateway.example.com/v1/responses` |
| Claude Messages | `https://gateway.example.com` | `https://gateway.example.com/v1/messages` |
| Claude Messages | `https://gateway.example.com/anthropic` | `https://gateway.example.com/anthropic/v1/messages` |

粘贴完整端点时表单与向导会识别三种已知后缀并建议根地址；后缀与所选协议不匹配时要求重新选择协议或地址，不静默改写协议。Provider ID 代表「端点 + 协议 + 凭据」的组合，须匹配 `^[a-z][a-z0-9-]{0,63}$`，不能为保留字 `custom`，也不能与现有 provider 重名；同一网关换用不同协议时应建立不同 ID（如 `gateway-chat` 与 `gateway-responses`）。允许本地 `http://` 地址；非本机 HTTP 地址需要确认「API Key 将通过未加密连接发送」。上下文窗口与最大输出 token 只是客户端配置，不代表服务端真实能力，应按服务文档修改；成本不要求填写，界面不会把未配置成本描述成免费。

保存成功仅表示配置有效并已落盘，连接尚未验证，首条真实 prompt 才发出模型请求。`/logout <id>` 只清除凭据，模型配置保留；之后可通过 `/login <id>` 重新录入密钥恢复调用。写入过程采用快照校验加有限回滚：保存前其他进程改动了配置会要求重新确认，写入或 reload 失败时撤销本次新增条目，回滚失败会明确提示需要手工修复的文件。若此前异常退出留下了孤立密钥（`auth.json` 有记录但 provider 不存在），向导会拒绝同 ID 并提示先 `/logout <id>` 清理。

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

启动：`python -m pi --provider local --model coder`。支持的 API 为「原生模型协议」列出的全部 API 加测试用 `faux`。同名 provider/model 条目会覆盖对应元数据；其他内置模型仍保留。模型输入类型支持 `text`、`image`，价格单位为美元/百万 token；元数据应与上游实际能力一致。

provider 还可设置 `api_key`、`api_key_env`、`headers`。`api_key` 和 header 值支持 `$NAME` / `${NAME}` 环境变量插值，不执行 shell 命令。标准设备码服务需添加 `oauth` 对象，包含 HTTPS `device_url`、`token_url`、`client_id` 和可选 `scope`；生成的 access token 必须能直接用于该兼容端点的 Bearer 认证。

交互命令：

```text
/models                         列出模型与能力参数
/model openai/gpt-4o             切换 provider 和模型
/model gpt-4o-mini               在当前 provider 内切换
/login                          交互选择登录方式与 provider（含自定义 API 模型）
/login custom                   打开自定义 API 模型表单
/login openai                   保存 API key
/login openrouter oauth         OAuth 登录
/logout openrouter              移除本地凭据
/sync-models                    刷新在线模型目录缓存
/settings                       查看合并后的默认设置
/settings thinking "high"       保存用户默认值
/reload                         重载配置、目录及当前模型
```

`/settings` 保存用户默认值，下次启动生效；`/reload` 可立即应用 thinking 和压缩配置。项目同名设置仍优先。模型目录的构成与在线同步见上文「模型目录同步」。

会话文件写在 `./.pi/sessions/*.jsonl`。启动时可指定 `--session <路径或ID>`、`--session-dir <目录>`、`--name <名称>`；`-c` 恢复最近会话。

新记录通过 `id`、`parent_id` 连接成树，另以追加记录保存活动节点和名称。切换分支不会删除原分支，重启后保留最后选择的节点；只有活动分支会送入模型。节点 ID 支持唯一前缀。为避免不完整的工具协议，不能切换或复制到尚缺工具结果的节点。

`/fork` 不带 ID 时复制当前分支；指定 ID 时复制至该消息之后，可在新会话中继续提问。当前为命令行文本操作，不含 TS 版的交互选择器、回填编辑器或离开分支时的自动摘要。JSONL 为 Python 自有格式，可重新加载，不支持读取 TS 会话文件；`/export-ts <path.jsonl>` 可将活动分支导出为 TS pi 会话格式。HTML 为转义后的静态文本。导出不会覆盖已有文件。

## 评测系统（documentation-lift evals）

`packages/evals` 的 Python 移植版位于 `pi/evals/`（运行器代码）与两个仓库目录：`evals/`（评测套件与夹具）和 `docker/`（`Dockerfile` + `entrypoint.py`，容器侧）。TS 版依赖 Vitest 与 `vitest-evals`；Python 版用 `pi.evals.framework` 本地实现同等的最小契约（Harness 适配器、运行规范化、结构化输出裁判、`describe_eval` 注册、用例运行器），报告仍写 Vitest JSON 结构（`testResults[].assertionResults[].meta`），因此 `report.ts` 的读取与配对逻辑按原样移植。

评测套件与 TS 版对应：`evals/smoke.eval.py` 与 `evals/documentation_audit.eval.py` 是 host 评测（非 docs 套件），`evals/examples.docs.eval.py` 是文档对比用例；`evals/acme_server.py`（进程内假 provider 服务器，openai/stream 两种协议形态与探针标志）与 `evals/configured_runtime.py`（重载后用隔离 agent 目录的模型配置发起真实补全的探针，错误为结构化输出）是 provider/model 类评测的共享夹具，单元测试与未来 provider docs 套件共用。`describe_eval` 支持 `title` 参数，供一文件多用例的参数化套件（对应 TS 的 `it.for`）使用。

运行一次文档对比：

```bash
python -m pi.evals.cli --provider openai --model gpt-4o-mini
# 等价环境变量：PI_PROVIDER / PI_MODEL；重复次数：--runs-per-variant N 或 PI_EVAL_RUNS_PER_VARIANT
# 用例过滤：-t "adds the model" 或 --testNamePattern=...
```

流程与 TS 版一致：构建 `without_docs` / `with_docs` 两个镜像 → 在两个容器内发现用例并要求完全一致 → 按 (case, variant, run) 计划任务（奇数轮先 without、偶数轮先 with，降低顺序偏差）→ 每个任务独立容器运行 → 写入 `protocol.json`、`expected-runs.json`、`observations.jsonl` → 读取观察、成对配对、产出 `report.json` / `report.txt`。成对 lift 只在两臂都恰好产生一个分数时计入；缺失、重复、跳过、未计分或出错都会阻塞该对，任一对被阻塞时隐藏标题通过率并退出码 1。结果目录为忽略的 `.eval/<时间戳>_<id>/`，其中 `evals/<variant>/sessions/<sha256>/session.jsonl` 保存原生会话快照。

host 评测（非 `*.docs.eval` 的普通套件）在宿主机直接运行，对应 TS 的 `npm run eval:host`：

```bash
set PI_PROVIDER=openai
set PI_MODEL=gpt-4o-mini
python -m pi.evals.host                              # 发现并运行全部 host 套件
python -m pi.evals.host evals/smoke.eval.py          # 指定套件
python -m pi.evals.host -t "matches the implementation" --report out/vitest.json
```

运行器顺序执行每个用例（对应 TS 的 `maxWorkers: 1`），聚合输出 Vitest JSON 报告（`--report` 落盘），任一用例失败退出码 1。host 套件不做结构化裁判打分，直接用断言判分（与 TS 相同）。

评测 harness（`create_pi_coding_agent_harness`）在临时根目录内组装与 CLI 相同的 Agent：隔离 HOME 与 agent 目录、注入宿主 `auth.json` 凭据、按名单选工具（文档评测固定为 `read/write/edit/grep/find/ls`，不暴露 shell 与联网搜索）、录制 JSONL 会话、统计 token/工具调用/成本，并在 POSIX 上按 `PI_EVAL_SANDBOX_UID/GID` 永久降权后再提示模型。输入步骤支持 `{"type": "reload"}`：Python Agent 没有可重建的扩展运行器，reload 重新读取隔离 agent 目录的模型配置（models.json/auth.json），刷新后的 registry 传给 `output` 判分钩子（异步钩子会被等待）；配置损坏不使运行失败，registry 保留上次成功状态，由探针以结构化错误上报。评测失败时把部分运行数据（事件、用量、错误）附在异常上，报告据此保留部分指标。

系统提示词现在包含 `<docs>` 一节（PI 文档路由：README/docs/examples 路径与阅读指引），`SystemPromptOptions.include_pi_documentation` 控制其出现：默认自动探测包内文档是否存在；评测 harness 强制包含并在 `without_docs` 变体中剥离该节（`exclude_pi_documentation`），配合镜像内删除文档文件构成对照实验。`<cwd>` 一节同步改为 `<cwd>\n路径\n</cwd>` 的分节格式。

与 TS 版的实现差异（同等功能的替换实现）：容器内不再运行 Vitest，而是 `docker/entrypoint.py` 直接发现/运行用例并写 Vitest JSON；host 评测运行器（`pi.evals.host`）取代 Vitest host project，报告结构一致但不经 Vitest；TS 的 Vite transformed-module 保护无对应物，改由镜像内 `chmod go-rwx` 保证评测源码仅 root 可读；`reload` 只重读模型配置（Python 会话无扩展重载）；acme/configured-runtime 夹具的 models.json 与探针沿用 Python 的 snake_case 键（`context_window`/`max_tokens`/`costs`），`configured_runtime` 对模型快照与 judge 期望保持 TS 的 camelCase 形状，models.json 解析失败的消息为 Python 的 `Cannot read configuration`（TS 为 `Failed to parse models.json`）；npm tarball 打包换为容器内 `pip install /repo/pi-python`，并把 README 复制到 site-packages 作为 `with_docs` 的文档。凭据来源探测在降权时按环境变量名近似（`PROVIDER_API_KEY` / `PI_PY_API_KEY`），TS 版使用 ModelRuntime 的精确 source。

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

## 持久化运行时（pi.durable）

`pi/durable/` 是 `packages/durable` 核心的 Python 移植：会话/任务/文档先提交后可见（`pi.durable.session`、`pi.durable.harness`、`pi.durable.storage`）。此外还移植了三块：

| Python 模块 | TS 源 | 内容 |
| --- | --- | --- |
| `pi.durable.env` | `packages/durable/src/env` | 可移植执行环境契约（`ExecutionEnv`：文件系统 + shell + watch，`Result` 化错误）与本地实现 `LocalExecutionEnv`（asyncio + os，Windows 打开文件带 FILE_SHARE_DELETE，支持打开期间改名）；流式 UTF-8 解码（BOM/非法序列与一次性解码一致）、有界内存行扫描 `LineScanner`、`LocalFileWatcher`（`env/node-watch.ts` 的移植：原生事件只触发防抖重扫，上报 = 快照 diff ∪ 事件路径，原生失败落轮询并报 `overflow`；后端为标准库 ctypes——Windows `ReadDirectoryChangesW`、Linux inotify、macOS FSEvents。与 TS 的偏离：Windows 默认 native，句柄只开在 target 的最近存在祖先目录上（TS 因句柄阻塞祖先改名而在 Windows 轮询），另配 2s 重扫定时器自愈覆盖；`watch_mode="auto"/"native"/"polling"` 可强制） |
| `pi.durable.tools` | `packages/durable/src/tools` | 持久化编码工具 `read`/`write`/`edit`/`bash`：跑在 `ExecutionEnv` 上而非宿主进程；read 截断走诊断（内容只有文件文本）、edit 精确唯一匹配（模糊归一 + 未改动行保留 + BOM/CRLF 保留）、bash 流式输出 + 尾窗口（跳过计数精确）+ 溢出落盘 + 超时/中止进程树终结；同文件 `edit`/`write` 经变更队列串行化 |
| `pi.durable.testing` | `packages/durable/src/testing` | 一致性套件：`create_storage_conformance`（23 个用例 × 任意后端）、`create_env_conformance`（21 个用例 × 任意 ExecutionEnv）、`createExpectAssertions` 对应的 `ConformanceAssertions`、存储基准（`seed_storage_benchmark`/`measure_*_benchmarks`）与 pytest 适配（`runner.py`）；套件与测试运行器无关，`tests/test_durable_*_conformance.py` 驱动 |

移植偏差：文档内容按完整版本存储（base 或 delta）；delta 操作批次（chord `r/s/d/a/t/p/m`，见 `pi.durable.delta`）在物化时重放到基准上，与 TS 语义一致。读取一律返回脱离存储的副本（调用方改动不影响存储），ID 分配在 2^53-1 耗尽时报错。

```python
from pi.durable.env import LocalExecutionEnv, Context
from pi.durable.tools import create_read_tool

env = LocalExecutionEnv(cwd=".")

class Api:  # 最小 ToolExecutionApi
    env = env
    output_window = None
    def output(self, chunk, skipped=None): print(chunk, end="")
    def diagnostic(self, diagnostic): print(diagnostic.message, file=__import__("sys").stderr)

result = __import__("asyncio").run(create_read_tool().execute({"path": "README.md", "limit": 5}, Api(), Context()))
```

## 测试

```bash
pip install pytest
python -m pytest tests -q
```

测试全部离线：provider 层用 canned SSE 通过 `httpx` mock 传输验证事件映射；agent loop 用 `faux` 脚本驱动；工具层直接操作临时目录。

## 项目信任、崩溃报告与诊断

启动时若项目目录（或其祖先）存在 `.pi/` 下的 `settings.json`、`mcp.json`、`extensions`、`skills`、`prompts`、`themes`、`SYSTEM.md`、`APPEND_SYSTEM.md`，或存在 `.agents/skills`（用户主目录的 `~/.agents/skills` 除外），会先要求项目信任决策：

- 决策保存在用户目录 `trust.json`（`{规范化路径: true/false}`），沿父目录就近生效；`null` 删除条目。`/trust` 打开决策选择器，`--approve` / `--no-approve`（`-a` / `-na`）为单次运行覆盖。
- settings 的 `default_project_trust`（`ask`/`always`/`never`，默认 `ask`）控制默认行为；该值只读用户配置，项目配置无法用它跳过询问。
- 未信任时跳过项目 `.pi` 下的设置、MCP、skills、prompts 与扩展，只保留用户目录资源；非交互模式（print/json/rpc）无法询问时同样按未信任处理并输出 warning。

其他应用层行为：

- 未捕获异常写入用户目录 `crashes.json`（最多 5 条、7 天有效）；下次交互启动提示一次。`/bug` 成功提交后清空。
- `/bug [hint]` 生成 bug 报告：环境与模型元数据（`api_key`、`token`、`password`、`authorization` 等敏感键及 URL 凭据/查询参数自动脱敏）、会话中失败回合的诊断；可选附带会话 JSONL，或在不含 transcript 时由当前模型生成 Markdown 摘要。交付方式为上传到 Radius 网关（`/v1/bug-reports`，匿名或带已保存的 Radius 凭据）或导出 zip 到当前目录；成功后写入 `pi.bug-report` 会话条目。`PI_OFFLINE=1` 时仅可导出 zip。
- JSON 与 RPC 模式接管 stdout：杂散 `print`（扩展、第三方库）转发到 stderr，协议输出始终写原始 stdout。
- OpenRouter、NVIDIA NIM、Cloudflare、OpenCode 的客户端归属请求头自动合并进模型请求；开关为 settings 的 `enable_install_telemetry`（默认开启），环境变量 `PI_TELEMETRY` 优先。
- `PI_TIMING=1` 在退出前于 stderr 打印启动耗时分组（解析、registry、agent、session、extensions）。

## 上下文压缩

Agent 和 CLI 默认启用自动上下文压缩：接近模型上下文窗口时，摘要旧消息并保留近期消息、系统指令和工具声明。切分不会拆开工具调用与其结果。明确的上下文溢出错误会触发强制压缩并重试一次；其他错误不会触发重试。摘要失败、为空或达到输出上限时不会替换已有摘要。

原始会话 JSONL 保持完整。摘要与压缩切点（checkpoint：已消费消息数 + 末条消息指纹）一起持久化为会话 `compaction` 条目；恢复会话时按 checkpoint 从加载的 transcript 重建切点，校验通过则直接续用，不再重复投影或重新产生摘要调用费用。checkpoint 与 transcript 不匹配（例如切到更早分支）时清空摘要与切点，与运行中的自愈行为一致；旧版本会话文件只有摘要、没有 checkpoint，恢复摘要但切点未知，行为同修复前。单条过大的最新输入、系统提示词或不可拆分工具结果仍可能超过窗口，需要缩短输入。

库调用可通过 `Agent(..., compaction=CompactionSettings(...))` 配置；从 `pi.agent.compaction` 导入 `CompactionSettings`。默认 `reserve_tokens=16384`、`keep_recent_tokens=20000`、`summary_tokens=2048`，可设置 `enabled=False` 关闭。未知模型应准确设置 `model.context_window`。Token 数使用 UTF-8 长度估算，并非模型的精确分词结果。

## 尚未移植

目标是 1:1 复刻，以下为与 TS 版的剩余差距：

- TS TUI 的通用组件布局（转录区内任意组件的横向组合、焦点栈与滚动视口）；Python 已有全屏对话布局（`--tui fullscreen`）、多行编辑、补全、选择器、主题、快捷键、ANSI Markdown、工具输出折叠、终端图片显示（Kitty/iTerm2 协议，全屏模式为半块字符）、扩展 `ctx.ui.custom` 交互视图（按键/鼠标转发、overlay 几何与 handle）、widget/header/footer 组件工厂及 `setEditorComponent` 编辑器接管
- prompt cache 预热（TS CacheWarmer：按 TTL 以 1 token 输出重放请求）
- 其余 provider 的专属认证流程；已实现 Claude、Codex、Copilot、Kimi、xAI、Radius 登录，OpenAI 的 Sign in with ChatGPT、OpenRouter PKCE 及可配置标准设备码 OAuth
- models.dev 在线目录仅在显式触发（`--sync-models` / `/sync-models` / `sync_models`）时刷新，Radius 发现在启动或登录后刷新；TS 版每次启动都会刷新目录。未收录的模型 id 以默认参数运行
