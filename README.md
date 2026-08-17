# LangChain Skill Runtime

[English](README_EN.md) | 简体中文

`langchain-skill-runtime` 是一个不依赖数据库的 Python 运行时：它把标准
`SKILL.md` 或业务系统提供的结构化 Skill/Tool 对象，编译成可直接交给
LangChain 使用的系统提示词和 `BaseTool` 工具集合。

```text
SKILL.md / 结构化对象 / Repository
                │
                ▼
           SkillRuntime
                │
                ├── system_prompt
                └── tuple[BaseTool, ...]
```

当前版本：`0.3.4`。项目仍处于 Alpha 阶段，公开 API 会遵循语义化版本进行演进。

## 核心能力

- 同时支持独立 `SKILL.md`、结构化对象和 Repository 三种输入来源。
- 一个 Skill 可以组合 Python Function、服务端脚本、客户端工具和 MCP 工具。
- 一个 `MCP` 类型对象可通过 `tools/list` 动态展开成 MCP Server 的全部工具。
- 通过 Transport 将工具调用委派给浏览器、桌面端或移动端，并按 `call_id` 接收结果。
- 输入、输出均可使用 JSON Schema 校验。
- MCP 凭据只通过 `env` 或 `secret_ref` 解析，避免在 `SKILL.md` 中保存明文密钥。
- 核心依赖保持精简；MCP 和 LangChain Agent 集成均为可选依赖。
- 提供 `py.typed`，支持类型检查。

## 安装

Python 版本要求：`>=3.11`。

在发布到 PyPI 前，可以直接从 GitHub 安装：

```bash
# 核心运行时
uv add 'langchain-skill-runtime @ git+https://github.com/zhouqingsheng-github/langchain-skill-runtime.git'

# 使用 MCP
uv add 'langchain-skill-runtime[mcp] @ git+https://github.com/zhouqingsheng-github/langchain-skill-runtime.git'

# 同时使用 LangChain Agent 和 MCP
uv add 'langchain-skill-runtime[agent,mcp] @ git+https://github.com/zhouqingsheng-github/langchain-skill-runtime.git'
```

发布到 PyPI 后，对应命令将简化为 `uv add langchain-skill-runtime`，并按需添加
`[mcp]` 或 `[agent,mcp]`。

## 快速开始

创建 `skills/calculator/SKILL.md`：

```yaml
---
id: calculator
name: calculator
description: 计算两个整数之和
version: 1.0.0
tools:
  - id: add-numbers
    name: add_numbers
    description: 计算两个整数之和
    type: PYTHON_FUNCTION
    input_schema:
      type: object
      properties:
        a: {type: integer}
        b: {type: integer}
      required: [a, b]
      additionalProperties: false
    output_schema:
      type: integer
    execution:
      registry_key: math.add
---

# 计算器

用户要求计算两个整数之和时，调用 `add_numbers`。
```

在宿主项目中注册函数并编译：

```python
import asyncio

from langchain_skill_runtime import SkillRuntime
from langchain_skill_runtime.adapters import PythonFunctionAdapter, ToolFactory
from langchain_skill_runtime.executors import InMemoryFunctionRegistry


async def add_numbers(a: int, b: int) -> int:
    return a + b


async def main() -> None:
    registry = InMemoryFunctionRegistry()
    registry.register("math.add", add_numbers)

    runtime = SkillRuntime(tool_factory=ToolFactory([PythonFunctionAdapter(registry)]))
    bundle = await runtime.compile_file("skills/calculator/SKILL.md")

    print(bundle.system_prompt)
    print(await bundle.tools[0].ainvoke({"a": 18, "b": 24}))  # 42


asyncio.run(main())
```

`registry_key` 不是给大模型看的工具名称，而是 `SKILL.md` 与宿主 Python
函数注册表之间的稳定绑定键。完整代码见
[计算器示例](examples/calculator)。

## 编译结果

三个编译入口都返回 `SkillBundle`：

- `bundle.system_prompt`：由 Skill 指令和实际可用工具编译出的系统提示词。
- `bundle.tools`：可直接调用或交给 LangChain Agent 的 `BaseTool` 元组。
- `bundle.diagnostics`：非致命诊断，例如可选工具不可用。
- `bundle.fingerprint`：Skill、版本和最终工具集合的稳定指纹。

## 三种 Skill 来源

### 1. 独立 `SKILL.md`

无需 Repository 或数据库：

```python
bundle = await runtime.compile_file("skills/navigation/SKILL.md")
```

文件的 YAML Frontmatter 描述身份、版本和工具；Markdown 正文描述 Skill 行为。

### 2. 结构化对象

适合已经从数据库、配置中心或业务 API 取得数据的系统：

```python
bundle = await runtime.compile_objects(skill_definition, resolved_tools, context)
```

库只接收 `SkillDefinition` 和 `ResolvedToolDefinition`，不关心数据原来存在哪里。

### 3. Repository

实现异步 `SkillRepository` 和 `ToolRepository` 协议后，可以按 Skill ID 编译：

```python
runtime = SkillRuntime(
    skill_repository=my_skill_repository,
    tool_repository=my_tool_repository,
    tool_factory=tool_factory,
)
bundle = await runtime.compile("navigation", context)
```

Repository 是可替换协议，不是内置数据库依赖。

## 支持的工具类型

| `type` | 定位方式 | 适配器 | 执行位置 |
| --- | --- | --- | --- |
| `PYTHON_FUNCTION` | `execution.registry_key` | `PythonFunctionAdapter` | 当前 Python 进程 |
| `SERVER_SCRIPT` | `execution.entry` 或 `artifact_id` | `ServerScriptAdapter` | 宿主管理的脚本执行器 |
| `CLIENT_JAVASCRIPT` | `execution.tool_key` | `ClientJavascriptAdapter` | 浏览器、桌面端或移动端 |
| `MCP` | `execution.server` 或 `server_ref` | `McpToolAdapter` | 标准 MCP Server |

`ToolFactory` 按 `type` 选择适配器。宿主只需注册实际会用到的适配器：

```python
tool_factory = ToolFactory(
    [
        PythonFunctionAdapter(function_registry),
        ClientJavascriptAdapter(client_transport),
        McpToolAdapter(mcp_provider),
    ]
)
```

## MCP 工具集

以下示例同时使用 MCP 工具和 LangChain Agent，因此安装两个可选依赖：

```bash
uv add 'langchain-skill-runtime[agent,mcp] @ git+https://github.com/zhouqingsheng-github/langchain-skill-runtime.git'
```

在 `SKILL.md` 中声明一个 `MCP` 对象且不写 `tool_name`，表示使用该 MCP
Server 暴露的全部工具，而不是单个固定工具：

```yaml
tools:
  - id: maps-mcp-server
    name: maps_mcp_server
    description: 地图 MCP Server 提供的全部工具
    type: MCP
    input_schema:
      type: object
      properties: {}
      additionalProperties: false
    execution:
      server_name: maps
      server:
        transport: streamable_http
        url: https://mcp.example.com/mcp
        headers:
          Authorization:
            env: MAPS_MCP_AUTHORIZATION
```

宿主必须明确允许内联远程地址：

```python
from langchain.agents import create_agent

from langchain_skill_runtime import SkillRuntime
from langchain_skill_runtime.adapters import (
    AllowHostsMcpUrlPolicy,
    LangChainMcpToolProvider,
    McpToolAdapter,
    ToolFactory,
)

async with LangChainMcpToolProvider(
    url_policy=AllowHostsMcpUrlPolicy({"mcp.example.com"})
) as provider:
    runtime = SkillRuntime(tool_factory=ToolFactory([McpToolAdapter(provider)]))
    bundle = await runtime.compile_file("skills/maps/SKILL.md")

    # 名称来自 MCP tools/list 的真实结果。
    print([tool.name for tool in bundle.tools])

    # 编译和多步工具调用都在同一 Provider 作用域内完成。
    agent = create_agent(
        model=model,
        tools=list(bundle.tools),
        system_prompt=bundle.system_prompt,
    )
    result = await agent.ainvoke(request)
```

需要多步操作且依赖 Server 状态的 MCP，应把编译和整次调用放在
`async with LangChainMcpToolProvider(...)` 作用域内。同一 `server_name`
的工具会共享同一 MCP Session，退出作用域时会统一释放。无状态单次调用
仍可以不使用显式作用域，保持原有行为。

也可以使用 `PublicHttpsMcpUrlPolicy`，要求地址使用 HTTPS 且所有 DNS 解析结果
都是公网地址。生产系统若把 MCP 配置集中治理，可在 `SKILL.md` 中使用
`server_ref`，并向 `LangChainMcpToolProvider` 提供自己的
`McpServerConfigProvider`。

`env` 从当前进程环境变量读取；`secret_ref` 由宿主实现的 `SecretProvider`
解析。敏感 Header、URL 参数或环境变量不得直接写明文。完整结构见
[MCP 工具集示例](examples/mcp_tool_collection)。

## 客户端工具与结果回传

`CLIENT_JAVASCRIPT` 适用于文件选择、浏览器 API、桌面端能力等只能在客户端
执行的工具。库负责请求建模、超时、版本校验和 `call_id` 关联，不接管你的
WebSocket 或消息队列。

```text
BaseTool.ainvoke()
        │
        ▼
PendingClientToolTransport ──发送 ClientToolRequest──► 客户端
        ▲                                              │
        └──── accept_result(ClientToolResult) ◄────────┘
```

编译时通过 `CompileContext` 声明当前会话及客户端能力：

```python
context = CompileContext(
    session_id="session-1",
    client_capabilities=(
        ClientCapability(tool_id="client.export.file", version="1.0.0"),
    ),
)
bundle = await runtime.compile_file("skills/export/SKILL.md", context)
```

发送回调把 `ClientToolRequest` 交给业务 Transport；收到客户端消息后，将其
转换为 `ClientToolResult` 并调用 `await transport.accept_result(result)`。
完整的内存模拟流程见[客户端工具示例](examples/client_tool)。

## 接入 LangChain Agent

安装 `agent` 可选依赖后，直接使用编译结果：

```python
from langchain.agents import create_agent

bundle = await runtime.compile_file("skills/navigation/SKILL.md")
agent = create_agent(
    model=model,
    tools=list(bundle.tools),
    system_prompt=bundle.system_prompt,
)
```

本库不绑定具体大模型供应商，模型初始化、鉴权和调用策略由宿主项目负责。

## 安全边界

- `SKILL.md` 是声明和指令，不应包含 Token、Cookie、密码或 API Key 明文。
- 远程 MCP 内联配置必须由宿主提供 `McpUrlPolicy`；默认策略不会静默信任任意地址。
- `SERVER_SCRIPT` 是否允许执行、如何隔离、资源限制和审计由宿主执行器负责。
- 客户端工具会校验会话、Tool ID 和版本；业务 Transport 仍需负责认证和连接权限。
- Repository、Secret Provider 和执行器都是宿主注入的协议，本库不会自行连接数据库或密钥系统。
- 不要把不可信 `SKILL.md` 直接视为已授权配置；加载前应在业务层完成来源和权限校验。

## 项目结构

```text
src/langchain_skill_runtime/
├── adapters/              # 四类 Tool 适配器与 ToolFactory
│   └── mcp/               # MCP 协议、Provider、URL 策略和适配器
├── client/                # 客户端请求、结果与 pending call 管理
├── executors/             # Function/Script 执行协议及内存实现
├── models/                # Skill、Tool、Context、Bundle 模型
├── parsing/               # SKILL.md 加载与解析
├── prompting/             # 系统提示词编译
├── repositories/          # 数据来源协议
└── runtime/               # SkillRuntime 编排
examples/                  # 面向使用者的最小示例
tests/                     # 单元测试和集成测试，不作为公开 API 文档
doc/                       # 中文设计与使用文档
```

## 本地开发

```bash
git clone https://github.com/zhouqingsheng-github/langchain-skill-runtime.git
cd langchain-skill-runtime
uv sync --all-extras

uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests examples
uv run mypy src
uv build
```

运行公开示例：

```bash
uv run python examples/calculator/main.py
uv run python examples/client_tool/main.py
```

MCP 示例中的域名和环境变量是占位配置；运行前请换成你被授权使用的真实
MCP Server，并配置对应凭据。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源。
