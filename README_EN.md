# LangChain Skill Runtime

English | [简体中文](README.md)

`langchain-skill-runtime` is a database-independent Python runtime that compiles
standard `SKILL.md` files or structured Skill/Tool objects into a system prompt
and a collection of LangChain `BaseTool` instances.

```text
SKILL.md / structured objects / repositories
                    │
                    ▼
               SkillRuntime
                    │
                    ├── system_prompt
                    └── tuple[BaseTool, ...]
```

Current version: `0.3.3`. The project is currently Alpha software; public APIs
will evolve under semantic versioning.

## Features

- Compile standalone `SKILL.md` files, structured objects, or repository-backed data.
- Combine Python functions, server scripts, client tools, and MCP tools in one Skill.
- Expand one `MCP` declaration into every tool returned by the server's `tools/list`.
- Delegate tool calls to browser, desktop, or mobile clients and correlate results by `call_id`.
- Validate inputs and outputs with JSON Schema.
- Resolve MCP credentials through `env` or `secret_ref`, never plaintext secrets in `SKILL.md`.
- Keep the core dependency set small; MCP and LangChain Agent support are optional extras.
- Ship a `py.typed` marker for type checkers.

## Installation

Python `>=3.11` is required.

Until the package is published on PyPI, install it directly from GitHub:

```bash
# Core runtime
uv add 'langchain-skill-runtime @ git+https://github.com/zhouqingsheng-github/langchain-skill-runtime.git'

# MCP support
uv add 'langchain-skill-runtime[mcp] @ git+https://github.com/zhouqingsheng-github/langchain-skill-runtime.git'

# LangChain Agent and MCP support
uv add 'langchain-skill-runtime[agent,mcp] @ git+https://github.com/zhouqingsheng-github/langchain-skill-runtime.git'
```

After a PyPI release, the core command becomes `uv add langchain-skill-runtime`,
with `[mcp]` or `[agent,mcp]` added as needed.

## Quick start

Create `skills/calculator/SKILL.md`:

```yaml
---
id: calculator
name: calculator
description: Add two integers
version: 1.0.0
tools:
  - id: add-numbers
    name: add_numbers
    description: Add two integers
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

# Calculator

Call `add_numbers` when the user asks to add two integers.
```

Register the host function and compile the Skill:

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

    runtime = SkillRuntime(
        tool_factory=ToolFactory([PythonFunctionAdapter(registry)])
    )
    bundle = await runtime.compile_file("skills/calculator/SKILL.md")

    print(bundle.system_prompt)
    print(await bundle.tools[0].ainvoke({"a": 18, "b": 24}))  # 42


asyncio.run(main())
```

`registry_key` is not the tool name shown to the model. It is the stable binding
between a `SKILL.md` declaration and the host application's Python function
registry. See the complete [calculator example](examples/calculator).

## Compilation result

All three compilation entry points return a `SkillBundle`:

- `bundle.system_prompt`: the system prompt compiled from the Skill instructions and available tools.
- `bundle.tools`: a tuple of `BaseTool` instances ready for direct invocation or a LangChain Agent.
- `bundle.diagnostics`: non-fatal diagnostics, such as an unavailable optional tool.
- `bundle.fingerprint`: a stable fingerprint of the Skill, version, and final tool collection.

## Three Skill sources

### 1. Standalone `SKILL.md`

No repository or database is required:

```python
bundle = await runtime.compile_file("skills/navigation/SKILL.md")
```

YAML Frontmatter declares identity, version, and tools. The Markdown body contains
the Skill instructions.

### 2. Structured objects

Use objects when your application already loads data from a database, configuration
service, or business API:

```python
bundle = await runtime.compile_objects(skill_definition, resolved_tools, context)
```

The runtime only consumes `SkillDefinition` and `ResolvedToolDefinition`; it does
not care where those objects came from.

### 3. Repositories

Implement the asynchronous `SkillRepository` and `ToolRepository` protocols to
compile a Skill by ID:

```python
runtime = SkillRuntime(
    skill_repository=my_skill_repository,
    tool_repository=my_tool_repository,
    tool_factory=tool_factory,
)
bundle = await runtime.compile("navigation", context)
```

Repositories are replaceable protocols, not a built-in database dependency.

## Supported tool types

| `type` | Binding | Adapter | Execution location |
| --- | --- | --- | --- |
| `PYTHON_FUNCTION` | `execution.registry_key` | `PythonFunctionAdapter` | Current Python process |
| `SERVER_SCRIPT` | `execution.entry` or `artifact_id` | `ServerScriptAdapter` | Host-managed script executor |
| `CLIENT_JAVASCRIPT` | `execution.tool_key` | `ClientJavascriptAdapter` | Browser, desktop, or mobile client |
| `MCP` | `execution.server` or `server_ref` | `McpToolAdapter` | Standard MCP Server |

`ToolFactory` selects an adapter by `type`. A host only registers the adapters it uses:

```python
tool_factory = ToolFactory(
    [
        PythonFunctionAdapter(function_registry),
        ClientJavascriptAdapter(client_transport),
        McpToolAdapter(mcp_provider),
    ]
)
```

## MCP tool collections

Install the MCP extra:

```bash
uv add 'langchain-skill-runtime[mcp] @ git+https://github.com/zhouqingsheng-github/langchain-skill-runtime.git'
```

An `MCP` declaration without `tool_name` represents every tool exposed by the
server, rather than one fixed tool:

```yaml
tools:
  - id: maps-mcp-server
    name: maps_mcp_server
    description: Every tool exposed by the maps MCP Server
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

The host must explicitly authorize an inline remote address:

```python
from langchain_skill_runtime import SkillRuntime
from langchain_skill_runtime.adapters import (
    AllowHostsMcpUrlPolicy,
    LangChainMcpToolProvider,
    McpToolAdapter,
    ToolFactory,
)

provider = LangChainMcpToolProvider(
    url_policy=AllowHostsMcpUrlPolicy({"mcp.example.com"})
)
runtime = SkillRuntime(tool_factory=ToolFactory([McpToolAdapter(provider)]))
bundle = await runtime.compile_file("skills/maps/SKILL.md")

# Names come from the real MCP tools/list response.
print([tool.name for tool in bundle.tools])
```

`PublicHttpsMcpUrlPolicy` is also available and requires HTTPS with exclusively
public DNS results. Applications with centrally governed MCP configuration can
use `server_ref` in `SKILL.md` and provide a custom `McpServerConfigProvider` to
`LangChainMcpToolProvider`.

`env` reads from the current process environment. `secret_ref` is resolved by a
host-provided `SecretProvider`. Sensitive headers, URL parameters, and environment
variables cannot contain plaintext credentials. See the complete
[MCP tool collection example](examples/mcp_tool_collection).

## Client tools and returned results

`CLIENT_JAVASCRIPT` supports capabilities that must execute in a browser, desktop
application, or mobile client. The runtime handles request models, timeouts,
version checks, and `call_id` correlation; it does not own your WebSocket or queue.

```text
BaseTool.ainvoke()
        │
        ▼
PendingClientToolTransport ──send ClientToolRequest──► client
        ▲                                             │
        └──── accept_result(ClientToolResult) ◄───────┘
```

Declare the current session and client capabilities in `CompileContext`:

```python
context = CompileContext(
    session_id="session-1",
    client_capabilities=(
        ClientCapability(tool_id="client.export.file", version="1.0.0"),
    ),
)
bundle = await runtime.compile_file("skills/export/SKILL.md", context)
```

The sender callback forwards `ClientToolRequest` to your business transport. When
a client message arrives, convert it into `ClientToolResult` and call
`await transport.accept_result(result)`. See the complete
[client tool example](examples/client_tool).

## LangChain Agent integration

After installing the `agent` extra, use the compiled result directly:

```python
from langchain.agents import create_agent

bundle = await runtime.compile_file("skills/navigation/SKILL.md")
agent = create_agent(
    model=model,
    tools=list(bundle.tools),
    system_prompt=bundle.system_prompt,
)
```

The runtime does not bind to a model provider. Model initialization,
authentication, and invocation policy belong to the host application.

## Security boundaries

- Treat `SKILL.md` as declarations and instructions; never store plaintext tokens, cookies, passwords, or API keys in it.
- Inline remote MCP configuration requires a host-provided `McpUrlPolicy`; arbitrary addresses are not silently trusted.
- The host script executor owns execution authorization, isolation, resource limits, and auditing for `SERVER_SCRIPT`.
- Client tools validate session, tool ID, and version; the business transport must still enforce authentication and connection permissions.
- Repositories, secret providers, and executors are host-injected protocols. The library does not connect to a database or secret store by itself.
- Do not treat an untrusted `SKILL.md` as authorized configuration. Validate source and permissions before loading it.

## Project layout

```text
src/langchain_skill_runtime/
├── adapters/              # Four Tool adapters and ToolFactory
│   └── mcp/               # MCP protocols, provider, URL policies, and adapter
├── client/                # Client requests, results, and pending call management
├── executors/             # Function/script protocols and in-memory implementations
├── models/                # Skill, Tool, Context, and Bundle models
├── parsing/               # SKILL.md loading and parsing
├── prompting/             # System prompt compilation
├── repositories/          # Data source protocols
└── runtime/               # SkillRuntime orchestration
examples/                  # Minimal public usage examples
tests/                     # Unit and integration tests, not public API documentation
doc/                       # Chinese design and usage documents
```

## Local development

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

Run the public examples:

```bash
uv run python examples/calculator/main.py
uv run python examples/client_tool/main.py
```

The MCP example uses a placeholder domain and environment variable. Replace them
with an authorized real MCP Server and its credentials before running it.

## License

Licensed under the [Apache License 2.0](LICENSE).

