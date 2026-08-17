# MCP 会话生命周期实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让同一宿主作用域内的 MCP 工具共享同一 Session，并在作用域结束时可靠释放，同时保持无状态调用兼容。

**Architecture:** `LangChainMcpToolProvider` 实现异步上下文管理器，用 `AsyncExitStack` 按 `server_name` 惰性创建、缓存并关闭 MCP Session。作用域内通过 `client.session()` 和 `load_mcp_tools(session)` 生成绑定同一 Session 的工具；作用域外仍调用 `client.get_tools()`。

**Tech Stack:** Python 3.11+、LangChain Core、langchain-mcp-adapters、pytest、pytest-asyncio、Ruff、Mypy、uv。

## Global Constraints

- 只在 `master` 分支开发，不创建或操作其他分支。
- 生产代码不出现 Playwright、`browser_*`、Chrome 或 CDP 条件分支。
- 不修改 `McpToolProvider`、`McpToolCollectionProvider`、`McpToolAdapter` 和 `SkillRuntime.compile_file()` 的现有公开签名。
- 不使用 Provider 异步作用域时，保留当前临时 Session 行为。
- 凭据解析、`server_ref` 和 MCP URL 安全策略不降级。
- 新版本为 `0.3.4`。
- 文档使用中文文件名并只放在 `doc/`。

---

### Task 1: 用 TDD 实现通用 MCP Session 作用域

**Files:**

- Create: `tests/unit/test_mcp_provider_lifecycle.py`
- Modify: `src/langchain_skill_runtime/adapters/mcp/provider.py`

**Interfaces:**

- Consumes: `LangChainMcpToolProvider.get_tools(server_name, server_config, context)`。
- Produces: `LangChainMcpToolProvider.__aenter__()` 和 `__aexit__()`。
- Produces: `session_tool_loader(session: Any) -> Awaitable[list[BaseTool]]` 可选构造参数。
- Produces: 作用域内按 `server_name` 缓存的 Session 工具集。

- [x] **Step 1: 写入状态 Session 测试替身和核心失败用例**

```python
import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.tools import BaseTool, tool

from langchain_skill_runtime.adapters.mcp import LangChainMcpToolProvider
from langchain_skill_runtime.models.context import CompileContext

STDIO_CONFIG = {"transport": "stdio", "command": "stateful-server"}


@tool
async def mcp_echo(text: str) -> str:
    """Echo one value."""
    return text


class StatefulSession:
    def __init__(self, server_name: str) -> None:
        self.server_name = server_name
        self.values: dict[str, str] = {}


class StatefulClient:
    def __init__(self) -> None:
        self.opened = 0
        self.closed = 0

    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        raise AssertionError("managed scope must use session()")

    @asynccontextmanager
    async def session(self, server_name: str) -> AsyncIterator[StatefulSession]:
        self.opened += 1
        await asyncio.sleep(0)
        try:
            yield StatefulSession(server_name)
        finally:
            self.closed += 1


async def load_stateful_tools(session: StatefulSession) -> list[BaseTool]:
    await asyncio.sleep(0)

    @tool("state_write")
    async def state_write(value: str) -> str:
        """Write one value into the current MCP session."""
        session.values["value"] = value
        return "stored"

    @tool("state_read")
    async def state_read() -> str:
        """Read one value from the current MCP session."""
        return session.values.get("value", "missing")

    return [state_write, state_read]


@pytest.mark.asyncio
async def test_managed_tools_share_one_session() -> None:
    client = StatefulClient()
    provider = LangChainMcpToolProvider(
        client_factory=lambda _connections: client,
        session_tool_loader=load_stateful_tools,
    )
    async with provider:
        tools = await provider.get_tools("stateful", STDIO_CONFIG, CompileContext())
        by_name = {item.name: item for item in tools}
        assert await by_name["state_write"].ainvoke({"value": "kept"}) == "stored"
        assert await by_name["state_read"].ainvoke({}) == "kept"
        again = await provider.get_tools("stateful", STDIO_CONFIG, CompileContext())
        assert again == tools
    assert client.opened == 1
    assert client.closed == 1
```

- [x] **Step 2: 在实现前写入全部生命周期边界用例**

```python
@pytest.mark.asyncio
async def test_managed_scope_closes_session_after_agent_error() -> None:
    client = StatefulClient()
    provider = LangChainMcpToolProvider(
        client_factory=lambda _connections: client,
        session_tool_loader=load_stateful_tools,
    )
    with pytest.raises(RuntimeError, match="agent failed"):
        async with provider:
            await provider.get_tools("stateful", STDIO_CONFIG, CompileContext())
            raise RuntimeError("agent failed")
    assert client.closed == 1


@pytest.mark.asyncio
async def test_managed_scope_isolates_server_names() -> None:
    clients: list[StatefulClient] = []

    def factory(_connections: dict[str, Mapping[str, Any]]) -> StatefulClient:
        client = StatefulClient()
        clients.append(client)
        return client

    provider = LangChainMcpToolProvider(
        client_factory=factory,
        session_tool_loader=load_stateful_tools,
    )
    async with provider:
        first = await provider.get_tools("first", STDIO_CONFIG, CompileContext())
        second = await provider.get_tools("second", STDIO_CONFIG, CompileContext())
        await {item.name: item for item in first}["state_write"].ainvoke(
            {"value": "first-only"}
        )
        assert (
            await {item.name: item for item in second}["state_read"].ainvoke({})
            == "missing"
        )
    assert len(clients) == 2
    assert all(client.closed == 1 for client in clients)


@pytest.mark.asyncio
async def test_managed_scope_concurrent_discovery_opens_one_session() -> None:
    client = StatefulClient()
    provider = LangChainMcpToolProvider(
        client_factory=lambda _connections: client,
        session_tool_loader=load_stateful_tools,
    )
    async with provider:
        first, second = await asyncio.gather(
            provider.get_tools("stateful", STDIO_CONFIG, CompileContext()),
            provider.get_tools("stateful", STDIO_CONFIG, CompileContext()),
        )
    assert first == second
    assert client.opened == 1
    assert client.closed == 1


@pytest.mark.asyncio
async def test_managed_scope_rejects_reentry_and_can_be_reused() -> None:
    provider = LangChainMcpToolProvider()
    async with provider:
        with pytest.raises(RuntimeError, match="已在运行作用域"):
            async with provider:
                pass
    async with provider:
        pass


class StatelessClient:
    def __init__(self) -> None:
        self.calls = 0

    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        self.calls += 1
        return [mcp_echo]


@pytest.mark.asyncio
async def test_unmanaged_provider_keeps_temporary_loading() -> None:
    client = StatelessClient()
    provider = LangChainMcpToolProvider(client_factory=lambda _connections: client)
    tools = await provider.get_tools("echo", STDIO_CONFIG, CompileContext())
    assert tools == [mcp_echo]
    assert client.calls == 1
```

- [x] **Step 3: 运行测试并确认 RED**

```bash
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache \
  uv run pytest tests/unit/test_mcp_provider_lifecycle.py -q
```

Expected: 5 个 lifecycle 用例 FAIL，原因是构造器尚无 `session_tool_loader` 且 Provider 尚未实现异步上下文管理协议；无状态用例 PASS。

- [x] **Step 4: 实现最小通用作用域**

`provider.py` 增加：

```python
import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from types import TracebackType

McpSessionToolLoader = Callable[[Any], Awaitable[list[BaseTool]]]
```

构造器增加 `session_tool_loader` 关键字参数和状态：

```python
self._session_tool_loader = (
    session_tool_loader or self._default_session_tool_loader
)
self._session_stack: AsyncExitStack | None = None
self._session_tools: dict[str, list[BaseTool]] = {}
self._session_lock: asyncio.Lock | None = None
```

异步作用域：

```python
async def __aenter__(self) -> "LangChainMcpToolProvider":
    if self._session_stack is not None:
        raise RuntimeError("MCP Provider 已在运行作用域中")
    stack = AsyncExitStack()
    await stack.__aenter__()
    self._session_stack = stack
    self._session_tools = {}
    self._session_lock = asyncio.Lock()
    return self

async def __aexit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: TracebackType | None,
) -> bool:
    stack = self._session_stack
    if stack is None:
        return False
    try:
        return await stack.__aexit__(exc_type, exc, traceback)
    finally:
        self._session_stack = None
        self._session_tools = {}
        self._session_lock = None
```

`get_tools()` 在现有配置、Secret 和 URL 策略处理后分流：

```python
if self._session_stack is not None:
    return await self._get_session_tools(server_name, prepared)
client = self._client_factory({server_name: prepared})
return await client.get_tools(server_name=server_name)
```

作用域发现和默认 Loader：

```python
async def _get_session_tools(
    self,
    server_name: str,
    prepared: Mapping[str, Any],
) -> list[BaseTool]:
    cached = self._session_tools.get(server_name)
    if cached is not None:
        return cached
    lock = self._session_lock
    stack = self._session_stack
    if lock is None or stack is None:
        raise RuntimeError("MCP Provider 作用域已结束")
    async with lock:
        cached = self._session_tools.get(server_name)
        if cached is not None:
            return cached
        client = self._client_factory({server_name: prepared})
        session_factory = getattr(client, "session", None)
        if session_factory is None:
            raise RuntimeError("MCP Client 不支持显式 Session")
        candidate = AsyncExitStack()
        await candidate.__aenter__()
        try:
            session = await candidate.enter_async_context(session_factory(server_name))
            tools = await self._session_tool_loader(session)
        except BaseException:
            await candidate.aclose()
            raise
        stack.push_async_callback(candidate.aclose)
        self._session_tools[server_name] = tools
        return tools

@staticmethod
async def _default_session_tool_loader(session: Any) -> list[BaseTool]:
    try:
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError:
        raise ToolUnavailableError(
            "使用 MCP Tool 需要安装 langchain-skill-runtime[mcp]"
        ) from None
    return cast(list[BaseTool], await load_mcp_tools(session))
```

- [x] **Step 5: 运行 GREEN、静态检查并提交**

```bash
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache \
  uv run pytest tests/unit/test_mcp_provider_lifecycle.py -q
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache \
  uv run ruff check src/langchain_skill_runtime/adapters/mcp/provider.py \
  tests/unit/test_mcp_provider_lifecycle.py
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache \
  uv run ruff format --check src/langchain_skill_runtime/adapters/mcp/provider.py \
  tests/unit/test_mcp_provider_lifecycle.py
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache \
  uv run mypy src/langchain_skill_runtime/adapters/mcp/provider.py \
  tests/unit/test_mcp_provider_lifecycle.py
git add src/langchain_skill_runtime/adapters/mcp/provider.py \
  tests/unit/test_mcp_provider_lifecycle.py
git commit -m "fix: 保持MCP工具调用会话"
```

Expected: `6 passed`，静态检查通过。

---

### Task 2: 更新 0.3.4 版本、文档和 wheel

**Files:**

- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `uv.lock`

**Interfaces:**

- Consumes: Task 1 的 `async with LangChainMcpToolProvider(...) as provider`。
- Produces: `langchain-skill-runtime==0.3.4` wheel 和中英文使用说明。

- [x] **Step 1: 升级版本并更新中英文 MCP 示例**

`pyproject.toml` 和 README 版本号从 `0.3.3` 改为 `0.3.4`。MCP 示例改为：

```python
async with LangChainMcpToolProvider(
    url_policy=AllowHostsMcpUrlPolicy({"mcp.example.com"})
) as provider:
    runtime = SkillRuntime(
        tool_factory=ToolFactory([McpToolAdapter(provider)])
    )
    bundle = await runtime.compile_file("skills/navigation/SKILL.md")
    agent = create_agent(
        model=model,
        tools=list(bundle.tools),
        system_prompt=bundle.system_prompt,
    )
    result = await agent.ainvoke(request)
```

文档明确：无状态单次工具可继续使用原方式；多步有状态 MCP
必须在 Provider 异步作用域内完成 Agent 运行。

- [x] **Step 2: 更新锁文件并执行库级验证**

```bash
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache uv lock
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache uv run mypy src tests
```

- [x] **Step 3: 构建 wheel 并提交**

```bash
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache uv build --wheel
test -f dist/langchain_skill_runtime-0.3.4-py3-none-any.whl
git add pyproject.toml README.md README_EN.md uv.lock
git commit -m "release: 准备0.3.4会话生命周期修复"
```

---

### Task 3: Demo 在同一 Provider 作用域内运行 Agent

**Files:**

- Modify: `/Users/zqs/PycharmProjects/AI/langchain-skill-runtime-openai-demo/pyproject.toml`
- Modify: `/Users/zqs/PycharmProjects/AI/langchain-skill-runtime-openai-demo/uv.lock`
- Modify: `/Users/zqs/PycharmProjects/AI/langchain-skill-runtime-openai-demo/src/skill_runtime_openai_demo/playwright.py`
- Create: `/Users/zqs/PycharmProjects/AI/langchain-skill-runtime-openai-demo/vendor/langchain_skill_runtime-0.3.4-py3-none-any.whl`

**Interfaces:**

- Consumes: Task 2 的 `0.3.4` wheel 和 Provider 异步作用域。
- Produces: 编译 Skill 和 Agent 多步工具调用共享同一 MCP Session 的 Demo。

- [x] **Step 1: 复现修复前的真实 RED**

```bash
cd /Users/zqs/PycharmProjects/AI/langchain-skill-runtime-openai-demo
.venv/bin/pytest tests/test_mcp.py::test_init_agent -q -s
```

Expected: 模型可调用 `browser_navigate`，但后续工具返回 `about:blank`，最终触发 `GraphRecursionError`。

- [x] **Step 2: 更新 wheel 依赖**

`pyproject.toml` 改为 `langchain-skill-runtime[agent,mcp]==0.3.4`，本地源改为：

```toml
langchain-skill-runtime = { path = "vendor/langchain_skill_runtime-0.3.4-py3-none-any.whl" }
```

将 Task 2 wheel 写入 Demo `vendor/`，再执行：

```bash
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-demo-uv-cache uv lock
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-demo-uv-cache uv sync
```

- [x] **Step 3: 让 Provider 作用域覆盖编译和 Agent 调用**

```python
async def compile_playwright_skill(
    provider: LangChainMcpToolProvider,
) -> SkillBundle:
    runtime = SkillRuntime(tool_factory=ToolFactory([McpToolAdapter(provider)]))
    return await runtime.compile_file(PLAYWRIGHT_SKILL_PATH)
```

`run_playwright_agent()` 使用：

```python
provider = LangChainMcpToolProvider(
    server_config_provider=PlaywrightServerConfigProvider(Path(temp_dir))
)
async with provider:
    bundle = await compile_playwright_skill(provider)
    agent = create_agent(
        model=model,
        tools=list(bundle.tools),
        system_prompt=bundle.system_prompt,
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 12},
    )
```

`recursion_limit=12` 只为正常多步操作提供步数，不用来规避 Session 丢失。

- [x] **Step 4: 运行 Demo 回归和静态检查**

```bash
.venv/bin/pytest tests/test_playwright_skill.py -q
.venv/bin/ruff check src/skill_runtime_openai_demo/playwright.py
.venv/bin/ruff format --check src/skill_runtime_openai_demo/playwright.py
.venv/bin/mypy src/skill_runtime_openai_demo/playwright.py
```

- [x] **Step 5: 只提交 Demo 本任务文件**

```bash
git commit --only -m "fix: 保持MCP Agent运行会话" -- \
  pyproject.toml uv.lock \
  vendor/langchain_skill_runtime-0.3.4-py3-none-any.whl \
  src/skill_runtime_openai_demo/playwright.py
```

---

### Task 4: 真实多步验收与最终回归

**Files:**

- Verify: `/Users/zqs/PycharmProjects/AI/langchain-skill-runtime-openai-demo/skills/playwright-browser/SKILL.md`
- Modify: `doc/MCP会话生命周期实施计划.md`

**Interfaces:**

- Consumes: Task 3 安装的 `langchain-skill-runtime==0.3.4`。
- Produces: 至少两次连续原生工具调用保持同一 Server 状态的真实证据。

- [x] **Step 1: 运行两步真实对话**

```bash
cd /Users/zqs/PycharmProjects/AI/langchain-skill-runtime-openai-demo
.venv/bin/python -m skill_runtime_openai_demo.playwright \
  "打开 https://example.com，先读取页面，再点击 More information 链接，告诉我最终页面标题"
```

Expected: 发现真实原生工具；至少调用导航和后续观察或点击；第二步不返回 `about:blank`；Agent 不触发 `GraphRecursionError`。

- [x] **Step 2: 运行百度业务对话**

```bash
.venv/bin/python -m skill_runtime_openai_demo.playwright \
  "打开百度搜索住小叮"
```

百度验证码是目标站结果；验收重点是连续工具调用不丢失 Session。

- [x] **Step 3: 运行最终验证**

```bash
cd /Users/zqs/PycharmProjects/AI/langchain-skill-runtime
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/langchain-skill-runtime-uv-cache uv run mypy src tests
rg -n "playwright|browser_|Chrome|CDP" src tests
git diff --check

cd /Users/zqs/PycharmProjects/AI/langchain-skill-runtime-openai-demo
AMAP_MAPS_API_KEY= .venv/bin/pytest -q --ignore=tests/test_mcp.py
git diff --check
```

Expected: Library `src/` 无业务特判匹配，库和 Demo 目标回归通过。

- [x] **Step 4: 记录执行结果**

在本文档末尾记录实际测试数量、wheel 文件名、Demo 原生工具数量、连续工具调用名称和最终页面结果。

## 实际执行结果

- 修复前 RED：5 个生命周期用例失败、1 个无状态兼容用例通过；失败原因是 Provider 尚未支持会话工具加载器和异步上下文协议。
- 库最终验证：`7 passed`；Ruff 检查通过；49 个文件格式检查通过；Mypy 检查 42 个源文件通过。
- 构建产物：`dist/langchain_skill_runtime-0.3.4-py3-none-any.whl`。
- Demo 依赖：虚拟环境实际安装 `langchain-skill-runtime==0.3.4`。
- Demo 回归：真实网络下 `12 passed`，包括现有高德 MCP 场景；目标 Playwright 文件 Ruff、格式和 Mypy 检查通过。
- 多步真实验收：发现 24 个 Playwright MCP 原生工具，连续调用 `browser_navigate`、`browser_snapshot`、`browser_click`，最终页面标题为 `Example Domains`；后续调用未回到 `about:blank`。
- 百度默认示例：调用 `browser_navigate` 打开百度“住小叮”公开查询地址，页面标题为 `住小叮_百度搜索`，搜索结果页正常加载。
- 已知模型波动：让当前模型从百度首页自行填写和点击时，曾生成不符合这版 MCP Schema 的 `ref` 参数；因此无参数稳定示例使用等价的公开查询地址。这不影响 Provider 会话复用结论。
