# MCP 会话生命周期设计文档

## 一、背景

`LangChainMcpToolProvider` 当前通过
`MultiServerMCPClient.get_tools()` 发现 MCP 工具。该方式会在每次工具调用时
创建新的 MCP Session。

对于高德路线查询等无状态工具，每次调用可以独立完成，因此问题不明显。
对于浏览器自动化等有状态 MCP Server，后续点击、输入和快照必须依赖前一步保留的
Server 状态。当工具调用切换到新 Session 后，状态会丢失。

真实现象是：第一次导航成功，第二次快照返回 `about:blank`，Agent
不断重试直到触发 `GraphRecursionError`。

## 二、目标

- 在宿主明确划定的一次 MCP 运行作用域内，复用同一个 MCP Session。
- 同一 `server_name` 发现的所有工具共享同一个 Server 状态。
- 作用域正常结束或异常结束时，可靠关闭 Session 和子进程。
- 设计适用于所有 MCP Server，不包含 Playwright、高德或其他业务特判。
- 保留当前不使用显式作用域的无状态调用方式，避免破坏现有用户。

## 三、非目标

- 不在 Runtime 中管理 Chrome、CDP、用户 Profile 或网站登录态。
- 不改变 MCP 协议、Server 工具 Schema 或 Agent 的工具选择策略。
- 不通过增大 Agent `recursion_limit` 掩盖状态丢失。
- 不在 SKILL.md 中增加任何会话类型或特定 MCP Server 标记。

## 四、方案对比

### 方案 A：Provider 显式异步作用域（采用）

`LangChainMcpToolProvider` 实现异步上下文管理协议。宿主在 `async with`
内编译 Skill 并运行 Agent。Provider 按 `server_name` 惰性创建并复用 Session，
退出作用域时由 `AsyncExitStack` 统一关闭。

优点：生命周期边界明确、向后兼容、不侵入 `SkillBundle`、可对任意 MCP Server 生效。

### 方案 B：SkillBundle 持有会话（不采用）

在 `SkillBundle` 中保存资源释放函数，并要求调用者手动执行 `aclose()`。
这会将运行资源引入原本只表示编译结果的模型，且调用者容易忘记关闭。

### 方案 C：外部持久化特定 Server 状态（不采用）

例如让宿主预先启动 Chrome 并通过 CDP 复用。该方案只能规避特定 Server
的会话问题，无法修复 Runtime 的通用 MCP 生命周期。

## 五、公开 API

使用方式为：

```python
async with LangChainMcpToolProvider(
    server_config_provider=server_config_provider,
) as provider:
    runtime = SkillRuntime(
        tool_factory=ToolFactory([McpToolAdapter(provider)]),
    )
    bundle = await runtime.compile_file(skill_path)
    agent = create_agent(
        model=model,
        tools=list(bundle.tools),
        system_prompt=bundle.system_prompt,
    )
    result = await agent.ainvoke(request)
```

语义如下：

- `__aenter__()` 开启一个 Provider 作用域，但不立即连接 Server。
- 首次发现某个 `server_name` 的工具时，惰性创建 MCP Client 和 Session。
- 同一作用域内再次请求同一 `server_name` 时，返回该 Session 上已加载的工具。
- 不同 `server_name` 建立相互隔离的 Session。
- `__aexit__()` 按逆序关闭所有 Session，然后清理内部缓存。
- 不使用 `async with` 时，`get_tool()` 和 `get_tools()` 继续采用当前的临时
  Session 行为。

Provider 实例不支持同时重复进入多个作用域。重复进入时返回可诊断的
Runtime 异常，防止 Session 归属不清。

## 六、内部组件

### 6.1 会话客户端协议

在现有 `_McpClient.get_tools()` 之外，引入可选的显式 Session 能力：

- `session(server_name)` 返回异步上下文管理器。
- Session 中通过 `load_mcp_tools(session)` 加载的 LangChain Tools 固定绑定到该 Session。
- 只有在 Provider 作用域内才要求 Client 支持显式 Session。

### 6.2 会话缓存

Provider 按 `server_name` 保存已加载的工具集合。一个作用域内，
`server_name` 是 MCP Server 的逻辑唯一标识；宿主不应用同一名称表示两个不同配置。

首次加载失败时不写入缓存，允许调用者处理异常后重试。

### 6.3 并发保护

使用 Provider 级异步锁保护 Session 的首次创建。同一 `server_name` 的并发发现
只创建一个 Session；已建立 Session 的工具正常并发调用由 MCP Client 和 Server
各自的能力边界决定。

## 七、错误和清理

- Server 配置、Secret 解析和 URL 安全策略保持现有逻辑。
- Session 建立或工具发现失败时，对外继续转换为受控的
  `ToolUnavailableError("MCP 工具发现失败")`。
- 作用域中 Agent 或工具调用抛异常时，`AsyncExitStack` 仍必须关闭所有 Session。
- 关闭后清除工具缓存、Client 引用和异步锁，Provider 可用于后续新的作用域。
- 不向诊断信息写入 URL 凭据、Header、Secret 或 Server 原始异常文本。

## 八、兼容性

- `McpToolProvider` 和 `McpToolCollectionProvider` 的公开协议不变。
- `McpToolAdapter` 和 `SkillRuntime.compile_file()` 的签名不变。
- 现有 `client_factory` 位置参数和无状态伪客户端用例继续可用。
- 需要会话保持的宿主通过 `async with provider` 显式选择新行为。
- 该修复作为向后兼容的补丁版本，版本号从 `0.3.3` 升级为 `0.3.4`。

## 九、测试设计

在 `master` 分支恢复必要的最小测试目录，不恢复已删除的历史文档和无关用例。

单元测试覆盖：

1. 不使用 Provider 作用域时，仍调用客户端 `get_tools()`。
2. 作用域内两次请求同一 `server_name`，只进入一次 Session 并返回同一组工具。
3. 作用域内调用两个工具，第二个工具能读取第一个工具写入的 Session 状态。
4. 不同 `server_name` 各自创建和关闭 Session。
5. 作用域正常退出和异常退出都会关闭 Session。
6. 重复进入同一 Provider 时给出受控异常。
7. Secret 解析、`server_ref` 和 URL 安全策略原有行为不回归。

端到端验收覆盖：

1. 构建 `0.3.4` wheel。
2. Demo 安装新 wheel 后，使用通用 Playwright SKILL 执行至少两步操作。
3. 确认导航后的快照、输入或点击仍处于同一页面，不再返回 `about:blank`。
4. 确认 Agent 能正常结束，不再触发 `GraphRecursionError`。

## 十、验收标准

- 生产代码不出现 `playwright`、`browser_*`、Chrome 或 CDP 条件分支。
- Session 作用域由宿主显式创建，并且无论成功或失败都可确定释放。
- 同一 Server 的多步工具调用共享 Session 状态。
- 现有无状态 MCP 工具调用和安全策略保持兼容。
- 单元测试、Ruff、格式检查和 Mypy 全部通过。
- 真实可见浏览器完成多步操作，作为状态 MCP 的端到端样例，但不作为
  Runtime 内部实现依赖。
