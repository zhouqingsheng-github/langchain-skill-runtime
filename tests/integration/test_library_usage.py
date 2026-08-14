"""完整使用 Demo：展示这个库支持的三种 Skill 输入方式。"""

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool

from langchain_skill_runtime import (
    ClientCapability,
    CompileContext,
    ResolvedToolDefinition,
    SkillDefinition,
    SkillRuntime,
    ToolType,
)
from langchain_skill_runtime.adapters import (
    ClientJavascriptAdapter,
    McpToolAdapter,
    PythonFunctionAdapter,
    ServerScriptAdapter,
    ToolFactory,
)
from langchain_skill_runtime.client import (
    ClientToolRequest,
    ClientToolResult,
    ClientToolStatus,
    PendingClientToolTransport,
)
from langchain_skill_runtime.executors import InMemoryFunctionRegistry
from langchain_skill_runtime.repositories import SkillRepository

TESTS_DIR = Path(__file__).resolve().parents[1]
SKILL_PATH = TESTS_DIR / "fixtures/skills/heterogeneous-tools/SKILL.md"


class MemorySkillRepository(SkillRepository):
    """模拟业务系统从数据库读取 Skill 对象。"""

    def __init__(self, skill: SkillDefinition) -> None:
        self._skill = skill

    async def get_skill(
        self,
        skill_id: str,
        context: CompileContext,
    ) -> SkillDefinition | None:
        del context
        return self._skill if skill_id == self._skill.id else None


class MemoryToolRepository:
    """模拟业务系统从数据库读取已授权的 Tool 对象。"""

    def __init__(self, tools: Sequence[ResolvedToolDefinition]) -> None:
        self._tools = tools

    async def list_tools(
        self,
        skill_id: str,
        context: CompileContext,
    ) -> Sequence[ResolvedToolDefinition]:
        del skill_id, context
        return self._tools


class DemoScriptExecutor:
    """示例 Script Executor；生产环境可替换为沙箱或任务服务。"""

    async def execute(
        self,
        artifact_id: str,
        arguments: Mapping[str, Any],
        context: CompileContext,
        timeout_seconds: float,
    ) -> Any:
        del artifact_id, context, timeout_seconds
        return {"status": "generated", "title": arguments["title"]}


@tool
async def mcp_echo(text: str) -> str:
    """测试用 MCP Tool，由 DemoMcpProvider 模拟发现。"""

    return f"mcp:{text}"


class DemoMcpProvider:
    """避免 Demo 依赖外部 MCP 服务，仅替代工具发现边界。"""

    async def get_tool(
        self,
        server_name: str,
        tool_name: str,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> BaseTool | None:
        del server_name, server_config, context
        return mcp_echo if tool_name == "mcp_echo" else None


class ToolBindingFakeModel(GenericFakeChatModel):
    """离线测试模型，允许 create_agent 绑定本库编译出的 Tools。"""

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ToolBindingFakeModel":
        del tools, tool_choice, kwargs
        return self


def database_skill_object() -> SkillDefinition:
    """模拟数据库记录映射成的 SkillDefinition。"""

    return SkillDefinition(
        id="heterogeneous-skill",
        name="heterogeneous-tool-test",
        description="验证一个技能同时使用 Function、脚本、客户端和 MCP 工具",
        content=SKILL_PATH.read_text(encoding="utf-8"),
        version="1.0.0",
    )


def database_tool_objects() -> list[ResolvedToolDefinition]:
    """模拟数据库 Tool 表和关系表映射后的对象列表。"""

    return [
        ResolvedToolDefinition(
            id="function-add",
            name="add_numbers",
            description="计算两个数字之和",
            tool_type=ToolType.PYTHON_FUNCTION,
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            output_schema={"type": "integer"},
            execution_config={"registry_key": "math.add"},
            version="1.0.0",
            sort_order=0,
        ),
        ResolvedToolDefinition(
            id="script-report",
            name="generate_report",
            description="生成测试报告",
            tool_type=ToolType.SERVER_SCRIPT,
            input_schema={
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
                "additionalProperties": False,
            },
            execution_config={"artifact_id": "report.test.v1"},
            timeout_seconds=15,
            version="1.0.0",
            sort_order=1,
        ),
        ResolvedToolDefinition(
            id="client-export",
            name="export_client_file",
            description="在客户端导出测试文件",
            tool_type=ToolType.CLIENT_JAVASCRIPT,
            input_schema={
                "type": "object",
                "properties": {"format": {"type": "string", "enum": ["xlsx"]}},
                "required": ["format"],
                "additionalProperties": False,
            },
            execution_config={"tool_key": "client.export.file"},
            timeout_seconds=10,
            version="1.0.0",
            sort_order=2,
        ),
        ResolvedToolDefinition(
            id="mcp-echo",
            name="mcp_echo",
            description="通过 MCP 回显文本",
            tool_type=ToolType.MCP,
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            execution_config={
                "server_name": "echo",
                "tool_name": "mcp_echo",
                "server": {"transport": "stdio", "command": "python", "args": []},
            },
            version="1.0.0",
            sort_order=3,
        ),
    ]


@pytest.mark.asyncio
async def test_complete_library_usage() -> None:
    # 第一步：注册受控 Python Function；SKILL.md 只保存 registry_key，不保存源码。
    function_registry = InMemoryFunctionRegistry()

    # 这是宿主系统真正提供的业务函数。
    async def add_numbers(a: int, b: int) -> int:
        return a + b

    # registry_key=math.add 会在编译 Function Tool 时查找到这个函数。
    function_registry.register("math.add", add_numbers)

    # 第二步：创建客户端 Transport；sender 在生产环境通常负责发送 WebSocket 消息。
    client_requests: asyncio.Queue[ClientToolRequest] = asyncio.Queue()

    async def sender(request: ClientToolRequest) -> None:
        await client_requests.put(request)

    client_transport = PendingClientToolTransport(sender)

    # 第三步：把四种 Adapter 注册到同一个 ToolFactory。
    tool_factory = ToolFactory(
        [
            PythonFunctionAdapter(function_registry),
            ServerScriptAdapter(DemoScriptExecutor()),
            ClientJavascriptAdapter(client_transport),
            McpToolAdapter(DemoMcpProvider()),
        ]
    )

    # 第四步：准备执行上下文；只有客户端 Tool 依赖 session 和 capability。
    context = CompileContext(
        session_id="session-1",
        client_capabilities=(
            ClientCapability(tool_id="client.export.file", version="1.0.0"),
        ),
    )

    # 用法一：直接读取市场或本地提供的 SKILL.md，不需要 Repository 或数据库。
    file_runtime = SkillRuntime(tool_factory=tool_factory)
    file_bundle = await file_runtime.compile_file(SKILL_PATH, context)

    # 用法二：把数据库查询结果映射成对象后直接编译，核心库不连接数据库。
    skill_object = database_skill_object()
    tool_objects = database_tool_objects()

    object_runtime = SkillRuntime(tool_factory=tool_factory)
    object_bundle = await object_runtime.compile_objects(
        skill=skill_object,
        tools=tool_objects,
        context=context,
    )

    # 用法三：保留旧 Repository 模式，现有项目可以继续按 Skill ID 编译。
    repository_runtime = SkillRuntime(
        skill_repository=MemorySkillRepository(skill_object),
        tool_repository=MemoryToolRepository(tool_objects),
        tool_factory=tool_factory,
    )

    repository_bundle = await repository_runtime.compile(
        "heterogeneous-skill",
        context,
    )
    # 三种来源最终进入同一编译核心，因此得到相同的 Tool 列表和系统提示词。
    expected_names = [
        "add_numbers",
        "generate_report",
        "export_client_file",
        "mcp_echo",
    ]
    assert [tool.name for tool in file_bundle.tools] == expected_names
    assert [tool.name for tool in object_bundle.tools] == expected_names
    assert [tool.name for tool in repository_bundle.tools] == expected_names
    assert file_bundle.system_prompt == object_bundle.system_prompt
    assert object_bundle.system_prompt == repository_bundle.system_prompt
    assert file_bundle.diagnostics == ()
    assert object_bundle.diagnostics == ()
    assert repository_bundle.diagnostics == ()

    # 业务代码建议按名称取得 Tool，避免依赖 Tool 在列表中的位置。
    tools_by_name = {tool.name: tool for tool in file_bundle.tools}

    # 调用 Python Function Tool，验证 registry_key 已正确连接到宿主函数。
    assert await tools_by_name["add_numbers"].ainvoke({"a": 2, "b": 5}) == 7

    # 调用 Server Script Tool，验证请求已委派给注入的 Script Executor。
    assert await tools_by_name["generate_report"].ainvoke({"title": "日报"}) == {
        "status": "generated",
        "title": "日报",
    }

    # 客户端 Tool 会先挂起等待，不会在服务端直接执行 JavaScript。
    client_call = asyncio.create_task(
        tools_by_name["export_client_file"].ainvoke({"format": "xlsx"})
    )

    # 模拟 WebSocket 层收到服务端生成的、带唯一 call_id 的客户端请求。
    client_request = await client_requests.get()

    # 模拟客户端执行完成后，把相同 call_id 的结果交回 Transport。
    await client_transport.accept_result(
        ClientToolResult(
            call_id=client_request.call_id,
            session_id=client_request.session_id,
            tool_id=client_request.tool_id,
            tool_version=client_request.tool_version,
            status=ClientToolStatus.SUCCESS,
            output={"file_name": "测试文件.xlsx"},
        )
    )

    # 原先等待的 Tool 调用会拿到对应客户端结果。
    assert await client_call == {"file_name": "测试文件.xlsx"}

    # 调用 MCP Tool，验证 Adapter 已把发现的 MCP Tool 转成 LangChain Tool。
    assert await tools_by_name["mcp_echo"].ainvoke({"text": "hello"}) == "mcp:hello"

    # 对象模式和旧 Repository 模式也可以直接执行相同的 Function Tool。
    assert await object_bundle.tools[0].ainvoke({"a": 3, "b": 4}) == 7
    assert await repository_bundle.tools[0].ainvoke({"a": 4, "b": 5}) == 9

    # 最后把 SkillBundle 交给 LangChain create_agent；真实项目替换为自己的模型。
    fake_model = ToolBindingFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "add_numbers",
                            "args": {"a": 10, "b": 20},
                            "id": "demo-tool-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="计算完成"),
            ]
        )
    )
    agent = create_agent(
        model=fake_model,
        tools=list(file_bundle.tools),
        system_prompt=file_bundle.system_prompt,
    )

    # Agent 会执行模型选择的 Tool，并把结果送回模型生成最终回复。
    agent_result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "计算 10 + 20"}]}
    )
    assert any(
        message.type == "tool" and message.content == "30"
        for message in agent_result["messages"]
    )
    assert agent_result["messages"][-1].content == "计算完成"
