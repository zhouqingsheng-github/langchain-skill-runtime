import asyncio
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_skill_runtime.adapters.client_javascript import (
    ClientJavascriptAdapter,
)
from langchain_skill_runtime.adapters.factory import ToolFactory
from langchain_skill_runtime.adapters.function import PythonFunctionAdapter
from langchain_skill_runtime.adapters.mcp import (
    LangChainMcpToolProvider,
    McpToolAdapter,
)
from langchain_skill_runtime.adapters.server_script import ServerScriptAdapter
from langchain_skill_runtime.client.models import (
    ClientToolRequest,
    ClientToolResult,
    ClientToolStatus,
)
from langchain_skill_runtime.client.transport import PendingClientToolTransport
from langchain_skill_runtime.executors.function_registry import (
    InMemoryFunctionRegistry,
)
from langchain_skill_runtime.models.context import ClientCapability, CompileContext
from langchain_skill_runtime.models.skill import SkillDefinition
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType
from langchain_skill_runtime.runtime.skill_runtime import SkillRuntime

SKILL_PATH = Path("tests/fixtures/skills/heterogeneous-tools/SKILL.md")
MCP_SERVER_PATH = Path("tests/fixtures/mcp/echo_server.py").resolve()


class MemorySkillRepository:
    async def get_skill(
        self,
        skill_id: str,
        context: CompileContext,
    ) -> SkillDefinition | None:
        del context
        if skill_id != "heterogeneous-skill":
            return None
        return SkillDefinition(
            id=skill_id,
            name="heterogeneous-tool-test",
            description="验证一个技能同时使用 Function、脚本、客户端和 MCP 工具",
            content=SKILL_PATH.read_text(encoding="utf-8"),
            version="1.0.0",
        )


class MemoryToolRepository:
    def __init__(self, tools: Sequence[ResolvedToolDefinition]) -> None:
        self._tools = tools

    async def list_tools(
        self,
        skill_id: str,
        context: CompileContext,
    ) -> Sequence[ResolvedToolDefinition]:
        del skill_id, context
        return self._tools


class TestScriptExecutor:
    async def execute(
        self,
        artifact_id: str,
        arguments: Mapping[str, Any],
        context: CompileContext,
        timeout_seconds: float,
    ) -> Any:
        assert artifact_id == "report.test.v1"
        assert context.tenant_id == "tenant-1"
        assert timeout_seconds == 15.0
        return {"status": "generated", "title": arguments["title"]}


class ToolBindingFakeModel(GenericFakeChatModel):
    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ToolBindingFakeModel":
        del tools, tool_choice, kwargs
        return self


def resolved_tools() -> list[ResolvedToolDefinition]:
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
            execution_config={"registry_key": "math.add"},
            version="1.0.0",
            sort_order=1,
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
            sort_order=2,
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
            sort_order=3,
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
                "server": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(MCP_SERVER_PATH)],
                },
            },
            version="1.0.0",
            sort_order=4,
        ),
    ]


@pytest.mark.asyncio
async def test_real_skill_compiles_and_executes_all_four_tool_types() -> None:
    registry = InMemoryFunctionRegistry()

    async def add_numbers(a: int, b: int) -> int:
        return a + b

    registry.register("math.add", add_numbers)
    client_requests: asyncio.Queue[ClientToolRequest] = asyncio.Queue()

    async def sender(call: ClientToolRequest) -> None:
        await client_requests.put(call)

    transport = PendingClientToolTransport(sender)
    runtime = SkillRuntime(
        skill_repository=MemorySkillRepository(),
        tool_repository=MemoryToolRepository(resolved_tools()),
        tool_factory=ToolFactory(
            [
                PythonFunctionAdapter(registry),
                ServerScriptAdapter(TestScriptExecutor()),
                ClientJavascriptAdapter(transport),
                McpToolAdapter(LangChainMcpToolProvider()),
            ]
        ),
    )
    context = CompileContext(
        tenant_id="tenant-1",
        user_id="user-1",
        session_id="session-1",
        client_capabilities=(
            ClientCapability(
                tool_id="client.export.file",
                version="1.0.0",
            ),
        ),
    )

    bundle = await runtime.compile("heterogeneous-skill", context)

    assert [tool.name for tool in bundle.tools] == [
        "add_numbers",
        "generate_report",
        "export_client_file",
        "mcp_echo",
    ]
    assert "# 异构工具测试技能" in bundle.system_prompt
    assert bundle.diagnostics == ()
    assert await bundle.tools[0].ainvoke({"a": 2, "b": 5}) == 7
    assert await bundle.tools[1].ainvoke({"title": "日报"}) == {
        "status": "generated",
        "title": "日报",
    }

    client_invocation = asyncio.create_task(
        bundle.tools[2].ainvoke({"format": "xlsx"})
    )
    client_request = await client_requests.get()
    await transport.accept_result(
        ClientToolResult(
            call_id=client_request.call_id,
            session_id=client_request.session_id,
            tool_id=client_request.tool_id,
            tool_version=client_request.tool_version,
            status=ClientToolStatus.SUCCESS,
            output={"file_name": "测试文件.xlsx"},
        )
    )
    assert await client_invocation == {"file_name": "测试文件.xlsx"}

    mcp_result = await bundle.tools[3].ainvoke({"text": "hello"})
    assert mcp_result[0]["type"] == "text"
    assert mcp_result[0]["text"] == "mcp:hello"

    model = ToolBindingFakeModel(messages=iter([AIMessage(content="装配成功")]))
    agent = create_agent(
        model=model,
        tools=list(bundle.tools),
        system_prompt=bundle.system_prompt,
    )
    agent_result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "验证装配"}]}
    )
    assert agent_result["messages"][-1].content == "装配成功"
