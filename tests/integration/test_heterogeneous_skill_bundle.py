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

TESTS_DIR = Path(__file__).resolve().parents[1]
SKILL_PATH = TESTS_DIR / "fixtures/skills/heterogeneous-tools/SKILL.md"
MCP_SERVER_PATH = TESTS_DIR / "fixtures/mcp/echo_server.py"


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
    # 创建 Python Function 注册表，供 Function Adapter 按 registry_key 查找函数。
    registry = InMemoryFunctionRegistry()

    # 定义测试用的异步加法函数，模拟服务端已经实现的业务 Function。
    async def add_numbers(a: int, b: int) -> int:
        # 返回两个输入参数的和，后面用具体结果验证工具是否真实执行。
        return a + b

    # 把函数注册为 math.add，使 SKILL Tool 配置可以通过该键找到它。
    registry.register("math.add", add_numbers)
    # 创建客户端请求队列，模拟 WebSocket 层收到服务端下发的 Tool 请求。
    client_requests: asyncio.Queue[ClientToolRequest] = asyncio.Queue()

    # 定义 Transport 的发送函数，收到请求后将其放入测试队列。
    async def sender(call: ClientToolRequest) -> None:
        # 保存带 call_id 的请求，稍后模拟客户端按该 call_id 返回结果。
        await client_requests.put(call)

    # 创建等待客户端异步回传结果的 Transport。
    transport = PendingClientToolTransport(sender)
    # 创建 Skill 运行时，并注入测试 Skill、Tool 定义和四类 Tool Adapter。
    runtime = SkillRuntime(
        # 提供测试 SKILL.md 的内存 Skill Repository。
        skill_repository=MemorySkillRepository(),
        # 提供 Function、脚本、客户端和 MCP 四个 Tool 定义。
        tool_repository=MemoryToolRepository(resolved_tools()),
        # ToolFactory 根据每个 Tool 的类型选择对应 Adapter。
        tool_factory=ToolFactory(
            [
                # 将 registry_key 指向的 Python Function 转为 LangChain Tool。
                PythonFunctionAdapter(registry),
                # 将服务端脚本 Tool 委派给测试 Script Executor。
                ServerScriptAdapter(TestScriptExecutor()),
                # 将客户端 Tool 请求委派给等待异步结果的 Transport。
                ClientJavascriptAdapter(transport),
                # 从本地 MCP Server 发现并包装 MCP Tool。
                McpToolAdapter(LangChainMcpToolProvider()),
            ]
        ),
    )
    # 创建本次 Skill 编译和执行使用的上下文。
    context = CompileContext(
        # tenant_id 会传递给服务端脚本执行器。
        tenant_id="tenant-1",
        # user_id 表示当前调用用户。
        user_id="user-1",
        # session_id 用于关联客户端 Tool 请求和结果。
        session_id="session-1",
        # 声明当前客户端具备 client.export.file 这个能力及对应版本。
        client_capabilities=(
            ClientCapability(
                # 客户端预注册的 Tool 标识。
                tool_id="client.export.file",
                # 客户端 Tool 版本必须与定义版本一致。
                version="1.0.0",
            ),
        ),
    )

    # 编译测试 Skill，得到系统提示词和四个可执行 LangChain Tool。
    bundle = await runtime.compile("heterogeneous-skill", context)

    # 验证四个 Tool 均已构建成功，并保持配置中的排序。
    assert [tool.name for tool in bundle.tools] == [
        # 第一个是 Python Function Tool。
        "add_numbers",
        # 第二个是服务端脚本 Tool。
        "generate_report",
        # 第三个是客户端 JavaScript Tool。
        "export_client_file",
        # 第四个是 MCP Tool。
        "mcp_echo",
    ]
    # 验证 SKILL.md 的 Markdown 指令已进入系统提示词。
    assert "# 异构工具测试技能" in bundle.system_prompt
    # 验证编译过程中没有缺失 Tool 或跳过 Tool 等诊断信息。
    assert bundle.diagnostics == ()
    # 调用 Function Tool，验证 2 + 5 的真实执行结果是 7。
    assert await bundle.tools[0].ainvoke({"a": 2, "b": 5}) == 7
    # 调用服务端脚本 Tool，验证 Executor 返回约定的结构化结果。
    assert await bundle.tools[1].ainvoke({"title": "日报"}) == {
        # status 表示报告已生成。
        "status": "generated",
        # title 应原样返回调用参数中的日报标题。
        "title": "日报",
    }

    # 后台启动客户端 Tool 调用，因为它需要等待客户端稍后回传结果。
    client_invocation = asyncio.create_task(bundle.tools[2].ainvoke({"format": "xlsx"}))
    # 从模拟发送队列取得服务端生成的客户端 Tool 请求。
    client_request = await client_requests.get()
    # 模拟客户端执行完成，并把带相同 call_id 的成功结果交还 Transport。
    await transport.accept_result(
        ClientToolResult(
            # 使用请求中的 call_id，将结果关联到正在等待的调用。
            call_id=client_request.call_id,
            # session_id 必须与原始客户端会话一致。
            session_id=client_request.session_id,
            # tool_id 必须与原始请求的客户端能力一致。
            tool_id=client_request.tool_id,
            # tool_version 必须与原始请求版本一致。
            tool_version=client_request.tool_version,
            # SUCCESS 表示客户端 Tool 执行成功。
            status=ClientToolStatus.SUCCESS,
            # output 是客户端 Tool 返回给 Agent 的结构化结果。
            output={"file_name": "测试文件.xlsx"},
        )
    )
    # 等待后台调用结束，并验证它取得了对应客户端返回结果。
    assert await client_invocation == {"file_name": "测试文件.xlsx"}

    # 调用本地 stdio MCP Server 暴露的回显工具。
    mcp_result = await bundle.tools[3].ainvoke({"text": "hello"})
    # 验证 MCP 返回的第一段内容是文本类型。
    assert mcp_result[0]["type"] == "text"
    # 验证 MCP Server 返回了预期的 mcp:hello 文本。
    assert mcp_result[0]["text"] == "mcp:hello"

    # 创建离线 Fake Model，预设它先调用 add_numbers，再输出最终回复。
    model = ToolBindingFakeModel(
        # messages 按 Agent 两轮模型响应的顺序提供。
        messages=iter(
            [
                # 第一轮响应要求 Agent 调用 add_numbers Tool。
                AIMessage(
                    # 本轮只发起 Tool Call，因此普通文本内容为空。
                    content="",
                    # tool_calls 描述模型选择的工具、参数和调用标识。
                    tool_calls=[
                        {
                            # 指定调用编译后的 add_numbers Tool。
                            "name": "add_numbers",
                            # 传给 Tool 的业务参数，预期结果为 7。
                            "args": {"a": 3, "b": 4},
                            # Tool Call 标识用于关联 Agent 消息。
                            "id": "agent-tool-call-1",
                            # 声明这是一条标准 Tool Call。
                            "type": "tool_call",
                        }
                    ],
                ),
                # 第二轮响应表示模型收到 Tool 结果后给出的最终回复。
                AIMessage(content="装配成功"),
            ]
        )
    )
    # 使用编译结果创建 LangChain Agent。
    agent = create_agent(
        # 注入不访问网络的 Fake Model。
        model=model,
        # 注入从 SKILL 编译得到的四个 Tool。
        tools=list(bundle.tools),
        # 注入从 SKILL.md 正文编译得到的系统提示词。
        system_prompt=bundle.system_prompt,
    )
    # 向 Agent 发送测试用户消息，触发完整的模型选 Tool 和执行链路。
    agent_result = await agent.ainvoke(
        # messages 是 LangChain Agent 接收的标准对话输入。
        {"messages": [{"role": "user", "content": "验证装配"}]}
    )
    # 验证 Agent 消息中包含 add_numbers 返回的 Tool 结果 7。
    assert any(
        # 找到类型为 tool 且内容为 7 的消息即可证明真实 Tool Call 已完成。
        message.type == "tool" and message.content == "7"
        # 遍历本次 Agent 执行产生的全部消息。
        for message in agent_result["messages"]
    )
    # 验证 Agent 最后一条消息是 Fake Model 预设的最终回复。
    assert agent_result["messages"][-1].content == "装配成功"
