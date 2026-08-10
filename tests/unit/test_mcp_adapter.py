from collections.abc import Mapping
from typing import Any

import pytest
from langchain_core.tools import BaseTool, tool

from langchain_skill_runtime.adapters.mcp import (
    LangChainMcpToolProvider,
    McpToolAdapter,
)
from langchain_skill_runtime.errors import ToolUnavailableError
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType


@tool
async def mcp_echo(text: str) -> str:
    """Echo text through MCP."""

    return f"mcp:{text}"


def definition() -> ResolvedToolDefinition:
    return ResolvedToolDefinition(
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
    )


class RecordingProvider:
    def __init__(self, returned_tool: BaseTool | None = mcp_echo) -> None:
        self.returned_tool = returned_tool
        self.calls: list[tuple[str, str, Mapping[str, Any], CompileContext]] = []

    async def get_tool(
        self,
        server_name: str,
        tool_name: str,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> BaseTool | None:
        self.calls.append((server_name, tool_name, server_config, context))
        return self.returned_tool


@pytest.mark.asyncio
async def test_mcp_adapter_builds_provider_tool() -> None:
    provider = RecordingProvider()
    context = CompileContext(tenant_id="tenant-1")

    built = await McpToolAdapter(provider).build(definition(), context)

    assert await built.ainvoke({"text": "ok"}) == "mcp:ok"
    assert provider.calls == [
        (
            "echo",
            "mcp_echo",
            {"transport": "stdio", "command": "python", "args": []},
            context,
        )
    ]


@pytest.mark.asyncio
async def test_mcp_adapter_rejects_missing_discovered_tool() -> None:
    with pytest.raises(ToolUnavailableError, match="mcp_echo"):
        await McpToolAdapter(RecordingProvider(None)).build(
            definition(), CompileContext()
        )


class FakeSecretProvider:
    async def resolve(self, reference: str, context: CompileContext) -> str:
        assert reference == "mcp/test/token"
        assert context.tenant_id == "tenant-1"
        return "secret-token-value"


class FakeMcpClient:
    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        assert server_name == "echo"
        return [mcp_echo]


@pytest.mark.asyncio
async def test_langchain_provider_resolves_secret_references() -> None:
    created_connections: list[dict[str, Mapping[str, Any]]] = []

    def client_factory(
        connections: dict[str, Mapping[str, Any]],
    ) -> FakeMcpClient:
        created_connections.append(connections)
        return FakeMcpClient()

    provider = LangChainMcpToolProvider(
        secret_provider=FakeSecretProvider(),
        client_factory=client_factory,
    )
    context = CompileContext(tenant_id="tenant-1")

    built = await provider.get_tool(
        "echo",
        "mcp_echo",
        {
            "transport": "http",
            "url": "https://example.invalid/mcp",
            "headers": {
                "Authorization": {"secret_ref": "mcp/test/token"},
            },
        },
        context,
    )

    assert built is mcp_echo
    assert created_connections == [
        {
            "echo": {
                "transport": "http",
                "url": "https://example.invalid/mcp",
                "headers": {"Authorization": "secret-token-value"},
            }
        }
    ]
