from collections.abc import Mapping
from typing import Any

import pytest
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from langchain_skill_runtime.adapters.mcp import (
    LangChainMcpToolProvider,
    McpToolAdapter,
)
from langchain_skill_runtime.errors import (
    ToolDefinitionError,
    ToolExecutionError,
    ToolUnavailableError,
)
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


class FakeServerConfigProvider:
    async def resolve(
        self,
        reference: str,
        context: CompileContext,
    ) -> Mapping[str, Any]:
        assert reference == "mcp/echo"
        assert context.tenant_id == "tenant-1"
        return {
            "transport": "http",
            "url": "https://example.invalid/mcp",
            "headers": {
                "Authorization": {"secret_ref": "mcp/test/token"},
            },
        }


class FakeMcpClient:
    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        assert server_name == "echo"
        return [mcp_echo]


class DefinitionFailingMcpClient:
    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        del server_name
        raise ToolDefinitionError("secret-token /internal/mcp/discovery")


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


@pytest.mark.asyncio
async def test_langchain_provider_preserves_legacy_positional_client_factory() -> None:
    created_connections: list[dict[str, Mapping[str, Any]]] = []

    def client_factory(
        connections: dict[str, Mapping[str, Any]],
    ) -> FakeMcpClient:
        created_connections.append(connections)
        return FakeMcpClient()

    provider = LangChainMcpToolProvider(FakeSecretProvider(), client_factory)

    built = await provider.get_tool(
        "echo",
        "mcp_echo",
        {"transport": "stdio", "command": "python"},
        CompileContext(tenant_id="tenant-1"),
    )

    assert built is mcp_echo
    assert created_connections == [
        {"echo": {"transport": "stdio", "command": "python"}}
    ]


@pytest.mark.asyncio
async def test_langchain_provider_resolves_registered_server_reference() -> None:
    created_connections: list[dict[str, Mapping[str, Any]]] = []

    def client_factory(
        connections: dict[str, Mapping[str, Any]],
    ) -> FakeMcpClient:
        created_connections.append(connections)
        return FakeMcpClient()

    provider = LangChainMcpToolProvider(
        secret_provider=FakeSecretProvider(),
        server_config_provider=FakeServerConfigProvider(),
        client_factory=client_factory,
    )

    built = await provider.get_tool(
        "echo",
        "mcp_echo",
        {"server_ref": "mcp/echo"},
        CompileContext(tenant_id="tenant-1"),
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


@pytest.mark.asyncio
async def test_langchain_provider_requires_server_config_provider_for_reference() -> (
    None
):
    provider = LangChainMcpToolProvider()

    with pytest.raises(ToolDefinitionError, match="McpServerConfigProvider"):
        await provider.get_tool(
            "echo",
            "mcp_echo",
            {"server_ref": "mcp/echo"},
            CompileContext(),
        )


class FailingSecretProvider:
    async def resolve(self, reference: str, context: CompileContext) -> str:
        del reference, context
        raise RuntimeError("resolved-secret-token /internal/secret/path")


class DefinitionFailingSecretProvider:
    async def resolve(self, reference: str, context: CompileContext) -> str:
        del reference, context
        raise ToolDefinitionError("resolved-secret-token /internal/secret/path")


class FailingServerConfigProvider:
    async def resolve(
        self,
        reference: str,
        context: CompileContext,
    ) -> Mapping[str, Any]:
        del reference, context
        raise RuntimeError("resolved-secret-token /internal/server/path")


@pytest.mark.asyncio
async def test_langchain_provider_sanitizes_server_config_provider_failure() -> None:
    provider = LangChainMcpToolProvider(
        server_config_provider=FailingServerConfigProvider()
    )

    with pytest.raises(ToolUnavailableError) as captured:
        await provider.get_tool(
            "echo",
            "mcp_echo",
            {"server_ref": "mcp/echo"},
            CompileContext(),
        )

    error_text = str(captured.value)
    assert "resolved-secret-token" not in error_text
    assert "/internal/server/path" not in error_text


@pytest.mark.parametrize(
    "server_config",
    [
        {
            "transport": "http",
            "url": "https://example.invalid/mcp",
            "headers": {"Authorization": "Bearer plaintext-token"},
        },
        {
            "transport": "stdio",
            "command": "python",
            "env": {"API_KEY": "plaintext-token"},
        },
        {
            "transport": "http",
            "url": "https://user:plaintext-token@example.invalid/mcp",
        },
        {
            "transport": "http",
            "url": "https://example.invalid/mcp?token=plaintext-token",
        },
    ],
)
@pytest.mark.asyncio
async def test_langchain_provider_rejects_plaintext_credentials(
    server_config: Mapping[str, Any],
) -> None:
    provider = LangChainMcpToolProvider()

    with pytest.raises(ToolDefinitionError, match="明文"):
        await provider.get_tool(
            "echo",
            "mcp_echo",
            server_config,
            CompileContext(),
        )


@pytest.mark.parametrize(
    "server_config",
    [
        {
            "transport": "http",
            "url": "https://example.invalid/mcp?api-version=2026-08-12",
            "headers": {"X-Tenant-ID": "tenant-1"},
        },
        {
            "transport": "stdio",
            "command": "python",
            "env": {"LOG_LEVEL": "INFO"},
        },
    ],
)
@pytest.mark.asyncio
async def test_langchain_provider_allows_non_sensitive_literal_configuration(
    server_config: Mapping[str, Any],
) -> None:
    created_connections: list[dict[str, Mapping[str, Any]]] = []

    def client_factory(
        connections: dict[str, Mapping[str, Any]],
    ) -> FakeMcpClient:
        created_connections.append(connections)
        return FakeMcpClient()

    provider = LangChainMcpToolProvider(client_factory=client_factory)

    built = await provider.get_tool(
        "echo",
        "mcp_echo",
        server_config,
        CompileContext(),
    )

    assert built is mcp_echo
    assert created_connections == [{"echo": dict(server_config)}]


@pytest.mark.asyncio
async def test_langchain_provider_sanitizes_secret_provider_failure() -> None:
    provider = LangChainMcpToolProvider(secret_provider=FailingSecretProvider())

    with pytest.raises(ToolUnavailableError) as captured:
        await provider.get_tool(
            "echo",
            "mcp_echo",
            {
                "transport": "http",
                "url": "https://example.invalid/mcp",
                "headers": {
                    "Authorization": {"secret_ref": "mcp/test/token"},
                },
            },
            CompileContext(),
        )

    error_text = str(captured.value)
    assert "resolved-secret-token" not in error_text
    assert "/internal/secret/path" not in error_text


@pytest.mark.asyncio
async def test_langchain_provider_sanitizes_secret_provider_definition_error() -> None:
    provider = LangChainMcpToolProvider(
        secret_provider=DefinitionFailingSecretProvider()
    )

    with pytest.raises(ToolUnavailableError) as captured:
        await provider.get_tool(
            "echo",
            "mcp_echo",
            {"headers": {"Authorization": {"secret_ref": "mcp/test/token"}}},
            CompileContext(),
        )

    error_text = str(captured.value)
    assert "resolved-secret-token" not in error_text
    assert "/internal/secret/path" not in error_text


@pytest.mark.asyncio
async def test_langchain_provider_sanitizes_mcp_discovery_definition_error() -> None:
    provider = LangChainMcpToolProvider(
        client_factory=lambda connections: DefinitionFailingMcpClient()
    )

    with pytest.raises(ToolUnavailableError) as captured:
        await provider.get_tool(
            "echo",
            "mcp_echo",
            {"transport": "stdio", "command": "python"},
            CompileContext(),
        )

    error_text = str(captured.value)
    assert "secret-token" not in error_text
    assert "/internal/mcp/discovery" not in error_text


@tool
async def mcp_echo_with_privilege(text: str, privileged: bool = False) -> str:
    """Echo text and expose a remote-only privileged argument."""

    return f"mcp:{text}:{privileged}"


@pytest.mark.asyncio
async def test_mcp_adapter_enforces_repository_input_schema() -> None:
    built = await McpToolAdapter(RecordingProvider(mcp_echo_with_privilege)).build(
        definition(), CompileContext()
    )

    with pytest.raises(ValidationError, match="privileged"):
        await built.ainvoke({"text": "ok", "privileged": True})


@tool
async def mcp_echo_requiring_scope(text: str, scope: str) -> str:
    """Echo text but require an unbound remote argument."""

    return f"mcp:{scope}:{text}"


@pytest.mark.asyncio
async def test_mcp_adapter_rejects_remote_required_argument_not_in_binding() -> None:
    with pytest.raises(ToolDefinitionError, match="scope"):
        await McpToolAdapter(RecordingProvider(mcp_echo_requiring_scope)).build(
            definition(), CompileContext()
        )


@tool
async def failing_mcp_echo(text: str) -> str:
    """Fail with an intentionally sensitive implementation message."""

    del text
    raise RuntimeError("secret-token /internal/mcp/path")


@pytest.mark.asyncio
async def test_mcp_adapter_sanitizes_execution_failure() -> None:
    built = await McpToolAdapter(RecordingProvider(failing_mcp_echo)).build(
        definition(), CompileContext()
    )

    with pytest.raises(ToolExecutionError) as captured:
        await built.ainvoke({"text": "ok"})

    assert "secret-token" not in str(captured.value)
    assert "/internal/mcp/path" not in str(captured.value)
