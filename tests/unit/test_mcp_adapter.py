from collections.abc import Mapping
from typing import Any

import httpx
import pytest
from jsonschema import ValidationError as JsonSchemaValidationError
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from langchain_skill_runtime.adapters.mcp import (
    AllowHostsMcpUrlPolicy,
    LangChainMcpToolProvider,
    McpToolAdapter,
    McpUrlPolicy,
    PublicHttpsMcpUrlPolicy,
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


@tool
async def mcp_length(text: str) -> int:
    """Return text length through MCP."""

    return len(text)


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


class FakeMcpCollectionClient:
    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        assert server_name == "amap"
        return [mcp_echo, mcp_length]


class AllowingUrlPolicy(McpUrlPolicy):
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def validate(
        self,
        url: str,
        context: CompileContext,
    ) -> tuple[str, ...]:
        del context
        self.urls.append(url)
        return ("8.8.8.8",)


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
        url_policy=AllowingUrlPolicy(),
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
    connection = dict(created_connections[0]["echo"])
    connection.pop("httpx_client_factory")
    assert connection == {
        "transport": "http",
        "url": "https://example.invalid/mcp",
        "headers": {"Authorization": "secret-token-value"},
    }


@pytest.mark.asyncio
async def test_langchain_provider_discovers_all_tools_and_resolves_env_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_connections: list[dict[str, Mapping[str, Any]]] = []

    def client_factory(
        connections: dict[str, Mapping[str, Any]],
    ) -> FakeMcpCollectionClient:
        created_connections.append(connections)
        return FakeMcpCollectionClient()

    monkeypatch.setenv("AMAP_MAPS_API_KEY", "test-amap-key")
    url_policy = AllowingUrlPolicy()
    provider = LangChainMcpToolProvider(
        client_factory=client_factory,
        url_policy=url_policy,
    )

    tools = await provider.get_tools(
        "amap",
        {
            "transport": "streamable_http",
            "url": "https://mcp.amap.com/mcp",
            "query": {"key": {"env": "AMAP_MAPS_API_KEY"}},
        },
        CompileContext(),
    )

    assert [item.name for item in tools] == ["mcp_echo", "mcp_length"]
    assert url_policy.urls == ["https://mcp.amap.com/mcp?key=test-amap-key"]
    connection = dict(created_connections[0]["amap"])
    http_client_factory = connection.pop("httpx_client_factory")
    assert connection == {
        "transport": "streamable_http",
        "url": "https://mcp.amap.com/mcp?key=test-amap-key",
    }
    client = http_client_factory()
    assert client.follow_redirects is False
    await client.aclose()


@pytest.mark.asyncio
async def test_langchain_provider_requires_host_policy_for_inline_server() -> None:
    provider = LangChainMcpToolProvider()

    with pytest.raises(ToolDefinitionError, match="McpUrlPolicy"):
        await provider.get_tools(
            "amap",
            {
                "transport": "streamable_http",
                "url": "https://mcp.amap.com/mcp",
            },
            CompileContext(),
        )


@pytest.mark.asyncio
async def test_custom_client_cannot_bypass_inline_url_policy() -> None:
    provider = LangChainMcpToolProvider(
        client_factory=lambda connections: FakeMcpCollectionClient()
    )

    with pytest.raises(ToolDefinitionError, match="McpUrlPolicy"):
        await provider.get_tools(
            "amap",
            {
                "transport": "streamable_http",
                "url": "https://127.0.0.1/mcp",
            },
            CompileContext(),
        )


@pytest.mark.asyncio
async def test_allow_hosts_mcp_url_policy_rejects_unapproved_host() -> None:
    async def public_addresses(host: str, port: int) -> tuple[str, ...]:
        del host, port
        return ("8.8.8.8",)

    policy = AllowHostsMcpUrlPolicy(
        {"mcp.amap.com"},
        resolver=public_addresses,
    )

    with pytest.raises(ToolDefinitionError, match="未经宿主允许"):
        await policy.validate("https://evil.example/mcp", CompileContext())


@pytest.mark.asyncio
async def test_allow_hosts_mcp_url_policy_trusts_approved_host_resolution() -> None:
    async def resolved_address(host: str, port: int) -> tuple[str, ...]:
        assert host == "mcp.amap.com"
        assert port == 443
        return ("198.18.0.104",)

    policy = AllowHostsMcpUrlPolicy(
        {"mcp.amap.com"},
        resolver=resolved_address,
    )

    assert await policy.validate("https://mcp.amap.com/mcp", CompileContext()) == (
        "198.18.0.104",
    )


@pytest.mark.asyncio
async def test_langchain_provider_rejects_plaintext_collection_query_key() -> None:
    provider = LangChainMcpToolProvider(
        client_factory=lambda connections: FakeMcpCollectionClient(),
        url_policy=AllowingUrlPolicy(),
    )

    with pytest.raises(ToolDefinitionError, match="明文"):
        await provider.get_tools(
            "amap",
            {
                "transport": "streamable_http",
                "url": "https://mcp.amap.com/mcp",
                "query": {"key": "plaintext-amap-key"},
            },
            CompileContext(),
        )


@pytest.mark.asyncio
async def test_langchain_provider_sanitizes_missing_collection_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AMAP_MAPS_API_KEY", raising=False)
    provider = LangChainMcpToolProvider(
        client_factory=lambda connections: FakeMcpCollectionClient(),
        url_policy=AllowingUrlPolicy(),
    )

    with pytest.raises(ToolUnavailableError) as captured:
        await provider.get_tools(
            "amap",
            {
                "transport": "streamable_http",
                "url": "https://mcp.amap.com/mcp",
                "query": {"key": {"env": "AMAP_MAPS_API_KEY"}},
            },
            CompileContext(),
        )

    assert "AMAP_MAPS_API_KEY" not in str(captured.value)
    assert "mcp.amap.com" not in str(captured.value)


def collection_definition() -> ResolvedToolDefinition:
    return ResolvedToolDefinition(
        id="amap-maps",
        name="amap_maps",
        description="高德地图 MCP 工具集合",
        tool_type=ToolType.MCP,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        execution_config={
            "server_name": "amap",
            "server": {
                "transport": "streamable_http",
                "url": "https://mcp.amap.com/mcp",
            },
        },
        version="1.0.0",
    )


class RecordingCollectionProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any], CompileContext]] = []

    async def get_tools(
        self,
        server_name: str,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> list[BaseTool]:
        self.calls.append((server_name, server_config, context))
        return [mcp_echo, mcp_length]


@pytest.mark.asyncio
async def test_mcp_adapter_builds_all_tools_from_collection_definition() -> None:
    provider = RecordingCollectionProvider()
    context = CompileContext(tenant_id="tenant-1")

    built = await McpToolAdapter(provider).build_many(
        collection_definition(),
        context,
    )

    assert [item.name for item in built] == ["mcp_echo", "mcp_length"]
    assert await built[0].ainvoke({"text": "ok"}) == "mcp:ok"
    assert await built[1].ainvoke({"text": "abcd"}) == 4
    assert provider.calls == [
        (
            "amap",
            {
                "transport": "streamable_http",
                "url": "https://mcp.amap.com/mcp",
            },
            context,
        )
    ]


@pytest.mark.asyncio
async def test_mcp_collection_preserves_remote_dict_input_schema() -> None:
    remote = mcp_echo.model_copy(
        update={
            "args_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            }
        }
    )

    class DictSchemaProvider:
        async def get_tools(
            self,
            server_name: str,
            server_config: Mapping[str, Any],
            context: CompileContext,
        ) -> list[BaseTool]:
            del server_name, server_config, context
            return [remote]

    built = await McpToolAdapter(DictSchemaProvider()).build_many(
        collection_definition(),
        CompileContext(),
    )

    assert await built[0].ainvoke({"text": "ok"}) == "mcp:ok"
    with pytest.raises(JsonSchemaValidationError):
        await built[0].ainvoke({"text": "ok", "unknown": True})


@pytest.mark.asyncio
async def test_mcp_collection_preserves_complete_remote_json_schema() -> None:
    complex_schema = {
        "$defs": {
            "nullableText": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
            }
        },
        "type": "object",
        "properties": {"text": {"$ref": "#/$defs/nullableText"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    remote = mcp_echo.model_copy(update={"args_schema": complex_schema})

    class ComplexSchemaProvider:
        async def get_tools(
            self,
            server_name: str,
            server_config: Mapping[str, Any],
            context: CompileContext,
        ) -> list[BaseTool]:
            del server_name, server_config, context
            return [remote]

    built = await McpToolAdapter(ComplexSchemaProvider()).build_many(
        collection_definition(),
        CompileContext(),
    )

    assert built[0].args_schema == complex_schema
    assert await built[0].ainvoke({"text": "ok"}) == "mcp:ok"
    with pytest.raises(JsonSchemaValidationError):
        await built[0].ainvoke({"text": 123})


@pytest.mark.asyncio
async def test_default_mcp_url_policy_rejects_private_dns_target() -> None:
    async def private_addresses(host: str, port: int) -> tuple[str, ...]:
        assert host == "mcp.example.com"
        assert port == 443
        return ("10.0.0.8",)

    provider = LangChainMcpToolProvider(
        client_factory=lambda connections: FakeMcpCollectionClient(),
        url_policy=PublicHttpsMcpUrlPolicy(private_addresses),
    )

    with pytest.raises(ToolDefinitionError, match="公网地址"):
        await provider.get_tools(
            "amap",
            {
                "transport": "streamable_http",
                "url": "https://mcp.example.com/mcp",
            },
            CompileContext(),
        )


@pytest.mark.parametrize(
    "address",
    ["224.0.0.1", "ff05::1", "::ffff:127.0.0.1", "198.18.0.104"],
)
@pytest.mark.asyncio
async def test_mcp_url_policy_rejects_non_unicast_addresses(address: str) -> None:
    async def unsafe_addresses(host: str, port: int) -> tuple[str, ...]:
        del host, port
        return (address,)

    policy = PublicHttpsMcpUrlPolicy(unsafe_addresses)

    with pytest.raises(ToolDefinitionError, match="公网地址"):
        await policy.validate("https://mcp.example.com/mcp", CompileContext())


@pytest.mark.asyncio
async def test_bound_mcp_transport_overrides_host_and_preserves_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[httpx.Request] = []

    class CapturingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, request=request, content=b"ok")

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", CapturingTransport)
    factory = LangChainMcpToolProvider._bound_http_client_factory(
        "https://mcp.amap.com:8443/mcp",
        ("8.8.8.8",),
    )

    async with factory(headers={"Host": "evil.internal"}) as client:
        response = await client.get("https://mcp.amap.com:8443/mcp")

    assert response.status_code == 200
    assert captured[0].url.host == "8.8.8.8"
    assert captured[0].headers["host"] == "mcp.amap.com:8443"
    assert captured[0].extensions["sni_hostname"] == "mcp.amap.com"


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
async def test_registered_server_reference_ignores_inline_url_policy() -> None:
    created_connections: list[dict[str, Mapping[str, Any]]] = []
    url_policy = AllowingUrlPolicy()

    def client_factory(
        connections: dict[str, Mapping[str, Any]],
    ) -> FakeMcpClient:
        created_connections.append(connections)
        return FakeMcpClient()

    provider = LangChainMcpToolProvider(
        secret_provider=FakeSecretProvider(),
        server_config_provider=FakeServerConfigProvider(),
        client_factory=client_factory,
        url_policy=url_policy,
    )

    built = await provider.get_tool(
        "echo",
        "mcp_echo",
        {"server_ref": "mcp/echo"},
        CompileContext(tenant_id="tenant-1"),
    )

    assert built is mcp_echo
    assert url_policy.urls == []
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
        {
            "transport": "http",
            "url": "https://example.invalid/mcp",
            "headers": {"X-ApiKey": "plaintext-token"},
        },
        {
            "transport": "stdio",
            "command": "python",
            "env": {"APIKEY": "plaintext-token"},
        },
        {
            "transport": "http",
            "url": "https://example.invalid/mcp?apikey=plaintext-token",
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

    provider = LangChainMcpToolProvider(
        client_factory=client_factory,
        url_policy=(
            AllowingUrlPolicy() if server_config.get("transport") == "http" else None
        ),
    )

    built = await provider.get_tool(
        "echo",
        "mcp_echo",
        server_config,
        CompileContext(),
    )

    assert built is mcp_echo
    connection = dict(created_connections[0]["echo"])
    connection.pop("httpx_client_factory", None)
    assert connection == dict(server_config)


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
