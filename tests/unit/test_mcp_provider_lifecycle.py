import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.tools import BaseTool, tool

from langchain_skill_runtime.adapters.mcp import LangChainMcpToolProvider
from langchain_skill_runtime.errors import ToolUnavailableError
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


class StatelessClient:
    def __init__(self) -> None:
        self.calls = 0

    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        self.calls += 1
        return [mcp_echo]


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


@pytest.mark.asyncio
async def test_failed_discovery_closes_session_and_allows_retry() -> None:
    client = StatefulClient()
    attempts = 0

    async def flaky_loader(session: StatefulSession) -> list[BaseTool]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("discovery failed")
        return await load_stateful_tools(session)

    provider = LangChainMcpToolProvider(
        client_factory=lambda _connections: client,
        session_tool_loader=flaky_loader,
    )

    async with provider:
        with pytest.raises(ToolUnavailableError, match="MCP 工具发现失败"):
            await provider.get_tools("stateful", STDIO_CONFIG, CompileContext())
        assert client.closed == 1
        tools = await provider.get_tools("stateful", STDIO_CONFIG, CompileContext())
        assert {item.name for item in tools} == {"state_write", "state_read"}

    assert attempts == 2
    assert client.opened == 2
    assert client.closed == 2


@pytest.mark.asyncio
async def test_unmanaged_provider_keeps_temporary_loading() -> None:
    client = StatelessClient()
    provider = LangChainMcpToolProvider(client_factory=lambda _connections: client)

    tools = await provider.get_tools("echo", STDIO_CONFIG, CompileContext())

    assert tools == [mcp_echo]
    assert client.calls == 1
