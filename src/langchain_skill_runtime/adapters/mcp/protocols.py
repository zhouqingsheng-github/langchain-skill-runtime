"""MCP provider, server configuration and URL policy protocols."""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from langchain_core.tools import BaseTool

from langchain_skill_runtime.models.context import CompileContext


class McpToolProvider(Protocol):
    """Discover one whitelisted LangChain Tool from an MCP server."""

    async def get_tool(
        self,
        server_name: str,
        tool_name: str,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> BaseTool | None: ...


class McpToolCollectionProvider(Protocol):
    """Discover all LangChain Tools exposed by one MCP server."""

    async def get_tools(
        self,
        server_name: str,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> list[BaseTool]: ...


class McpServerConfigProvider(Protocol):
    """Resolve one host-approved MCP server configuration by reference."""

    async def resolve(
        self,
        reference: str,
        context: CompileContext,
    ) -> Mapping[str, Any]: ...


class McpUrlPolicy(Protocol):
    """Authorize a resolved remote MCP URL before any network request."""

    async def validate(
        self,
        url: str,
        context: CompileContext,
    ) -> tuple[str, ...]: ...


AddressResolver = Callable[[str, int], Awaitable[tuple[str, ...]]]
