"""MCP Tool discovery and LangChain adapter."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

from langchain_core.tools import BaseTool

from langchain_skill_runtime.errors import ToolDefinitionError, ToolUnavailableError
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType
from langchain_skill_runtime.secrets import SecretProvider


class McpToolProvider(Protocol):
    """Discover one whitelisted LangChain Tool from an MCP server."""

    async def get_tool(
        self,
        server_name: str,
        tool_name: str,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> BaseTool | None: ...


class _McpClient(Protocol):
    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]: ...


McpClientFactory = Callable[[dict[str, Mapping[str, Any]]], _McpClient]


class LangChainMcpToolProvider:
    """Use langchain-mcp-adapters while keeping it an optional dependency."""

    def __init__(
        self,
        secret_provider: SecretProvider | None = None,
        client_factory: McpClientFactory | None = None,
    ) -> None:
        self._secret_provider = secret_provider
        self._client_factory = client_factory or self._default_client_factory

    async def get_tool(
        self,
        server_name: str,
        tool_name: str,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> BaseTool | None:
        resolved = await self._resolve_value(dict(server_config), context)
        if not isinstance(resolved, Mapping):
            raise ToolDefinitionError("MCP server 配置必须是对象")

        try:
            client = self._client_factory({server_name: dict(resolved)})
            tools = await client.get_tools(server_name=server_name)
        except ToolDefinitionError:
            raise
        except Exception:  # noqa: BLE001 - sanitize arbitrary MCP client failures
            raise ToolUnavailableError("MCP 工具发现失败") from None
        return next((item for item in tools if item.name == tool_name), None)

    async def _resolve_value(self, value: Any, context: CompileContext) -> Any:
        if isinstance(value, Mapping):
            if set(value) == {"secret_ref"}:
                reference = value.get("secret_ref")
                if not isinstance(reference, str) or not reference.strip():
                    raise ToolDefinitionError("MCP secret_ref 不能为空")
                if self._secret_provider is None:
                    raise ToolDefinitionError("MCP Secret 引用缺少 SecretProvider")
                return await self._secret_provider.resolve(reference, context)
            return {
                str(key): await self._resolve_value(item, context)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [await self._resolve_value(item, context) for item in value]
        return value

    @staticmethod
    def _default_client_factory(
        connections: dict[str, Mapping[str, Any]],
    ) -> _McpClient:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            raise ToolUnavailableError(
                "使用 MCP Tool 需要安装 langchain-skill-runtime[mcp]"
            ) from None
        return cast(_McpClient, MultiServerMCPClient(cast(Any, connections)))


class McpToolAdapter:
    """Build one explicitly configured MCP Tool."""

    tool_type = ToolType.MCP

    def __init__(self, provider: McpToolProvider) -> None:
        self._provider = provider

    async def build(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> BaseTool:
        if definition.tool_type is not self.tool_type:
            raise ToolDefinitionError("McpToolAdapter 收到错误的 Tool 类型")

        server_name = definition.execution_config.get("server_name")
        tool_name = definition.execution_config.get("tool_name")
        server = definition.execution_config.get("server")
        if not isinstance(server_name, str) or not server_name.strip():
            raise ToolDefinitionError("MCP Tool 必须配置 server_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ToolDefinitionError("MCP Tool 必须配置 tool_name")
        if not isinstance(server, Mapping):
            raise ToolDefinitionError("MCP Tool 必须配置 server 对象")

        discovered = await self._provider.get_tool(
            server_name,
            tool_name,
            server,
            context,
        )
        if discovered is None:
            raise ToolUnavailableError(f"MCP Tool 不可用: {tool_name}")
        if (
            discovered.name != definition.name
            or discovered.description != definition.description
        ):
            return discovered.model_copy(
                update={
                    "name": definition.name,
                    "description": definition.description,
                }
            )
        return discovered
