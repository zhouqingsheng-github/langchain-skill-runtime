"""MCP Tool discovery and LangChain adapter."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast
from urllib.parse import parse_qsl, urlsplit

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from langchain_skill_runtime.adapters.invocation import ToolInvocationGuard
from langchain_skill_runtime.adapters.structured import (
    SchemaValidatedStructuredTool,
    explicit_default_fields,
)
from langchain_skill_runtime.errors import ToolDefinitionError, ToolUnavailableError
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType
from langchain_skill_runtime.schemas.json_schema import JsonSchemaModelFactory
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


class McpServerConfigProvider(Protocol):
    """Resolve one host-approved MCP server configuration by reference."""

    async def resolve(
        self,
        reference: str,
        context: CompileContext,
    ) -> Mapping[str, Any]: ...


class _McpClient(Protocol):
    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]: ...


McpClientFactory = Callable[[dict[str, Mapping[str, Any]]], _McpClient]


class LangChainMcpToolProvider:
    """Use langchain-mcp-adapters while keeping it an optional dependency."""

    def __init__(
        self,
        secret_provider: SecretProvider | None = None,
        client_factory: McpClientFactory | None = None,
        *,
        server_config_provider: McpServerConfigProvider | None = None,
    ) -> None:
        self._secret_provider = secret_provider
        self._server_config_provider = server_config_provider
        self._client_factory = client_factory or self._default_client_factory

    async def get_tool(
        self,
        server_name: str,
        tool_name: str,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> BaseTool | None:
        trusted_config = await self._resolve_server_config(server_config, context)
        self._validate_secret_references(trusted_config)
        try:
            resolved = await self._resolve_value(trusted_config, context)
        except ToolDefinitionError:
            raise
        except Exception:  # noqa: BLE001 - sanitize SecretProvider failures
            raise ToolUnavailableError("MCP Secret 解析失败") from None
        if not isinstance(resolved, Mapping):
            raise ToolDefinitionError("MCP server 配置必须是对象")

        try:
            client = self._client_factory({server_name: dict(resolved)})
            tools = await client.get_tools(server_name=server_name)
        except Exception:  # noqa: BLE001 - sanitize arbitrary MCP client failures
            raise ToolUnavailableError("MCP 工具发现失败") from None
        return next((item for item in tools if item.name == tool_name), None)

    async def _resolve_server_config(
        self,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> dict[str, Any]:
        if "server_ref" not in server_config:
            return dict(server_config)
        if set(server_config) != {"server_ref"}:
            raise ToolDefinitionError("MCP server_ref 配置不能包含其他字段")

        reference = server_config.get("server_ref")
        if not isinstance(reference, str) or not reference.strip():
            raise ToolDefinitionError("MCP server_ref 不能为空")
        if self._server_config_provider is None:
            raise ToolDefinitionError("MCP server_ref 缺少 McpServerConfigProvider")
        try:
            resolved = await self._server_config_provider.resolve(reference, context)
        except Exception:  # noqa: BLE001 - sanitize external provider failures
            raise ToolUnavailableError("MCP Server 配置解析失败") from None
        if not isinstance(resolved, Mapping):
            raise ToolDefinitionError("MCP Server 配置必须是对象")
        return dict(resolved)

    @classmethod
    def _validate_secret_references(cls, server_config: Mapping[str, Any]) -> None:
        url = server_config.get("url")
        if url is not None and not cls._is_secret_ref(url):
            if not isinstance(url, str):
                raise ToolDefinitionError("MCP URL 配置非法")
            parsed = urlsplit(url)
            sensitive_query = any(
                cls._is_sensitive_name(name)
                for name, _ in parse_qsl(parsed.query, keep_blank_values=True)
            )
            sensitive_fragment = cls._is_sensitive_name(parsed.fragment)
            if (
                parsed.username
                or parsed.password
                or sensitive_query
                or sensitive_fragment
            ):
                raise ToolDefinitionError("MCP URL 不允许包含明文凭据")

        headers = server_config.get("headers")
        if headers is not None:
            if not isinstance(headers, Mapping):
                raise ToolDefinitionError("MCP headers 配置必须是对象")
            for name, value in headers.items():
                if cls._is_sensitive_name(name) and not cls._is_secret_ref(value):
                    raise ToolDefinitionError(
                        "MCP Header 凭据不允许使用明文，必须配置 secret_ref"
                    )

        environment = server_config.get("env")
        if environment is not None:
            if not isinstance(environment, Mapping):
                raise ToolDefinitionError("MCP env 配置必须是对象")
            for name, value in environment.items():
                if cls._is_sensitive_name(name) and not cls._is_secret_ref(value):
                    raise ToolDefinitionError(
                        "MCP 环境变量凭据不允许使用明文，必须配置 secret_ref"
                    )

        cls._validate_nested_sensitive_values(server_config)

    @classmethod
    def _validate_nested_sensitive_values(cls, value: Any) -> None:
        if cls._is_secret_ref(value):
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if cls._is_sensitive_name(key) and not cls._is_secret_ref(item):
                    raise ToolDefinitionError(
                        "MCP 敏感配置不允许使用明文，必须配置 secret_ref"
                    )
                cls._validate_nested_sensitive_values(item)
        elif isinstance(value, list):
            for item in value:
                cls._validate_nested_sensitive_values(item)

    @staticmethod
    def _is_secret_ref(value: Any) -> bool:
        return (
            isinstance(value, Mapping)
            and set(value) == {"secret_ref"}
            and isinstance(value.get("secret_ref"), str)
            and bool(value["secret_ref"].strip())
        )

    @staticmethod
    def _is_sensitive_name(value: Any) -> bool:
        normalized = str(value).casefold()
        return any(
            marker in normalized
            for marker in (
                "api-key",
                "api_key",
                "authorization",
                "cookie",
                "credential",
                "password",
                "secret",
                "token",
            )
        )

    async def _resolve_value(self, value: Any, context: CompileContext) -> Any:
        if isinstance(value, Mapping):
            if set(value) == {"secret_ref"}:
                reference = value.get("secret_ref")
                if not isinstance(reference, str) or not reference.strip():
                    raise ToolDefinitionError("MCP secret_ref 不能为空")
                if self._secret_provider is None:
                    raise ToolDefinitionError("MCP Secret 引用缺少 SecretProvider")
                try:
                    return await self._secret_provider.resolve(reference, context)
                except Exception:  # noqa: BLE001 - sanitize external provider failures
                    raise ToolUnavailableError("MCP Secret 解析失败") from None
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

    def __init__(
        self,
        provider: McpToolProvider,
        schema_factory: JsonSchemaModelFactory | None = None,
    ) -> None:
        self._provider = provider
        self._schema_factory = schema_factory or JsonSchemaModelFactory()

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
        server_ref = definition.execution_config.get("server_ref")
        if not isinstance(server_name, str) or not server_name.strip():
            raise ToolDefinitionError("MCP Tool 必须配置 server_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ToolDefinitionError("MCP Tool 必须配置 tool_name")
        if server is not None and server_ref is not None:
            raise ToolDefinitionError("MCP Tool 不能同时配置 server 和 server_ref")
        if server_ref is not None:
            if not isinstance(server_ref, str) or not server_ref.strip():
                raise ToolDefinitionError("MCP Tool server_ref 不能为空")
            server_config: Mapping[str, Any] = {"server_ref": server_ref}
        elif isinstance(server, Mapping):
            server_config = server
        else:
            raise ToolDefinitionError("MCP Tool 必须配置 server 或 server_ref")

        discovered = await self._provider.get_tool(
            server_name,
            tool_name,
            server_config,
            context,
        )
        if discovered is None:
            raise ToolUnavailableError(f"MCP Tool 不可用: {tool_name}")
        self._validate_remote_schema(definition, discovered)
        args_model = self._schema_factory.create(
            f"{definition.name.title().replace('_', '')}Input",
            definition.input_schema,
        )
        guard = ToolInvocationGuard(definition)

        async def invoke_mcp(**arguments: Any) -> Any:
            async def execute() -> Any:
                return await discovered.ainvoke(arguments)

            return await guard.invoke(execute)

        return SchemaValidatedStructuredTool.from_function(
            coroutine=invoke_mcp,
            name=definition.name,
            description=definition.description,
            args_schema=args_model,
            explicit_default_fields=explicit_default_fields(definition.input_schema),
        )

    @staticmethod
    def _validate_remote_schema(
        definition: ResolvedToolDefinition,
        discovered: BaseTool,
    ) -> None:
        remote_args = discovered.args_schema
        if isinstance(remote_args, dict):
            remote_schema = remote_args
        elif isinstance(remote_args, type) and issubclass(remote_args, BaseModel):
            remote_schema = remote_args.model_json_schema()
        else:
            raise ToolDefinitionError("MCP Tool 缺少可校验的输入 Schema")

        bound_properties = definition.input_schema.get("properties", {})
        remote_properties = remote_schema.get("properties", {})
        if not isinstance(bound_properties, dict) or not isinstance(
            remote_properties, dict
        ):
            raise ToolDefinitionError("MCP Tool 输入 Schema properties 非法")

        missing_remote_fields = set(bound_properties) - set(remote_properties)
        if missing_remote_fields:
            names = ", ".join(sorted(missing_remote_fields))
            raise ToolDefinitionError(f"MCP Tool 远端缺少绑定参数: {names}")

        bound_required = set(definition.input_schema.get("required", []))
        remote_required = set(remote_schema.get("required", []))
        uncovered_required = remote_required - bound_required
        if uncovered_required:
            names = ", ".join(sorted(uncovered_required))
            raise ToolDefinitionError(f"MCP Tool 存在未绑定的必填参数: {names}")

        for name, bound_property in bound_properties.items():
            remote_property = remote_properties[name]
            if not isinstance(bound_property, dict) or not isinstance(
                remote_property, dict
            ):
                raise ToolDefinitionError(f"MCP Tool 参数 Schema 非法: {name}")
            bound_type = bound_property.get("type")
            remote_type = remote_property.get("type")
            if (
                bound_type is not None
                and remote_type is not None
                and bound_type != remote_type
            ):
                raise ToolDefinitionError(f"MCP Tool 参数类型不兼容: {name}")
