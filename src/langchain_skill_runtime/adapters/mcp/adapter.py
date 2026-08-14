"""Adapter that wraps discovered MCP tools as validated LangChain tools."""

from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from langchain_skill_runtime.adapters.invocation import ToolInvocationGuard
from langchain_skill_runtime.adapters.mcp.protocols import (
    McpToolCollectionProvider,
    McpToolProvider,
)
from langchain_skill_runtime.adapters.structured import (
    SchemaValidatedStructuredTool,
    explicit_default_fields,
)
from langchain_skill_runtime.errors import ToolDefinitionError, ToolUnavailableError
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType
from langchain_skill_runtime.schemas.json_schema import JsonSchemaModelFactory


class McpToolAdapter:
    """Build one explicitly configured MCP Tool or one MCP tool collection."""

    tool_type = ToolType.MCP

    def __init__(
        self,
        provider: McpToolProvider | McpToolCollectionProvider,
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

        get_tool = getattr(self._provider, "get_tool", None)
        if get_tool is None:
            raise ToolDefinitionError("单个 MCP Tool Provider 缺少 get_tool")
        discovered = await get_tool(
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

    async def build_many(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> tuple[BaseTool, ...]:
        """Build a legacy single binding or expand an MCP server collection."""

        tool_name = definition.execution_config.get("tool_name")
        if tool_name is not None:
            return (await self.build(definition, context),)

        server_name, server_config = self._collection_connection(definition)
        get_tools = getattr(self._provider, "get_tools", None)
        if get_tools is None:
            raise ToolDefinitionError("MCP 工具集合 Provider 缺少 get_tools")
        discovered = await get_tools(server_name, server_config, context)
        if not discovered:
            raise ToolUnavailableError("MCP Server 未发现可用工具")
        return tuple(
            self._wrap_collection_tool(definition, remote) for remote in discovered
        )

    @staticmethod
    def _collection_connection(
        definition: ResolvedToolDefinition,
    ) -> tuple[str, Mapping[str, Any]]:
        server_name = definition.execution_config.get("server_name")
        server = definition.execution_config.get("server")
        server_ref = definition.execution_config.get("server_ref")
        if not isinstance(server_name, str) or not server_name.strip():
            raise ToolDefinitionError("MCP Tool 必须配置 server_name")
        if server is not None and server_ref is not None:
            raise ToolDefinitionError("MCP Tool 不能同时配置 server 和 server_ref")
        if server_ref is not None:
            if not isinstance(server_ref, str) or not server_ref.strip():
                raise ToolDefinitionError("MCP Tool server_ref 不能为空")
            return server_name, {"server_ref": server_ref}
        if isinstance(server, Mapping):
            return server_name, server
        raise ToolDefinitionError("MCP 工具集合必须配置 server 或 server_ref")

    @staticmethod
    def _remote_args_schema(discovered: BaseTool) -> type[BaseModel] | dict[str, Any]:
        remote_args = discovered.args_schema
        if isinstance(remote_args, dict):
            try:
                Draft202012Validator.check_schema(remote_args)
            except SchemaError:
                raise ToolDefinitionError("MCP Tool 输入 Schema 非法") from None
            return remote_args
        if isinstance(remote_args, type) and issubclass(remote_args, BaseModel):
            return remote_args
        raise ToolDefinitionError("MCP Tool 缺少可校验的输入 Schema")

    def _wrap_collection_tool(
        self,
        definition: ResolvedToolDefinition,
        discovered: BaseTool,
    ) -> BaseTool:
        args_schema = self._remote_args_schema(discovered)
        guard_definition = definition.model_copy(update={"output_schema": None})
        guard = ToolInvocationGuard(guard_definition)

        async def invoke_mcp(**arguments: Any) -> Any:
            if isinstance(args_schema, dict):
                Draft202012Validator(args_schema).validate(arguments)

            async def execute() -> Any:
                return await discovered.ainvoke(arguments)

            return await guard.invoke(execute)

        return SchemaValidatedStructuredTool.from_function(
            coroutine=invoke_mcp,
            name=discovered.name,
            description=discovered.description,
            args_schema=args_schema,
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
