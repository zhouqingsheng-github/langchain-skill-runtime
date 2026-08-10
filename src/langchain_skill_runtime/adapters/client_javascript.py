"""Client JavaScript proxy Tool adapter."""

import asyncio
from typing import Any

from langchain_core.tools import BaseTool

from langchain_skill_runtime.adapters.invocation import ToolInvocationGuard
from langchain_skill_runtime.adapters.structured import (
    SchemaValidatedStructuredTool,
    explicit_default_fields,
)
from langchain_skill_runtime.client.models import ClientToolRequest
from langchain_skill_runtime.client.transport import ClientToolTransport
from langchain_skill_runtime.errors import (
    ClientToolError,
    ClientToolTimeoutError,
    ToolDefinitionError,
    ToolUnavailableError,
)
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType
from langchain_skill_runtime.schemas.json_schema import JsonSchemaModelFactory


class ClientJavascriptAdapter:
    """Build an async Tool that delegates to a registered client capability."""

    tool_type = ToolType.CLIENT_JAVASCRIPT

    def __init__(
        self,
        transport: ClientToolTransport,
        schema_factory: JsonSchemaModelFactory | None = None,
    ) -> None:
        self._transport = transport
        self._schema_factory = schema_factory or JsonSchemaModelFactory()

    async def build(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> BaseTool:
        if definition.tool_type is not self.tool_type:
            raise ToolDefinitionError("ClientJavascriptAdapter 收到错误的 Tool 类型")

        tool_key = definition.execution_config.get("tool_key")
        if not isinstance(tool_key, str) or not tool_key.strip():
            raise ToolDefinitionError("Client JavaScript 必须配置 tool_key")
        if not context.session_id:
            raise ToolUnavailableError("客户端 Tool 需要 session_id")

        capability = next(
            (
                item
                for item in context.client_capabilities
                if item.tool_id == tool_key and item.version == definition.version
            ),
            None,
        )
        if capability is None or not capability.available:
            raise ToolUnavailableError("客户端 capability 不存在或版本不匹配")
        if not await self._transport.is_available(
            context.session_id,
            tool_key,
            definition.version,
        ):
            raise ToolUnavailableError("客户端 Tool 当前不可用")

        args_model = self._schema_factory.create(
            f"{definition.name.title().replace('_', '')}Input",
            definition.input_schema,
        )
        guard = ToolInvocationGuard(definition)

        async def execute_client_tool(**arguments: Any) -> Any:
            async def execute() -> Any:
                try:
                    async with asyncio.timeout(float(definition.timeout_seconds)):
                        result = await self._transport.invoke(
                            ClientToolRequest(
                                session_id=context.session_id or "",
                                tool_id=tool_key,
                                tool_version=definition.version,
                                arguments=arguments,
                                timeout_seconds=definition.timeout_seconds,
                            )
                        )
                except TimeoutError:
                    raise ClientToolTimeoutError("客户端 Tool 执行超时") from None
                except ClientToolError as exc:
                    raise type(exc)("客户端 Tool 执行失败") from None
                return result.output

            return await guard.invoke(
                execute,
                enforce_timeout=False,
                passthrough_errors=(ClientToolError,),
            )

        return SchemaValidatedStructuredTool.from_function(
            coroutine=execute_client_tool,
            name=definition.name,
            description=definition.description,
            args_schema=args_model,
            explicit_default_fields=explicit_default_fields(definition.input_schema),
        )
