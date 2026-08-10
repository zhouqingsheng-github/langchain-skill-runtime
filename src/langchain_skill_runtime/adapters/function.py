"""Python Function Tool adapter."""

import asyncio
import inspect
from typing import Any

from langchain_core.tools import BaseTool

from langchain_skill_runtime.adapters.invocation import ToolInvocationGuard
from langchain_skill_runtime.adapters.structured import (
    SchemaValidatedStructuredTool,
    explicit_default_fields,
)
from langchain_skill_runtime.errors import (
    FunctionNotRegisteredError,
    ToolDefinitionError,
)
from langchain_skill_runtime.executors.function_registry import FunctionRegistry
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType
from langchain_skill_runtime.schemas.json_schema import JsonSchemaModelFactory


class PythonFunctionAdapter:
    """Build a Tool around a trusted registry callable."""

    tool_type = ToolType.PYTHON_FUNCTION

    def __init__(
        self,
        registry: FunctionRegistry,
        schema_factory: JsonSchemaModelFactory | None = None,
    ) -> None:
        self._registry = registry
        self._schema_factory = schema_factory or JsonSchemaModelFactory()

    async def build(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> BaseTool:
        del context
        if definition.tool_type is not self.tool_type:
            raise ToolDefinitionError("PythonFunctionAdapter 收到错误的 Tool 类型")

        registry_key = definition.execution_config.get("registry_key")
        if not isinstance(registry_key, str) or not registry_key.strip():
            raise ToolDefinitionError("Python Function 必须配置 registry_key")

        function = self._registry.get(registry_key)
        if function is None:
            raise FunctionNotRegisteredError(f"Function 未注册: {registry_key}")

        args_model = self._schema_factory.create(
            f"{definition.name.title().replace('_', '')}Input",
            definition.input_schema,
        )
        guard = ToolInvocationGuard(definition)

        async def invoke_function(**arguments: Any) -> Any:
            async def execute() -> Any:
                if inspect.iscoroutinefunction(function):
                    return await function(**arguments)
                result = await asyncio.to_thread(function, **arguments)
                if inspect.isawaitable(result):
                    return await result
                return result

            return await guard.invoke(execute)

        return SchemaValidatedStructuredTool.from_function(
            coroutine=invoke_function,
            name=definition.name,
            description=definition.description,
            args_schema=args_model,
            explicit_default_fields=explicit_default_fields(definition.input_schema),
        )
