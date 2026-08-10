"""Server script proxy Tool adapter."""

from typing import Any

from langchain_core.tools import BaseTool

from langchain_skill_runtime.adapters.invocation import ToolInvocationGuard
from langchain_skill_runtime.adapters.structured import (
    SchemaValidatedStructuredTool,
    explicit_default_fields,
)
from langchain_skill_runtime.errors import ToolDefinitionError
from langchain_skill_runtime.executors.server_script_executor import (
    ServerScriptExecutor,
)
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType
from langchain_skill_runtime.schemas.json_schema import JsonSchemaModelFactory


class ServerScriptAdapter:
    """Delegate a Tool invocation to a business-owned script executor."""

    tool_type = ToolType.SERVER_SCRIPT

    def __init__(
        self,
        executor: ServerScriptExecutor,
        schema_factory: JsonSchemaModelFactory | None = None,
    ) -> None:
        self._executor = executor
        self._schema_factory = schema_factory or JsonSchemaModelFactory()

    async def build(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> BaseTool:
        if definition.tool_type is not self.tool_type:
            raise ToolDefinitionError("ServerScriptAdapter 收到错误的 Tool 类型")

        artifact_id = definition.execution_config.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ToolDefinitionError("Server Script 必须配置 artifact_id")

        args_model = self._schema_factory.create(
            f"{definition.name.title().replace('_', '')}Input",
            definition.input_schema,
        )
        guard = ToolInvocationGuard(definition)

        async def execute_script(**arguments: Any) -> Any:
            async def execute() -> Any:
                return await self._executor.execute(
                    artifact_id=artifact_id,
                    arguments=arguments,
                    context=context,
                    timeout_seconds=float(definition.timeout_seconds),
                )

            return await guard.invoke(execute)

        return SchemaValidatedStructuredTool.from_function(
            coroutine=execute_script,
            name=definition.name,
            description=definition.description,
            args_schema=args_model,
            explicit_default_fields=explicit_default_fields(definition.input_schema),
        )
