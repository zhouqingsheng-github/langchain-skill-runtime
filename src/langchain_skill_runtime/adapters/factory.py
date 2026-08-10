"""Registry and dispatch for Tool adapters."""

from collections.abc import Iterable

from langchain_core.tools import BaseTool

from langchain_skill_runtime.adapters.base import ToolAdapter
from langchain_skill_runtime.errors import (
    DuplicateToolAdapterError,
    ToolAdapterNotFoundError,
)
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType


class ToolFactory:
    """Dispatch resolved definitions to their registered adapter."""

    def __init__(self, adapters: Iterable[ToolAdapter] = ()) -> None:
        self._adapters: dict[ToolType, ToolAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ToolAdapter) -> None:
        if adapter.tool_type in self._adapters:
            raise DuplicateToolAdapterError(
                f"ToolAdapter 已注册: {adapter.tool_type.value}"
            )
        self._adapters[adapter.tool_type] = adapter

    async def build(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> BaseTool:
        adapter = self._adapters.get(definition.tool_type)
        if adapter is None:
            raise ToolAdapterNotFoundError(
                f"未注册 ToolAdapter: {definition.tool_type.value}"
            )
        return await adapter.build(definition, context)
