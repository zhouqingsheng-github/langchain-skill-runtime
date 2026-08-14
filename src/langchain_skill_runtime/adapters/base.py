"""Common Tool adapter contract."""

from typing import Protocol, runtime_checkable

from langchain_core.tools import BaseTool

from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType


class ToolAdapter(Protocol):
    """Converts one execution type into a LangChain BaseTool."""

    @property
    def tool_type(self) -> ToolType: ...

    async def build(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> BaseTool: ...


@runtime_checkable
class ToolCollectionAdapter(Protocol):
    """Optional adapter capability that expands one definition into many tools."""

    async def build_many(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> tuple[BaseTool, ...]: ...
