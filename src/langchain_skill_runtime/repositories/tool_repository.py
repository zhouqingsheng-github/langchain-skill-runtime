"""Tool repository protocol."""

from collections.abc import Sequence
from typing import Protocol

from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition


class ToolRepository(Protocol):
    """Loads authorized and Skill-resolved Tool definitions."""

    async def list_tools(
        self,
        skill_id: str,
        context: CompileContext,
    ) -> Sequence[ResolvedToolDefinition]: ...
