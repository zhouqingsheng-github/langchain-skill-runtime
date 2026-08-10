"""Secret reference resolution extension point."""

from typing import Protocol

from langchain_skill_runtime.models.context import CompileContext


class SecretProvider(Protocol):
    """Resolve a named Secret without storing it in ToolDefinition."""

    async def resolve(self, reference: str, context: CompileContext) -> str: ...
