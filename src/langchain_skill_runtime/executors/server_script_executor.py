"""Protocol for a business-owned, sandboxed script executor."""

from collections.abc import Mapping
from typing import Any, Protocol

from langchain_skill_runtime.models.context import CompileContext


class ServerScriptExecutor(Protocol):
    """Executes a versioned script artifact outside the core library."""

    async def execute(
        self,
        artifact_id: str,
        arguments: Mapping[str, Any],
        context: CompileContext,
        timeout_seconds: float,
    ) -> Any: ...
