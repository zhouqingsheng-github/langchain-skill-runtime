from collections.abc import Mapping
from typing import Any

import pytest

from langchain_skill_runtime.adapters.server_script import ServerScriptAdapter
from langchain_skill_runtime.errors import ToolDefinitionError
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType


class RecordingScriptExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any], CompileContext, float]] = []

    async def execute(
        self,
        artifact_id: str,
        arguments: Mapping[str, Any],
        context: CompileContext,
        timeout_seconds: float,
    ) -> Any:
        self.calls.append((artifact_id, arguments, context, timeout_seconds))
        return {"status": "generated", "title": arguments["title"]}


def script_definition(artifact_id: str = "report.test.v1") -> ResolvedToolDefinition:
    return ResolvedToolDefinition(
        id="script-report",
        name="generate_report",
        description="生成测试报告",
        tool_type=ToolType.SERVER_SCRIPT,
        input_schema={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
        execution_config={"artifact_id": artifact_id},
        timeout_seconds=12,
        version="1.0.0",
    )


@pytest.mark.asyncio
async def test_script_adapter_delegates_to_executor() -> None:
    executor = RecordingScriptExecutor()
    context = CompileContext(tenant_id="tenant-1", user_id="user-1")
    tool = await ServerScriptAdapter(executor).build(script_definition(), context)

    result = await tool.ainvoke({"title": "营业日报"})

    assert result == {"status": "generated", "title": "营业日报"}
    assert executor.calls == [
        ("report.test.v1", {"title": "营业日报"}, context, 12.0)
    ]


@pytest.mark.asyncio
async def test_script_adapter_rejects_blank_artifact_id() -> None:
    with pytest.raises(ToolDefinitionError, match="artifact_id"):
        await ServerScriptAdapter(RecordingScriptExecutor()).build(
            script_definition(""), CompileContext()
        )
