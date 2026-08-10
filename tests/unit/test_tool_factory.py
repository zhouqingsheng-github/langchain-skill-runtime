from typing import Any

import pytest
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from langchain_skill_runtime.adapters.factory import ToolFactory
from langchain_skill_runtime.errors import (
    DuplicateToolAdapterError,
    ToolAdapterNotFoundError,
)
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType


class ValueInput(BaseModel):
    value: str = ""


class FakeAdapter:
    tool_type = ToolType.PYTHON_FUNCTION

    async def build(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> BaseTool:
        del context

        async def invoke(value: str = "") -> Any:
            return {"tool": definition.name, "value": value}

        return StructuredTool.from_function(
            coroutine=invoke,
            name=definition.name,
            description=definition.description,
            args_schema=ValueInput,
        )


def definition(
    tool_type: ToolType = ToolType.PYTHON_FUNCTION,
) -> ResolvedToolDefinition:
    return ResolvedToolDefinition(
        id="tool-1",
        name="test_tool",
        description="测试工具",
        tool_type=tool_type,
        input_schema={"type": "object", "properties": {}},
        version="1.0.0",
    )


@pytest.mark.asyncio
async def test_factory_dispatches_matching_adapter() -> None:
    factory = ToolFactory([FakeAdapter()])

    built = await factory.build(definition(), CompileContext())

    assert await built.ainvoke({"value": "ok"}) == {
        "tool": "test_tool",
        "value": "ok",
    }


def test_factory_rejects_duplicate_adapter_type() -> None:
    with pytest.raises(DuplicateToolAdapterError):
        ToolFactory([FakeAdapter(), FakeAdapter()])


@pytest.mark.asyncio
async def test_factory_rejects_unregistered_tool_type() -> None:
    with pytest.raises(ToolAdapterNotFoundError, match="MCP"):
        await ToolFactory().build(definition(ToolType.MCP), CompileContext())
