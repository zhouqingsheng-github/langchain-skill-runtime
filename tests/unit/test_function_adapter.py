import pytest

from langchain_skill_runtime.adapters.function import PythonFunctionAdapter
from langchain_skill_runtime.errors import FunctionNotRegisteredError
from langchain_skill_runtime.executors.function_registry import (
    InMemoryFunctionRegistry,
)
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType


def function_definition(registry_key: str = "math.add") -> ResolvedToolDefinition:
    return ResolvedToolDefinition(
        id="function-add",
        name="add_numbers",
        description="计算两个数字之和",
        tool_type=ToolType.PYTHON_FUNCTION,
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        execution_config={"registry_key": registry_key},
        version="1.0.0",
    )


@pytest.mark.asyncio
async def test_function_adapter_invokes_registered_async_function() -> None:
    registry = InMemoryFunctionRegistry()

    async def add_numbers(a: int, b: int) -> int:
        return a + b

    registry.register("math.add", add_numbers)
    tool = await PythonFunctionAdapter(registry).build(
        function_definition(), CompileContext()
    )

    assert await tool.ainvoke({"a": 2, "b": 3}) == 5


@pytest.mark.asyncio
async def test_function_adapter_invokes_registered_sync_function() -> None:
    registry = InMemoryFunctionRegistry()
    registry.register("math.add", lambda a, b: a + b)

    tool = await PythonFunctionAdapter(registry).build(
        function_definition(), CompileContext()
    )

    assert await tool.ainvoke({"a": 10, "b": 4}) == 14


@pytest.mark.asyncio
async def test_function_adapter_rejects_unknown_registry_key() -> None:
    with pytest.raises(FunctionNotRegisteredError, match="math.missing"):
        await PythonFunctionAdapter(InMemoryFunctionRegistry()).build(
            function_definition("math.missing"), CompileContext()
        )
