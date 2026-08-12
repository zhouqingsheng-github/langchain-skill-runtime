import asyncio

import pytest

from langchain_skill_runtime.adapters.function import PythonFunctionAdapter
from langchain_skill_runtime.errors import (
    FunctionNotRegisteredError,
    ToolExecutionError,
    ToolExecutionTimeoutError,
    ToolOutputTooLargeError,
    ToolOutputValidationError,
)
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


@pytest.mark.asyncio
async def test_function_adapter_does_not_inject_omitted_optional_field() -> None:
    registry = InMemoryFunctionRegistry()

    def add_with_default(a: int, b: int = 7) -> int:
        return a + b

    registry.register("math.add", add_with_default)
    definition = function_definition().model_copy(
        update={
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a"],
                "additionalProperties": False,
            }
        }
    )
    tool = await PythonFunctionAdapter(registry).build(definition, CompileContext())

    assert await tool.ainvoke({"a": 3}) == 10


@pytest.mark.asyncio
async def test_function_adapter_enforces_timeout() -> None:
    registry = InMemoryFunctionRegistry()

    async def slow_add(a: int, b: int) -> int:
        await asyncio.sleep(0.05)
        return a + b

    registry.register("math.add", slow_add)
    definition = function_definition().model_copy(update={"timeout_seconds": 0.01})
    tool = await PythonFunctionAdapter(registry).build(definition, CompileContext())

    with pytest.raises(ToolExecutionTimeoutError):
        await tool.ainvoke({"a": 1, "b": 2})


@pytest.mark.asyncio
async def test_function_adapter_validates_output_schema() -> None:
    registry = InMemoryFunctionRegistry()
    registry.register("math.add", lambda a, b: "not-an-integer")
    definition = function_definition().model_copy(
        update={"output_schema": {"type": "integer"}}
    )
    tool = await PythonFunctionAdapter(registry).build(definition, CompileContext())

    with pytest.raises(ToolOutputValidationError):
        await tool.ainvoke({"a": 1, "b": 2})


@pytest.mark.asyncio
async def test_function_adapter_limits_output_size() -> None:
    registry = InMemoryFunctionRegistry()
    registry.register("math.add", lambda a, b: "x" * 100)
    definition = function_definition().model_copy(update={"max_output_bytes": 16})
    tool = await PythonFunctionAdapter(registry).build(definition, CompileContext())

    with pytest.raises(ToolOutputTooLargeError):
        await tool.ainvoke({"a": 1, "b": 2})


class AsyncCallable:
    async def __call__(self, a: int, b: int) -> int:
        return a + b


@pytest.mark.asyncio
async def test_function_adapter_awaits_async_callable_object() -> None:
    registry = InMemoryFunctionRegistry()
    registry.register("math.add", AsyncCallable())
    tool = await PythonFunctionAdapter(registry).build(
        function_definition(), CompileContext()
    )

    assert await tool.ainvoke({"a": 4, "b": 6}) == 10


@pytest.mark.asyncio
async def test_function_adapter_preserves_allowed_additional_properties() -> None:
    registry = InMemoryFunctionRegistry()

    def collect_arguments(**arguments: object) -> dict[str, object]:
        return arguments

    registry.register("math.add", collect_arguments)
    definition = function_definition().model_copy(
        update={
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
                "additionalProperties": True,
            }
        }
    )
    tool = await PythonFunctionAdapter(registry).build(definition, CompileContext())

    assert await tool.ainvoke({"a": 1, "extra": "kept"}) == {
        "a": 1,
        "extra": "kept",
    }


@pytest.mark.asyncio
async def test_function_adapter_sanitizes_execution_failure() -> None:
    registry = InMemoryFunctionRegistry()

    def fail(a: int, b: int) -> int:
        del a, b
        raise RuntimeError("secret-token /internal/function/path")

    registry.register("math.add", fail)
    tool = await PythonFunctionAdapter(registry).build(
        function_definition(), CompileContext()
    )

    with pytest.raises(ToolExecutionError) as captured:
        await tool.ainvoke({"a": 1, "b": 2})

    assert "secret-token" not in str(captured.value)
    assert "/internal/function/path" not in str(captured.value)
