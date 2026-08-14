"""Compile and invoke a PYTHON_FUNCTION Tool from SKILL.md."""

import asyncio
from pathlib import Path

from langchain_skill_runtime import SkillRuntime
from langchain_skill_runtime.adapters import PythonFunctionAdapter, ToolFactory
from langchain_skill_runtime.executors import InMemoryFunctionRegistry

SKILL_PATH = Path(__file__).with_name("SKILL.md")


async def add_numbers(a: int, b: int) -> int:
    return a + b


async def main() -> None:
    registry = InMemoryFunctionRegistry()
    registry.register("math.add", add_numbers)
    runtime = SkillRuntime(tool_factory=ToolFactory([PythonFunctionAdapter(registry)]))
    bundle = await runtime.compile_file(SKILL_PATH)
    result = await bundle.tools[0].ainvoke({"a": 18, "b": 24})
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
