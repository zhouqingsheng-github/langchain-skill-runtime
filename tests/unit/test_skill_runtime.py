from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from langchain_skill_runtime.adapters.factory import ToolFactory
from langchain_skill_runtime.errors import (
    SkillCompileError,
    SkillDisabledError,
    SkillNotFoundError,
    SkillRuntimeConfigurationError,
    ToolBuildError,
)
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.skill import SkillDefinition
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType
from langchain_skill_runtime.runtime.skill_runtime import SkillRuntime

SKILL_CONTENT = """---
name: frontmatter-name
description: Frontmatter 描述
allowed-tools:
  - successful_tool
  - optional_failed_tool
  - unbound_tool
---

# 测试技能

只允许调用成功装配的工具。
"""


class ValueInput(BaseModel):
    value: str = ""


class MemorySkillRepository:
    def __init__(self, skill: SkillDefinition | None) -> None:
        self.skill = skill

    async def get_skill(
        self,
        skill_id: str,
        context: CompileContext,
    ) -> SkillDefinition | None:
        del skill_id, context
        return self.skill


class MemoryToolRepository:
    def __init__(self, tools: Sequence[ResolvedToolDefinition]) -> None:
        self.tools = tools

    async def list_tools(
        self,
        skill_id: str,
        context: CompileContext,
    ) -> Sequence[ResolvedToolDefinition]:
        del skill_id, context
        return self.tools


class RuntimeTestAdapter:
    tool_type = ToolType.PYTHON_FUNCTION

    async def build(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> BaseTool:
        del context
        if definition.execution_config.get("fail"):
            raise ToolBuildError("internal failure detail")

        async def invoke(value: str = "") -> Any:
            return {"name": definition.name, "value": value}

        return StructuredTool.from_function(
            coroutine=invoke,
            name=definition.name,
            description=definition.description,
            args_schema=ValueInput,
        )


def skill(*, enabled: bool = True, version: str = "1.0.0") -> SkillDefinition:
    return SkillDefinition(
        id="skill-1",
        name="database-name",
        description="数据库治理描述",
        content=SKILL_CONTENT,
        version=version,
        enabled=enabled,
    )


def tool_definition(
    name: str,
    *,
    version: str = "1.0.0",
    required: bool = True,
    fail: bool = False,
    sort_order: int = 0,
) -> ResolvedToolDefinition:
    return ResolvedToolDefinition(
        id=f"id-{name}",
        name=name,
        description=f"{name} 描述",
        tool_type=ToolType.PYTHON_FUNCTION,
        input_schema={"type": "object", "properties": {}},
        execution_config={"fail": fail},
        version=version,
        required=required,
        sort_order=sort_order,
    )


def runtime(
    repository_skill: SkillDefinition | None,
    tools: Sequence[ResolvedToolDefinition] = (),
) -> SkillRuntime:
    return SkillRuntime(
        skill_repository=MemorySkillRepository(repository_skill),
        tool_repository=MemoryToolRepository(tools),
        tool_factory=ToolFactory([RuntimeTestAdapter()]),
    )


@pytest.mark.asyncio
async def test_runtime_rejects_missing_and_disabled_skill() -> None:
    with pytest.raises(SkillNotFoundError):
        await runtime(None).compile("skill-1", CompileContext())
    with pytest.raises(SkillDisabledError):
        await runtime(skill(enabled=False)).compile("skill-1", CompileContext())


@pytest.mark.asyncio
async def test_runtime_stops_when_required_tool_fails() -> None:
    required_failure = tool_definition("successful_tool", fail=True)

    with pytest.raises(SkillCompileError, match="successful_tool") as captured:
        await runtime(skill(), [required_failure]).compile("skill-1", CompileContext())

    assert "internal failure detail" not in str(captured.value)


@pytest.mark.asyncio
async def test_runtime_skips_optional_tool_and_prompts_only_successful_tools() -> None:
    tools = [
        tool_definition(
            "optional_failed_tool", required=False, fail=True, sort_order=2
        ),
        tool_definition("successful_tool", sort_order=1),
    ]

    bundle = await runtime(skill(), tools).compile("skill-1", CompileContext())

    assert [item.name for item in bundle.tools] == ["successful_tool"]
    assert "successful_tool 描述" in bundle.system_prompt
    assert "optional_failed_tool 描述" not in bundle.system_prompt
    assert "unbound_tool" not in bundle.system_prompt
    assert {item.code for item in bundle.diagnostics} >= {
        "SKILL_NAME_MISMATCH",
        "SKILL_DESCRIPTION_MISMATCH",
        "TOOL_SKIPPED",
        "ALLOWED_TOOL_NOT_BOUND",
    }


@pytest.mark.asyncio
async def test_runtime_fingerprint_changes_only_with_compiled_versions() -> None:
    first = await runtime(skill(), [tool_definition("successful_tool")]).compile(
        "skill-1", CompileContext()
    )
    same = await runtime(skill(), [tool_definition("successful_tool")]).compile(
        "skill-1", CompileContext()
    )
    changed = await runtime(
        skill(), [tool_definition("successful_tool", version="2.0.0")]
    ).compile("skill-1", CompileContext())

    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != changed.fingerprint


@pytest.mark.asyncio
async def test_runtime_rejects_duplicate_exposed_tool_names() -> None:
    first = tool_definition("successful_tool", sort_order=1)
    second = tool_definition("successful_tool", sort_order=2).model_copy(
        update={"id": "another-tool-id"}
    )

    with pytest.raises(SkillCompileError, match="重复"):
        await runtime(skill(), [first, second]).compile("skill-1", CompileContext())


@pytest.mark.asyncio
async def test_runtime_compiles_prompt_only_skill_file(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(
        """---
id: market-skill
name: market-skill
description: 市场纯提示词技能
version: 1.0.0
---

# 市场技能

只使用提示词完成任务。
""",
        encoding="utf-8",
    )
    file_runtime = SkillRuntime(tool_factory=ToolFactory())

    bundle = await file_runtime.compile_file(skill_path)

    assert bundle.skill_id == "market-skill"
    assert bundle.tools == ()
    assert "只使用提示词完成任务" in bundle.system_prompt


@pytest.mark.asyncio
async def test_runtime_compiles_skill_and_tool_objects_without_repositories() -> None:
    object_runtime = SkillRuntime(
        tool_factory=ToolFactory([RuntimeTestAdapter()]),
    )

    bundle = await object_runtime.compile_objects(
        skill=skill(),
        tools=[tool_definition("successful_tool")],
        context=CompileContext(),
    )

    assert bundle.skill_id == "skill-1"
    assert [item.name for item in bundle.tools] == ["successful_tool"]
    assert bundle.name == "database-name"


@pytest.mark.asyncio
async def test_runtime_requires_repositories_only_for_legacy_compile() -> None:
    repository_free_runtime = SkillRuntime(tool_factory=ToolFactory())

    with pytest.raises(SkillRuntimeConfigurationError):
        await repository_free_runtime.compile("skill-1", CompileContext())
