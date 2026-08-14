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
    ToolDefinitionError,
    ToolUnavailableError,
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


class ControlledFailureAdapter:
    tool_type = ToolType.PYTHON_FUNCTION

    def __init__(self, error: ToolDefinitionError | ToolUnavailableError) -> None:
        self._error = error

    async def build(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> BaseTool:
        del definition, context
        raise self._error


class ExpandingMcpAdapter:
    tool_type = ToolType.MCP

    async def build(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> BaseTool:
        del definition, context
        raise AssertionError("集合型 MCP 不应走单工具 build")

    async def build_many(
        self,
        definition: ResolvedToolDefinition,
        context: CompileContext,
    ) -> tuple[BaseTool, ...]:
        del context

        async def geo(address: str) -> str:
            return f"location:{address}"

        async def route(origin: str, destination: str) -> str:
            return f"route:{origin}:{destination}"

        return (
            StructuredTool.from_function(
                coroutine=geo,
                name="maps_geo",
                description="地址解析",
            ),
            StructuredTool.from_function(
                coroutine=route,
                name="maps_direction_driving",
                description="驾车路线规划",
            ),
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


@pytest.mark.parametrize(
    "controlled_error",
    [
        ToolUnavailableError("MCP Server 地址解析失败"),
        ToolDefinitionError("内联 MCP Server 必须配置宿主 McpUrlPolicy"),
    ],
)
@pytest.mark.asyncio
async def test_runtime_reports_controlled_tool_failure_reason(
    controlled_error: ToolDefinitionError | ToolUnavailableError,
) -> None:
    controlled_runtime = SkillRuntime(
        skill_repository=MemorySkillRepository(skill()),
        tool_repository=MemoryToolRepository(
            [tool_definition("successful_tool", fail=True)]
        ),
        tool_factory=ToolFactory([ControlledFailureAdapter(controlled_error)]),
    )

    with pytest.raises(SkillCompileError) as captured:
        await controlled_runtime.compile("skill-1", CompileContext())

    assert str(captured.value) == (
        f"必需 Tool 构建失败: successful_tool；原因：{controlled_error}"
    )


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
async def test_runtime_expands_one_mcp_definition_into_discovered_tools() -> None:
    mcp_definition = ResolvedToolDefinition(
        id="amap-maps",
        name="amap_maps",
        description="高德地图 MCP 工具集合",
        tool_type=ToolType.MCP,
        input_schema={"type": "object", "properties": {}},
        execution_config={"server_name": "amap", "server": {}},
        version="1.0.0",
    )
    object_runtime = SkillRuntime(
        tool_factory=ToolFactory([ExpandingMcpAdapter()]),
    )

    bundle = await object_runtime.compile_objects(
        skill=skill(),
        tools=[mcp_definition],
        context=CompileContext(),
    )

    assert [item.name for item in bundle.tools] == [
        "maps_geo",
        "maps_direction_driving",
    ]
    assert "maps_geo: 地址解析" in bundle.system_prompt
    assert "maps_direction_driving: 驾车路线规划" in bundle.system_prompt


@pytest.mark.asyncio
async def test_runtime_keeps_collection_semantics_for_one_same_named_tool() -> None:
    class OneToolMcpAdapter:
        tool_type = ToolType.MCP

        async def build(
            self,
            definition: ResolvedToolDefinition,
            context: CompileContext,
        ) -> BaseTool:
            del definition, context
            raise AssertionError("集合型 MCP 不应走单工具 build")

        async def build_many(
            self,
            definition: ResolvedToolDefinition,
            context: CompileContext,
        ) -> tuple[BaseTool, ...]:
            del definition, context

            async def invoke(value: str) -> str:
                return value

            return (
                StructuredTool.from_function(
                    coroutine=invoke,
                    name="amap_maps",
                    description="远端真实工具说明",
                ),
            )

    definition = ResolvedToolDefinition(
        id="amap-maps",
        name="amap_maps",
        description="集合对象说明",
        tool_type=ToolType.MCP,
        input_schema={"type": "object", "properties": {}},
        execution_config={"server_name": "amap", "server": {}},
        version="1.0.0",
    )
    object_runtime = SkillRuntime(
        tool_factory=ToolFactory([OneToolMcpAdapter()]),
    )

    bundle = await object_runtime.compile_objects(
        skill=skill(),
        tools=[definition],
        context=CompileContext(),
    )

    assert "amap_maps: 远端真实工具说明" in bundle.system_prompt
    assert "集合对象说明" not in bundle.system_prompt


@pytest.mark.asyncio
async def test_runtime_requires_repositories_only_for_legacy_compile() -> None:
    repository_free_runtime = SkillRuntime(tool_factory=ToolFactory())

    with pytest.raises(SkillRuntimeConfigurationError):
        await repository_free_runtime.compile("skill-1", CompileContext())
