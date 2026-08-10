"""Compile one authorized Skill into a prompt and concrete Tools."""

import hashlib
import json
import re
from collections.abc import Sequence

from langchain_core.tools import BaseTool

from langchain_skill_runtime.adapters.factory import ToolFactory
from langchain_skill_runtime.diagnostics import Diagnostic, DiagnosticLevel
from langchain_skill_runtime.errors import (
    SkillCompileError,
    SkillDisabledError,
    SkillNotFoundError,
)
from langchain_skill_runtime.models.bundle import SkillBundle
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.skill import ParsedSkill
from langchain_skill_runtime.models.tool import ResolvedToolDefinition
from langchain_skill_runtime.parsing.skill_parser import SkillParser
from langchain_skill_runtime.prompting.prompt_compiler import PromptCompiler
from langchain_skill_runtime.repositories.skill_repository import SkillRepository
from langchain_skill_runtime.repositories.tool_repository import ToolRepository


class SkillRuntime:
    """Coordinate repository reads, parsing, Tool builds and Prompt compilation."""

    POLICY_VERSION = "1"

    def __init__(
        self,
        skill_repository: SkillRepository,
        tool_repository: ToolRepository,
        tool_factory: ToolFactory,
        skill_parser: SkillParser | None = None,
        prompt_compiler: PromptCompiler | None = None,
    ) -> None:
        self._skill_repository = skill_repository
        self._tool_repository = tool_repository
        self._tool_factory = tool_factory
        self._skill_parser = skill_parser or SkillParser()
        self._prompt_compiler = prompt_compiler or PromptCompiler()

    async def compile(
        self,
        skill_id: str,
        context: CompileContext,
    ) -> SkillBundle:
        skill = await self._skill_repository.get_skill(skill_id, context)
        if skill is None:
            raise SkillNotFoundError(f"Skill 不存在: {skill_id}")
        if not skill.enabled:
            raise SkillDisabledError(f"Skill 已停用: {skill_id}")

        parsed = self._skill_parser.parse(skill.content)
        diagnostics = self._governance_diagnostics(
            skill.name, skill.description, parsed
        )
        governed = parsed.model_copy(
            update={"name": skill.name, "description": skill.description}
        )

        resolved_tools = sorted(
            await self._tool_repository.list_tools(skill_id, context),
            key=lambda item: item.sort_order,
        )
        self._validate_exposed_names(resolved_tools)
        diagnostics.extend(self._binding_diagnostics(governed, resolved_tools))

        built_tools: list[BaseTool] = []
        successful_definitions: list[ResolvedToolDefinition] = []
        for definition in resolved_tools:
            if not definition.enabled:
                diagnostics.append(
                    Diagnostic(
                        code="TOOL_DISABLED",
                        message=f"Tool 已停用: {definition.name}",
                        level=DiagnosticLevel.INFO,
                        tool_id=definition.id,
                    )
                )
                continue
            try:
                built = await self._tool_factory.build(definition, context)
            except Exception:  # noqa: BLE001 - sanitize adapter boundary failures
                if definition.required:
                    raise SkillCompileError(
                        f"必需 Tool 构建失败: {definition.name}"
                    ) from None
                diagnostics.append(
                    Diagnostic(
                        code="TOOL_SKIPPED",
                        message=f"非必需 Tool 已跳过: {definition.name}",
                        tool_id=definition.id,
                    )
                )
                continue
            built_tools.append(built)
            successful_definitions.append(definition)

        system_prompt = self._prompt_compiler.compile(
            governed,
            successful_definitions,
        )
        return SkillBundle(
            skill_id=skill.id,
            name=skill.name,
            description=skill.description,
            system_prompt=system_prompt,
            tools=tuple(built_tools),
            diagnostics=tuple(diagnostics),
            fingerprint=self._fingerprint(
                skill.id,
                skill.version,
                successful_definitions,
            ),
        )

    @staticmethod
    def _governance_diagnostics(
        name: str,
        description: str,
        parsed: ParsedSkill,
    ) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        if parsed.name != name:
            diagnostics.append(
                Diagnostic(
                    code="SKILL_NAME_MISMATCH",
                    message="Frontmatter name 与治理主数据不一致，已采用治理主数据",
                )
            )
        if parsed.description != description:
            diagnostics.append(
                Diagnostic(
                    code="SKILL_DESCRIPTION_MISMATCH",
                    message="Frontmatter description 与治理主数据不一致，已采用治理主数据",
                )
            )
        return diagnostics

    @staticmethod
    def _binding_diagnostics(
        parsed: ParsedSkill,
        tools: Sequence[ResolvedToolDefinition],
    ) -> list[Diagnostic]:
        allowed = set(parsed.allowed_tools)
        bound = {item.name for item in tools}
        diagnostics = [
            Diagnostic(
                code="ALLOWED_TOOL_NOT_BOUND",
                message=f"SKILL.md 声明但未绑定 Tool: {name}",
            )
            for name in sorted(allowed - bound)
        ]
        diagnostics.extend(
            Diagnostic(
                code="BOUND_TOOL_NOT_DECLARED",
                message=f"已绑定但 SKILL.md 未声明 Tool: {name}",
                level=DiagnosticLevel.INFO,
            )
            for name in sorted(bound - allowed)
        )
        return diagnostics

    @classmethod
    def _fingerprint(
        cls,
        skill_id: str,
        skill_version: str,
        tools: Sequence[ResolvedToolDefinition],
    ) -> str:
        payload = {
            "policy_version": cls.POLICY_VERSION,
            "skill_id": skill_id,
            "skill_version": skill_version,
            "tools": [
                {
                    "id": item.id,
                    "version": item.version,
                    "name": item.name,
                    "type": item.tool_type.value,
                }
                for item in tools
            ],
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_exposed_names(tools: Sequence[ResolvedToolDefinition]) -> None:
        seen: set[str] = set()
        for tool in tools:
            if not tool.enabled:
                continue
            if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool.name) is None:
                raise SkillCompileError("Tool 名称不符合模型调用约束")
            if tool.name in seen:
                raise SkillCompileError(f"存在重复 Tool 名称: {tool.name}")
            seen.add(tool.name)
