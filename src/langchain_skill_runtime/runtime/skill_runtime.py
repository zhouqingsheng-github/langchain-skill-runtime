"""Compile one authorized Skill into a prompt and concrete Tools."""

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

from langchain_core.tools import BaseTool

from langchain_skill_runtime.adapters.factory import ToolFactory
from langchain_skill_runtime.diagnostics import Diagnostic, DiagnosticLevel
from langchain_skill_runtime.errors import (
    SkillCompileError,
    SkillDisabledError,
    SkillNotFoundError,
    SkillRuntimeConfigurationError,
)
from langchain_skill_runtime.models.bundle import SkillBundle
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.skill import (
    ParsedSkill,
    SkillDefinition,
    SkillDocument,
)
from langchain_skill_runtime.models.tool import ResolvedToolDefinition
from langchain_skill_runtime.parsing.skill_file_loader import SkillFileLoader
from langchain_skill_runtime.parsing.skill_parser import SkillParser
from langchain_skill_runtime.prompting.prompt_compiler import PromptCompiler
from langchain_skill_runtime.repositories.skill_repository import SkillRepository
from langchain_skill_runtime.repositories.tool_repository import ToolRepository


class SkillRuntime:
    """Coordinate repository reads, parsing, Tool builds and Prompt compilation."""

    POLICY_VERSION = "1"

    def __init__(
        self,
        skill_repository: SkillRepository | None = None,
        tool_repository: ToolRepository | None = None,
        tool_factory: ToolFactory | None = None,
        skill_parser: SkillParser | None = None,
        prompt_compiler: PromptCompiler | None = None,
        skill_file_loader: SkillFileLoader | None = None,
    ) -> None:
        if tool_factory is None:
            raise SkillRuntimeConfigurationError("SkillRuntime 缺少 ToolFactory")
        self._skill_repository = skill_repository
        self._tool_repository = tool_repository
        self._tool_factory = tool_factory
        self._skill_parser = skill_parser or SkillParser()
        self._prompt_compiler = prompt_compiler or PromptCompiler()
        self._skill_file_loader = skill_file_loader or SkillFileLoader(
            self._skill_parser
        )

    async def compile(
        self,
        skill_id: str,
        context: CompileContext,
    ) -> SkillBundle:
        if self._skill_repository is None or self._tool_repository is None:
            raise SkillRuntimeConfigurationError("Repository 编译模式缺少 Repository")
        skill = await self._skill_repository.get_skill(skill_id, context)
        if skill is None:
            raise SkillNotFoundError(f"Skill 不存在: {skill_id}")
        if not skill.enabled:
            raise SkillDisabledError(f"Skill 已停用: {skill.id}")
        tools = await self._tool_repository.list_tools(skill_id, context)
        return await self.compile_objects(skill, tools, context)

    async def compile_file(
        self,
        path: str | Path,
        context: CompileContext | None = None,
    ) -> SkillBundle:
        """Compile one standalone SKILL.md without any Repository."""

        document = self._skill_file_loader.load(path)
        return await self._compile_document(document, context or CompileContext())

    async def compile_objects(
        self,
        skill: SkillDefinition,
        tools: Sequence[ResolvedToolDefinition],
        context: CompileContext | None = None,
    ) -> SkillBundle:
        """Compile Skill and Tool objects prepared by any business storage."""

        if not skill.enabled:
            raise SkillDisabledError(f"Skill 已停用: {skill.id}")

        parsed = self._skill_parser.parse(skill.content)
        diagnostics = self._governance_diagnostics(
            skill.name, skill.description, parsed
        )
        governed = parsed.model_copy(
            update={"name": skill.name, "description": skill.description}
        )
        document = SkillDocument(
            id=skill.id,
            name=governed.name,
            description=governed.description,
            version=skill.version,
            instructions=governed.instructions,
            allowed_tools=governed.allowed_tools,
            tools=tuple(tools),
            metadata={**governed.metadata, **skill.metadata},
        )
        return await self._compile_document(
            document,
            context or CompileContext(),
            diagnostics,
        )

    async def _compile_document(
        self,
        document: SkillDocument,
        context: CompileContext,
        initial_diagnostics: Sequence[Diagnostic] = (),
    ) -> SkillBundle:
        governed = ParsedSkill(
            id=document.id,
            name=document.name,
            description=document.description,
            version=document.version,
            instructions=document.instructions,
            allowed_tools=document.allowed_tools,
            metadata=document.metadata,
        )
        resolved_tools = sorted(
            document.tools,
            key=lambda item: item.sort_order,
        )
        self._validate_exposed_names(resolved_tools)
        diagnostics = list(initial_diagnostics)
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
            skill_id=document.id,
            name=document.name,
            description=document.description,
            system_prompt=system_prompt,
            tools=tuple(built_tools),
            diagnostics=tuple(diagnostics),
            fingerprint=self._fingerprint(
                document.id,
                document.version,
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
