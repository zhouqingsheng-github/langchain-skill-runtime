"""Skill repository protocol."""

from typing import Protocol

from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.models.skill import SkillDefinition


class SkillRepository(Protocol):
    """Loads authorized Skill definitions from business storage."""

    async def get_skill(
        self,
        skill_id: str,
        context: CompileContext,
    ) -> SkillDefinition | None: ...
