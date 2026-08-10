"""Compiled Skill output."""

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict

from langchain_skill_runtime.diagnostics import Diagnostic


class SkillBundle(BaseModel):
    """Prompt and concrete LangChain Tools for one Skill."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    skill_id: str
    name: str
    description: str
    system_prompt: str
    tools: tuple[BaseTool, ...]
    resources: tuple[Any, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    fingerprint: str
