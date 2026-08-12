"""Skill source and parsed models."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from langchain_skill_runtime.models.tool import ResolvedToolDefinition


class SkillDefinition(BaseModel):
    """Raw Skill data supplied by a repository."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str
    content: str
    version: str
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedSkill(BaseModel):
    """Structured SKILL.md content."""

    model_config = ConfigDict(frozen=True)

    id: str | None = None
    name: str
    description: str
    version: str = "0.0.0"
    instructions: str
    compatibility: str | None = None
    allowed_tools: tuple[str, ...] = ()
    tool_declarations: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillDocument(BaseModel):
    """One normalized Skill ready for source-independent compilation."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str
    version: str
    instructions: str
    allowed_tools: tuple[str, ...] = ()
    tools: tuple[ResolvedToolDefinition, ...] = ()
    source_root: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
