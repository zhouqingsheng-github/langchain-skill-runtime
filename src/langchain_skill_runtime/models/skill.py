"""Skill source and parsed models."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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

    name: str
    description: str
    instructions: str
    compatibility: str | None = None
    allowed_tools: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
