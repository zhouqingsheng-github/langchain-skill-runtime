"""Tool definitions and Skill binding models."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class ToolType(StrEnum):
    """Supported execution channels."""

    PYTHON_FUNCTION = "PYTHON_FUNCTION"
    SERVER_SCRIPT = "SERVER_SCRIPT"
    CLIENT_JAVASCRIPT = "CLIENT_JAVASCRIPT"
    MCP = "MCP"


class ToolDefinition(BaseModel):
    """Reusable Tool definition before Skill-specific binding."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str
    tool_type: ToolType
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    execution_config: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: PositiveFloat = 30.0
    version: str
    enabled: bool = True


class SkillToolBinding(BaseModel):
    """Skill-specific Tool settings."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    tool_id: str
    required: bool = True
    enabled: bool = True
    sort_order: int = 0
    tool_alias: str | None = None
    config_override: dict[str, Any] = Field(default_factory=dict)


class ResolvedToolDefinition(BaseModel):
    """A repository-resolved Tool ready for adapter compilation."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str
    tool_type: ToolType
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    execution_config: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: PositiveFloat = 30.0
    version: str
    required: bool = True
    enabled: bool = True
    sort_order: int = 0

    @classmethod
    def resolve(
        cls,
        definition: ToolDefinition,
        binding: SkillToolBinding,
    ) -> "ResolvedToolDefinition":
        """Merge a reusable definition with its Skill binding."""

        return cls(
            id=definition.id,
            name=binding.tool_alias or definition.name,
            description=definition.description,
            tool_type=definition.tool_type,
            input_schema=definition.input_schema,
            output_schema=definition.output_schema,
            execution_config={
                **definition.execution_config,
                **binding.config_override,
            },
            timeout_seconds=definition.timeout_seconds,
            version=definition.version,
            required=binding.required,
            enabled=definition.enabled and binding.enabled,
            sort_order=binding.sort_order,
        )
