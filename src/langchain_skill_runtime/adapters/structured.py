"""StructuredTool variant that preserves omitted JSON Schema fields."""

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class SchemaValidatedStructuredTool(StructuredTool):
    """Validate input while excluding optional fields the caller omitted."""

    explicit_default_fields: frozenset[str] = Field(
        default_factory=frozenset,
        exclude=True,
    )

    def _parse_input(
        self,
        tool_input: str | dict[str, Any],
        tool_call_id: str | None,
    ) -> str | dict[str, Any]:
        args_schema = self.args_schema
        if (
            isinstance(tool_input, dict)
            and isinstance(args_schema, type)
            and issubclass(args_schema, BaseModel)
        ):
            validated = args_schema.model_validate(tool_input)
            result = {
                key: getattr(validated, key)
                for key in tool_input
                if key in args_schema.model_fields
            }
            for key in self.explicit_default_fields:
                if key not in result:
                    result[key] = getattr(validated, key)
            return result
        return super()._parse_input(tool_input, tool_call_id)


def explicit_default_fields(input_schema: dict[str, Any]) -> frozenset[str]:
    """Return property names that explicitly declare a JSON Schema default."""

    properties = input_schema.get("properties", {})
    if not isinstance(properties, dict):
        return frozenset()
    return frozenset(
        name
        for name, schema in properties.items()
        if isinstance(name, str) and isinstance(schema, dict) and "default" in schema
    )
