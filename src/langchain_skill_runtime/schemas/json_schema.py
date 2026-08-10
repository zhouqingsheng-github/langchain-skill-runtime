"""Convert the supported JSON Schema subset to Pydantic models."""

from typing import Any, Literal

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, create_model

from langchain_skill_runtime.errors import ToolDefinitionError


class JsonSchemaModelFactory:
    """Build strict Pydantic input models from JSON Schema objects."""

    def create(self, name: str, schema: dict[str, Any]) -> type[BaseModel]:
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ToolDefinitionError("Tool input_schema 不是合法 JSON Schema") from exc

        if schema.get("type") != "object":
            raise ToolDefinitionError("Tool input_schema 根节点必须是 object")
        return self._object_model(name, schema)

    def _object_model(self, name: str, schema: dict[str, Any]) -> type[BaseModel]:
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ToolDefinitionError("JSON Schema properties 必须是对象")

        raw_required = schema.get("required", [])
        if not isinstance(raw_required, list) or not all(
            isinstance(item, str) for item in raw_required
        ):
            raise ToolDefinitionError("JSON Schema required 必须是字符串列表")
        required = set(raw_required)

        fields: dict[str, tuple[Any, Any]] = {}
        for field_name, field_schema in properties.items():
            if not isinstance(field_name, str) or not isinstance(field_schema, dict):
                raise ToolDefinitionError("JSON Schema 字段定义非法")
            field_type = self._python_type(
                field_schema,
                self._nested_name(name, field_name),
            )
            if field_name in required:
                default: Any = ...
            else:
                default = field_schema.get("default", None)
                if "default" not in field_schema:
                    field_type = field_type | None
            fields[field_name] = (field_type, default)

        extra_behavior: Literal["forbid", "allow"] = (
            "forbid" if schema.get("additionalProperties") is False else "allow"
        )
        return create_model(  # type: ignore[call-overload]
            name,
            __config__=ConfigDict(extra=extra_behavior),
            **fields,
        )

    def _python_type(self, schema: dict[str, Any], nested_name: str) -> Any:
        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            return Literal.__getitem__(tuple(enum_values))

        schema_type = schema.get("type")
        if schema_type == "string":
            return str
        if schema_type == "integer":
            return int
        if schema_type == "number":
            return float
        if schema_type == "boolean":
            return bool
        if schema_type == "object":
            return self._object_model(nested_name, schema)
        if schema_type == "array":
            item_schema = schema.get("items", {})
            if not isinstance(item_schema, dict) or not item_schema:
                raise ToolDefinitionError("JSON Schema array 必须声明 items")
            item_type = self._python_type(item_schema, f"{nested_name}Item")
            return list[item_type]  # type: ignore[valid-type,misc]
        raise ToolDefinitionError(f"不支持的 JSON Schema 类型: {schema_type!r}")

    @staticmethod
    def _nested_name(parent: str, field_name: str) -> str:
        parts = [part.capitalize() for part in field_name.replace("-", "_").split("_")]
        return f"{parent}{''.join(parts)}"
