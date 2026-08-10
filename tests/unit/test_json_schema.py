import pytest
from pydantic import ValidationError

from langchain_skill_runtime.errors import ToolDefinitionError
from langchain_skill_runtime.schemas.json_schema import JsonSchemaModelFactory


def test_schema_factory_builds_required_optional_and_array_fields() -> None:
    model = JsonSchemaModelFactory().create(
        "AddNumbersInput",
        {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "number", "default": 1},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            "required": ["a"],
            "additionalProperties": False,
        },
    )

    assert model(a=2).model_dump() == {"a": 2, "b": 1, "tags": []}
    with pytest.raises(ValidationError):
        model(tags=[])
    with pytest.raises(ValidationError):
        model(a=2, unknown=True)


def test_schema_factory_supports_nested_objects_and_enum() -> None:
    model = JsonSchemaModelFactory().create(
        "ExportInput",
        {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["xlsx", "csv"]},
                "filters": {
                    "type": "object",
                    "properties": {"hotel_id": {"type": "integer"}},
                    "required": ["hotel_id"],
                    "additionalProperties": False,
                },
            },
            "required": ["format", "filters"],
        },
    )

    value = model(format="xlsx", filters={"hotel_id": 1001})
    assert value.model_dump() == {
        "format": "xlsx",
        "filters": {"hotel_id": 1001},
    }
    with pytest.raises(ValidationError):
        model(format="pdf", filters={"hotel_id": 1001})


def test_schema_factory_rejects_non_object_root() -> None:
    with pytest.raises(ToolDefinitionError, match="object"):
        JsonSchemaModelFactory().create("InvalidInput", {"type": "string"})
