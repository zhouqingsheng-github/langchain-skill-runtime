"""Uniform timeout and output contract enforcement for Tool invocations."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from pydantic import BaseModel

from langchain_skill_runtime.errors import (
    ToolDefinitionError,
    ToolExecutionError,
    ToolExecutionTimeoutError,
    ToolOutputTooLargeError,
    ToolOutputValidationError,
)
from langchain_skill_runtime.models.tool import ResolvedToolDefinition


class ToolInvocationGuard:
    """Apply execution limits consistently across all Tool types."""

    def __init__(self, definition: ResolvedToolDefinition) -> None:
        self._timeout_seconds = float(definition.timeout_seconds)
        self._max_output_bytes = int(definition.max_output_bytes)
        self._output_validator: Draft202012Validator | None = None
        if definition.output_schema is not None:
            try:
                Draft202012Validator.check_schema(definition.output_schema)
            except SchemaError as exc:
                raise ToolDefinitionError(
                    "Tool output_schema 不是合法 JSON Schema"
                ) from exc
            self._output_validator = Draft202012Validator(definition.output_schema)

    async def invoke(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        enforce_timeout: bool = True,
        passthrough_errors: tuple[type[Exception], ...] = (),
    ) -> Any:
        try:
            if enforce_timeout:
                async with asyncio.timeout(self._timeout_seconds):
                    result = await operation()
            else:
                result = await operation()
        except TimeoutError:
            raise ToolExecutionTimeoutError("Tool 执行超时") from None
        except Exception as exc:
            if isinstance(exc, passthrough_errors):
                raise
            raise ToolExecutionError("Tool 执行失败") from None

        if self._output_validator is not None:
            errors = list(self._output_validator.iter_errors(result))
            if errors:
                raise ToolOutputValidationError("Tool 输出不符合 output_schema")

        encoded = json.dumps(
            result,
            ensure_ascii=False,
            default=self._json_default,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > self._max_output_bytes:
            raise ToolOutputTooLargeError("Tool 输出超过大小限制")
        return result

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        return repr(value)
