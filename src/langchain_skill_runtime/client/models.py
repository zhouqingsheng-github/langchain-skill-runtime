"""Messages exchanged with a client Tool runner."""

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat


class ClientToolStatus(StrEnum):
    """Terminal state reported by the client."""

    SUCCESS = "success"
    ERROR = "error"


class ClientToolRequest(BaseModel):
    """A versioned client Tool invocation."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    tool_id: str
    tool_version: str
    arguments: dict[str, Any]
    timeout_seconds: PositiveFloat


class ClientToolResult(BaseModel):
    """A structured client Tool result correlated by call_id."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    session_id: str
    tool_id: str
    tool_version: str
    status: ClientToolStatus
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: NonNegativeInt | None = None
