"""Non-fatal compile diagnostics."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DiagnosticLevel(StrEnum):
    """Severity for a compile diagnostic."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Diagnostic(BaseModel):
    """A safe diagnostic that can be returned to callers."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    level: DiagnosticLevel = DiagnosticLevel.WARNING
    tool_id: str | None = None
