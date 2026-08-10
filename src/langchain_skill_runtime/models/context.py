"""Compile-time identity, authorization and client capability context."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClientCapability(BaseModel):
    """One Tool capability declared by an online client."""

    model_config = ConfigDict(frozen=True)

    tool_id: str
    version: str
    available: bool = True


class CompileContext(BaseModel):
    """Runtime data used for filtering and binding Tools."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    permissions: frozenset[str] = frozenset()
    client_capabilities: tuple[ClientCapability, ...] = ()
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)
