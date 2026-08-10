import asyncio
from typing import Any

import pytest

from langchain_skill_runtime.adapters.client_javascript import (
    ClientJavascriptAdapter,
)
from langchain_skill_runtime.client.models import (
    ClientToolRequest,
    ClientToolResult,
    ClientToolStatus,
)
from langchain_skill_runtime.errors import (
    ClientToolTimeoutError,
    ToolExecutionError,
    ToolUnavailableError,
)
from langchain_skill_runtime.models.context import ClientCapability, CompileContext
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType


class RecordingTransport:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.availability_checks: list[tuple[str, str, str]] = []
        self.requests: list[ClientToolRequest] = []

    async def is_available(
        self,
        session_id: str,
        tool_id: str,
        tool_version: str,
    ) -> bool:
        self.availability_checks.append((session_id, tool_id, tool_version))
        return self.available

    async def invoke(self, call: ClientToolRequest) -> ClientToolResult:
        self.requests.append(call)
        return ClientToolResult(
            call_id=call.call_id,
            session_id=call.session_id,
            tool_id=call.tool_id,
            tool_version=call.tool_version,
            status=ClientToolStatus.SUCCESS,
            output={"file_name": "测试文件.xlsx"},
        )


def definition() -> ResolvedToolDefinition:
    return ResolvedToolDefinition(
        id="client-export",
        name="export_client_file",
        description="在客户端导出文件",
        tool_type=ToolType.CLIENT_JAVASCRIPT,
        input_schema={
            "type": "object",
            "properties": {"format": {"type": "string", "enum": ["xlsx"]}},
            "required": ["format"],
            "additionalProperties": False,
        },
        execution_config={"tool_key": "client.export.file"},
        timeout_seconds=20,
        version="1.0.0",
    )


def context(*, capability_available: bool = True) -> CompileContext:
    return CompileContext(
        session_id="session-1",
        client_capabilities=(
            ClientCapability(
                tool_id="client.export.file",
                version="1.0.0",
                available=capability_available,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_client_adapter_waits_for_transport_result() -> None:
    transport = RecordingTransport()
    tool = await ClientJavascriptAdapter(transport).build(definition(), context())

    result: Any = await tool.ainvoke({"format": "xlsx"})

    assert result == {"file_name": "测试文件.xlsx"}
    assert transport.availability_checks == [
        ("session-1", "client.export.file", "1.0.0")
    ]
    assert transport.requests[0].session_id == "session-1"
    assert transport.requests[0].tool_id == "client.export.file"
    assert transport.requests[0].arguments == {"format": "xlsx"}
    assert transport.requests[0].timeout_seconds == 20.0


@pytest.mark.asyncio
async def test_client_adapter_rejects_missing_session() -> None:
    with pytest.raises(ToolUnavailableError, match="session"):
        await ClientJavascriptAdapter(RecordingTransport()).build(
            definition(), CompileContext()
        )


@pytest.mark.asyncio
async def test_client_adapter_rejects_missing_capability() -> None:
    with pytest.raises(ToolUnavailableError, match="capability"):
        await ClientJavascriptAdapter(RecordingTransport()).build(
            definition(), CompileContext(session_id="session-1")
        )


@pytest.mark.asyncio
async def test_client_adapter_rejects_offline_transport() -> None:
    with pytest.raises(ToolUnavailableError, match="不可用"):
        await ClientJavascriptAdapter(RecordingTransport(available=False)).build(
            definition(), context()
        )


class TimeoutTransport(RecordingTransport):
    async def invoke(self, call: ClientToolRequest) -> ClientToolResult:
        del call
        raise ClientToolTimeoutError("secret-token /internal/client/timeout")


@pytest.mark.asyncio
async def test_client_adapter_preserves_transport_timeout_type() -> None:
    tool = await ClientJavascriptAdapter(TimeoutTransport()).build(
        definition(), context()
    )

    with pytest.raises(ClientToolTimeoutError) as captured:
        await tool.ainvoke({"format": "xlsx"})

    assert "secret-token" not in str(captured.value)
    assert "/internal/client/timeout" not in str(captured.value)


class FailingTransport(RecordingTransport):
    async def invoke(self, call: ClientToolRequest) -> ClientToolResult:
        del call
        raise RuntimeError("secret-token /internal/client/path")


@pytest.mark.asyncio
async def test_client_adapter_sanitizes_unknown_transport_failure() -> None:
    tool = await ClientJavascriptAdapter(FailingTransport()).build(
        definition(), context()
    )

    with pytest.raises(ToolExecutionError) as captured:
        await tool.ainvoke({"format": "xlsx"})

    assert "secret-token" not in str(captured.value)
    assert "/internal/client/path" not in str(captured.value)


class SlowTransport(RecordingTransport):
    async def invoke(self, call: ClientToolRequest) -> ClientToolResult:
        await asyncio.sleep(0.05)
        return await super().invoke(call)


@pytest.mark.asyncio
async def test_client_adapter_enforces_timeout_when_transport_does_not() -> None:
    slow_definition = definition().model_copy(update={"timeout_seconds": 0.01})
    tool = await ClientJavascriptAdapter(SlowTransport()).build(
        slow_definition, context()
    )

    with pytest.raises(ClientToolTimeoutError):
        await tool.ainvoke({"format": "xlsx"})
