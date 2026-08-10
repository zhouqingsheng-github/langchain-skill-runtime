import asyncio

import pytest

from langchain_skill_runtime.client.models import (
    ClientToolRequest,
    ClientToolResult,
    ClientToolStatus,
)
from langchain_skill_runtime.client.transport import PendingClientToolTransport
from langchain_skill_runtime.errors import (
    ClientToolConnectionLostError,
    ClientToolExecutionError,
    ClientToolResultMismatchError,
    ClientToolTimeoutError,
)


def request(timeout_seconds: float = 1.0) -> ClientToolRequest:
    return ClientToolRequest(
        session_id="session-1",
        tool_id="client.export.file",
        tool_version="1.0.0",
        arguments={"format": "xlsx"},
        timeout_seconds=timeout_seconds,
    )


def successful_result(call: ClientToolRequest) -> ClientToolResult:
    return ClientToolResult(
        call_id=call.call_id,
        session_id=call.session_id,
        tool_id=call.tool_id,
        tool_version=call.tool_version,
        status=ClientToolStatus.SUCCESS,
        output={"file_name": "测试文件.xlsx"},
    )


@pytest.mark.asyncio
async def test_transport_correlates_result_by_call_id() -> None:
    sent: asyncio.Queue[ClientToolRequest] = asyncio.Queue()

    async def sender(call: ClientToolRequest) -> None:
        await sent.put(call)

    transport = PendingClientToolTransport(sender)
    invocation = asyncio.create_task(transport.invoke(request()))
    emitted = await sent.get()

    assert await transport.accept_result(successful_result(emitted)) is True
    assert (await invocation).output == {"file_name": "测试文件.xlsx"}
    assert await transport.accept_result(successful_result(emitted)) is False


@pytest.mark.asyncio
async def test_transport_times_out_and_rejects_late_result() -> None:
    sent: asyncio.Queue[ClientToolRequest] = asyncio.Queue()

    async def sender(call: ClientToolRequest) -> None:
        await sent.put(call)

    transport = PendingClientToolTransport(sender)
    invocation = asyncio.create_task(transport.invoke(request(0.01)))
    emitted = await sent.get()

    with pytest.raises(ClientToolTimeoutError):
        await invocation
    assert await transport.accept_result(successful_result(emitted)) is False


@pytest.mark.asyncio
async def test_transport_rejects_mismatched_result_without_losing_call() -> None:
    sent: asyncio.Queue[ClientToolRequest] = asyncio.Queue()

    async def sender(call: ClientToolRequest) -> None:
        await sent.put(call)

    transport = PendingClientToolTransport(sender)
    invocation = asyncio.create_task(transport.invoke(request()))
    emitted = await sent.get()
    mismatch = successful_result(emitted).model_copy(update={"session_id": "forged"})

    with pytest.raises(ClientToolResultMismatchError):
        await transport.accept_result(mismatch)

    await transport.accept_result(successful_result(emitted))
    assert (await invocation).status is ClientToolStatus.SUCCESS


@pytest.mark.asyncio
async def test_transport_fails_all_pending_calls_for_disconnected_session() -> None:
    sent: asyncio.Queue[ClientToolRequest] = asyncio.Queue()

    async def sender(call: ClientToolRequest) -> None:
        await sent.put(call)

    transport = PendingClientToolTransport(sender)
    invocation = asyncio.create_task(transport.invoke(request()))
    await sent.get()

    assert await transport.fail_session("session-1") == 1
    with pytest.raises(ClientToolConnectionLostError):
        await invocation


@pytest.mark.asyncio
async def test_transport_converts_remote_error_to_typed_exception() -> None:
    sent: asyncio.Queue[ClientToolRequest] = asyncio.Queue()

    async def sender(call: ClientToolRequest) -> None:
        await sent.put(call)

    transport = PendingClientToolTransport(sender)
    invocation = asyncio.create_task(transport.invoke(request()))
    emitted = await sent.get()
    result = ClientToolResult(
        call_id=emitted.call_id,
        session_id=emitted.session_id,
        tool_id=emitted.tool_id,
        tool_version=emitted.tool_version,
        status=ClientToolStatus.ERROR,
        error_code="EXPORT_FAILED",
        error_message="导出失败",
    )

    assert await transport.accept_result(result) is True
    with pytest.raises(ClientToolExecutionError, match="导出失败"):
        await invocation
