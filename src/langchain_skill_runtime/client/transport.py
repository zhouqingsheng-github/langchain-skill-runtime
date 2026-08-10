"""Client Tool transport protocol and generic pending-call implementation."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from langchain_skill_runtime.client.models import ClientToolRequest, ClientToolResult
from langchain_skill_runtime.client.pending_calls import PendingCallManager
from langchain_skill_runtime.errors import (
    ClientToolConnectionLostError,
    ClientToolTimeoutError,
)


class ClientToolTransport(Protocol):
    """Availability check and request/response transport contract."""

    async def is_available(
        self,
        session_id: str,
        tool_id: str,
        tool_version: str,
    ) -> bool: ...

    async def invoke(self, request: ClientToolRequest) -> ClientToolResult: ...


ClientToolSender = Callable[[ClientToolRequest], Awaitable[None]]
ClientAvailabilityChecker = Callable[[str, str, str], Awaitable[bool]]


class PendingClientToolTransport:
    """Wait for business WebSocket code to call accept_result()."""

    def __init__(
        self,
        sender: ClientToolSender,
        availability_checker: ClientAvailabilityChecker | None = None,
    ) -> None:
        self._sender = sender
        self._availability_checker = availability_checker
        self._pending = PendingCallManager()

    async def is_available(
        self,
        session_id: str,
        tool_id: str,
        tool_version: str,
    ) -> bool:
        if self._availability_checker is None:
            return True
        return await self._availability_checker(session_id, tool_id, tool_version)

    async def invoke(self, request: ClientToolRequest) -> ClientToolResult:
        future = await self._pending.register(request)
        try:
            try:
                await self._sender(request)
            except Exception:  # noqa: BLE001 - sanitize business transport failures
                raise ClientToolConnectionLostError(
                    "客户端 Tool 请求发送失败"
                ) from None
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=float(request.timeout_seconds),
                )
            except TimeoutError as exc:
                raise ClientToolTimeoutError("客户端工具执行超时") from exc
        finally:
            await self._pending.remove(request.call_id)

    async def accept_result(self, result: ClientToolResult) -> bool:
        return await self._pending.accept_result(result)

    async def fail_session(self, session_id: str) -> int:
        return await self._pending.fail_session(session_id)
