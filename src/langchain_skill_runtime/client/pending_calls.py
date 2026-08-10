"""Concurrent correlation of client requests and results."""

import asyncio
from dataclasses import dataclass

from langchain_skill_runtime.client.models import (
    ClientToolRequest,
    ClientToolResult,
    ClientToolStatus,
)
from langchain_skill_runtime.errors import (
    ClientToolConnectionLostError,
    ClientToolExecutionError,
    ClientToolResultMismatchError,
)


@dataclass(frozen=True)
class _PendingCall:
    request: ClientToolRequest
    future: asyncio.Future[ClientToolResult]


class PendingCallManager:
    """Own pending calls without owning the underlying WebSocket."""

    def __init__(self) -> None:
        self._pending: dict[str, _PendingCall] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        request: ClientToolRequest,
    ) -> asyncio.Future[ClientToolResult]:
        future = asyncio.get_running_loop().create_future()
        async with self._lock:
            if request.call_id in self._pending:
                raise ValueError(f"重复的客户端 Tool call_id: {request.call_id}")
            self._pending[request.call_id] = _PendingCall(request, future)
        return future

    async def remove(self, call_id: str) -> None:
        async with self._lock:
            self._pending.pop(call_id, None)

    async def accept_result(self, result: ClientToolResult) -> bool:
        async with self._lock:
            pending = self._pending.get(result.call_id)
            if pending is None or pending.future.done():
                return False
            self._validate_result(pending.request, result)
            if result.status is ClientToolStatus.SUCCESS:
                pending.future.set_result(result)
            else:
                message = result.error_message or result.error_code or "客户端工具执行失败"
                pending.future.set_exception(ClientToolExecutionError(message))
            return True

    async def fail_session(self, session_id: str) -> int:
        async with self._lock:
            affected = [
                pending
                for pending in self._pending.values()
                if pending.request.session_id == session_id and not pending.future.done()
            ]
            for pending in affected:
                pending.future.set_exception(
                    ClientToolConnectionLostError("客户端连接已断开")
                )
            return len(affected)

    @staticmethod
    def _validate_result(
        request: ClientToolRequest,
        result: ClientToolResult,
    ) -> None:
        expected = (
            request.session_id,
            request.tool_id,
            request.tool_version,
        )
        actual = (
            result.session_id,
            result.tool_id,
            result.tool_version,
        )
        if actual != expected:
            raise ClientToolResultMismatchError("客户端结果与待处理调用不匹配")
