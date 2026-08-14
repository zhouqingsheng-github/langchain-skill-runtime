"""Delegate a Tool call to a client and correlate its returned result."""

import asyncio
import json
from pathlib import Path

from langchain_skill_runtime import ClientCapability, CompileContext, SkillRuntime
from langchain_skill_runtime.adapters import ClientJavascriptAdapter, ToolFactory
from langchain_skill_runtime.client import (
    ClientToolRequest,
    ClientToolResult,
    ClientToolStatus,
    PendingClientToolTransport,
)

SKILL_PATH = Path(__file__).with_name("SKILL.md")


async def main() -> None:
    requests: asyncio.Queue[ClientToolRequest] = asyncio.Queue()

    async def send_to_client(request: ClientToolRequest) -> None:
        await requests.put(request)

    transport = PendingClientToolTransport(send_to_client)
    runtime = SkillRuntime(
        tool_factory=ToolFactory([ClientJavascriptAdapter(transport)])
    )
    context = CompileContext(
        session_id="example-session",
        client_capabilities=(
            ClientCapability(tool_id="client.export.file", version="1.0.0"),
        ),
    )
    bundle = await runtime.compile_file(SKILL_PATH, context)

    invocation = asyncio.create_task(bundle.tools[0].ainvoke({"format": "xlsx"}))
    request = await requests.get()
    await transport.accept_result(
        ClientToolResult(
            call_id=request.call_id,
            session_id=request.session_id,
            tool_id=request.tool_id,
            tool_version=request.tool_version,
            status=ClientToolStatus.SUCCESS,
            output={"file_name": "example.xlsx"},
        )
    )
    print(json.dumps(await invocation, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
