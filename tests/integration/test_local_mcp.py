import sys
from pathlib import Path

import pytest

from langchain_skill_runtime.adapters.mcp import LangChainMcpToolProvider
from langchain_skill_runtime.models.context import CompileContext


@pytest.mark.asyncio
async def test_local_stdio_mcp_tool_is_discovered_and_invoked() -> None:
    provider = LangChainMcpToolProvider()
    server_path = Path("tests/fixtures/mcp/echo_server.py").resolve()

    discovered = await provider.get_tool(
        "echo",
        "mcp_echo",
        {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(server_path)],
        },
        CompileContext(),
    )

    assert discovered is not None
    result = await discovered.ainvoke({"text": "ok"})
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "mcp:ok"
