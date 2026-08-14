import sys
from pathlib import Path

import pytest

from langchain_skill_runtime.adapters.mcp import LangChainMcpToolProvider
from langchain_skill_runtime.models.context import CompileContext

TESTS_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_local_stdio_mcp_tool_is_discovered_and_invoked() -> None:
    provider = LangChainMcpToolProvider()
    server_path = TESTS_DIR / "fixtures/mcp/echo_server.py"

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


@pytest.mark.asyncio
async def test_local_stdio_mcp_collection_discovers_and_invokes_all_tools() -> None:
    provider = LangChainMcpToolProvider()
    server_path = TESTS_DIR / "fixtures/mcp/echo_server.py"

    discovered = await provider.get_tools(
        "echo",
        {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(server_path)],
        },
        CompileContext(),
    )

    assert [tool.name for tool in discovered] == ["mcp_echo", "mcp_text_length"]
    result = await discovered[1].ainvoke({"text": "route"})
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "5"
