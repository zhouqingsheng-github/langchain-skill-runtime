"""Discover all tools exposed by a remote MCP server."""

import asyncio
from pathlib import Path

from langchain_skill_runtime import SkillRuntime
from langchain_skill_runtime.adapters import (
    AllowHostsMcpUrlPolicy,
    LangChainMcpToolProvider,
    McpToolAdapter,
    ToolFactory,
)

SKILL_PATH = Path(__file__).with_name("SKILL.md")


async def main() -> None:
    provider = LangChainMcpToolProvider(
        url_policy=AllowHostsMcpUrlPolicy({"mcp.example.com"})
    )
    runtime = SkillRuntime(tool_factory=ToolFactory([McpToolAdapter(provider)]))
    bundle = await runtime.compile_file(SKILL_PATH)
    print([tool.name for tool in bundle.tools])


if __name__ == "__main__":
    asyncio.run(main())
