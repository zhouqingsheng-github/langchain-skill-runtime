from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool, tool

from langchain_skill_runtime import CompileContext, SkillRuntime
from langchain_skill_runtime.adapters import McpToolAdapter, ToolFactory

TESTS_DIR = Path(__file__).resolve().parents[1]
SKILL_PATH = TESTS_DIR / "fixtures/skills/mcp-tool-collection/SKILL.md"


@tool
async def maps_geo(address: str) -> str:
    """将地址解析为经纬度。"""

    return f"location:{address}"


@tool
async def maps_direction_driving(origin: str, destination: str) -> str:
    """规划驾车路线。"""

    return f"route:{origin}:{destination}"


class MapsCollectionProvider:
    async def get_tools(
        self,
        server_name: str,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> list[BaseTool]:
        del context
        if server_name != "maps":
            raise AssertionError("MCP server_name 未按 SKILL.md 传递")
        expected_config = {
            "transport": "streamable_http",
            "url": "https://maps.example.invalid/mcp",
            "query": {"key": {"env": "TEST_MAPS_MCP_KEY"}},
        }
        if server_config != expected_config:
            raise AssertionError("MCP 环境变量引用未按 SKILL.md 原样传递")
        return [maps_geo, maps_direction_driving]


@pytest.mark.asyncio
async def test_real_skill_expands_one_mcp_object_into_all_server_tools() -> None:
    runtime = SkillRuntime(
        tool_factory=ToolFactory([McpToolAdapter(MapsCollectionProvider())])
    )

    bundle = await runtime.compile_file(SKILL_PATH)

    print(await bundle.tools[1].ainvoke({"origin": "杭州东站", "destination": "西湖"}))
