"""对外公开示例的可运行性与 Skill 结构。"""

import json
import subprocess
import sys
from pathlib import Path

from langchain_skill_runtime import ToolType
from langchain_skill_runtime.parsing.skill_file_loader import SkillFileLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_example(relative_path: str) -> str:
    completed = subprocess.run(
        [sys.executable, relative_path],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_calculator_example_runs_from_repository_root() -> None:
    assert run_example("examples/calculator/main.py") == "42"


def test_client_tool_example_runs_request_and_result_round_trip() -> None:
    assert json.loads(run_example("examples/client_tool/main.py")) == {
        "file_name": "example.xlsx"
    }


def test_mcp_collection_example_is_a_server_collection() -> None:
    document = SkillFileLoader().load(
        PROJECT_ROOT / "examples/mcp_tool_collection/SKILL.md"
    )

    assert len(document.tools) == 1
    tool = document.tools[0]
    assert tool.tool_type is ToolType.MCP
    assert "tool_name" not in tool.execution_config
    assert tool.execution_config["server"]["headers"]["Authorization"] == {
        "env": "EXAMPLE_MCP_AUTHORIZATION"
    }
