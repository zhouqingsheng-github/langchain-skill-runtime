from pathlib import Path

import pytest

from langchain_skill_runtime.errors import (
    SkillFileNotFoundError,
    ToolDefinitionError,
)
from langchain_skill_runtime.models.tool import ToolType
from langchain_skill_runtime.parsing.skill_file_loader import SkillFileLoader


def write_skill(tmp_path: Path, frontmatter: str, body: str = "# 使用说明") -> Path:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return skill_path


def test_file_loader_builds_document_from_frontmatter(tmp_path: Path) -> None:
    skill_path = write_skill(
        tmp_path,
        """id: calculator
name: calculator
description: 计算器技能
version: 1.2.0
tools:
  - name: add_numbers
    description: 计算两个数字之和
    type: PYTHON_FUNCTION
    input_schema:
      type: object
      properties:
        a: {type: integer}
        b: {type: integer}
      required: [a, b]
      additionalProperties: false
    output_schema:
      type: integer
    execution:
      registry_key: math.add""",
    )

    document = SkillFileLoader().load(skill_path)

    assert document.id == "calculator"
    assert document.version == "1.2.0"
    assert document.instructions == "# 使用说明"
    assert document.source_root == tmp_path.resolve()
    assert [tool.name for tool in document.tools] == ["add_numbers"]
    assert document.tools[0].tool_type is ToolType.PYTHON_FUNCTION
    assert document.tools[0].execution_config == {"registry_key": "math.add"}
    assert document.tools[0].version == "1.2.0"


def test_file_loader_accepts_prompt_only_market_skill(tmp_path: Path) -> None:
    skill_path = write_skill(
        tmp_path,
        """name: market-skill
description: 不声明工具的市场技能
allowed-tools:
  - external_tool""",
    )

    document = SkillFileLoader().load(skill_path)

    assert document.id == "market-skill"
    assert document.version == "0.0.0"
    assert document.allowed_tools == ("external_tool",)
    assert document.tools == ()


def test_file_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SkillFileNotFoundError):
        SkillFileLoader().load(tmp_path / "SKILL.md")


def test_file_loader_resolves_script_inside_skill_root(tmp_path: Path) -> None:
    skill_path = write_skill(
        tmp_path,
        """name: report-skill
description: 报告技能
tools:
  - name: generate_report
    description: 生成报告
    type: SERVER_SCRIPT
    input_schema:
      type: object
      properties: {}
    execution:
      entry: scripts/generate_report.py""",
    )

    document = SkillFileLoader().load(skill_path)

    assert document.tools[0].execution_config == {
        "artifact_id": str((tmp_path / "scripts/generate_report.py").resolve())
    }


@pytest.mark.parametrize("entry", ["../escape.py", "/tmp/escape.py"])
def test_file_loader_rejects_script_outside_skill_root(
    tmp_path: Path,
    entry: str,
) -> None:
    skill_path = write_skill(
        tmp_path,
        f"""name: unsafe-script
description: 非法脚本路径
tools:
  - name: unsafe_script
    description: 非法脚本
    type: SERVER_SCRIPT
    input_schema:
      type: object
      properties: {{}}
    execution:
      entry: {entry}""",
    )

    with pytest.raises(ToolDefinitionError, match="Skill 根目录"):
        SkillFileLoader().load(skill_path)


def test_file_loader_rejects_unknown_tool_type(tmp_path: Path) -> None:
    skill_path = write_skill(
        tmp_path,
        """name: invalid-tool
description: 非法工具类型
tools:
  - name: unknown
    description: 未知工具
    type: UNKNOWN
    input_schema:
      type: object
      properties: {}
    execution: {}""",
    )

    with pytest.raises(ToolDefinitionError, match="Tool 类型"):
        SkillFileLoader().load(skill_path)


def test_file_loader_rejects_plaintext_mcp_credentials(tmp_path: Path) -> None:
    skill_path = write_skill(
        tmp_path,
        """name: unsafe-mcp
description: 包含明文凭据的 MCP 工具
tools:
  - name: query_hotel
    description: 查询酒店数据
    type: MCP
    input_schema:
      type: object
      properties: {}
    execution:
      server_name: hotel
      tool_name: query_hotel
      server:
        transport: http
        url: https://mcp.example.com/mcp
        headers:
          Authorization: Bearer plaintext-token""",
    )

    with pytest.raises(ToolDefinitionError, match="secret_ref"):
        SkillFileLoader().load(skill_path)
