from pathlib import Path

import pytest

from langchain_skill_runtime.errors import SkillParseError
from langchain_skill_runtime.parsing.skill_parser import SkillParser

TESTS_DIR = Path(__file__).resolve().parents[1]
FIXTURE = TESTS_DIR / "fixtures/skills/heterogeneous-tools/SKILL.md"


def test_parse_real_skill_fixture_preserves_instructions() -> None:
    parsed = SkillParser().parse(FIXTURE.read_text(encoding="utf-8"))

    assert parsed.name == "heterogeneous-tool-test"
    assert parsed.allowed_tools == (
        "add_numbers",
        "generate_report",
        "export_client_file",
        "mcp_echo",
    )
    assert "不得创建或调用未绑定的工具" in parsed.instructions


def test_parse_rejects_missing_frontmatter() -> None:
    with pytest.raises(SkillParseError, match="Frontmatter"):
        SkillParser().parse("# only markdown")
