import pytest

from langchain_skill_runtime.errors import SkillParseError
from langchain_skill_runtime.parsing.skill_parser import SkillParser


def test_parse_rejects_missing_frontmatter() -> None:
    with pytest.raises(SkillParseError, match="Frontmatter"):
        SkillParser().parse("# only markdown")
