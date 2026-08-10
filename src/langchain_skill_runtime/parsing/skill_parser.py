"""Parse Agent Skills YAML Frontmatter without rewriting instructions."""

from collections.abc import Mapping
from typing import Any

import yaml

from langchain_skill_runtime.errors import SkillParseError
from langchain_skill_runtime.models.skill import ParsedSkill


class SkillParser:
    """Parse a complete SKILL.md string."""

    def parse(self, content: str) -> ParsedSkill:
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            raise SkillParseError("SKILL.md 缺少 YAML Frontmatter 起始分隔符")

        closing_index = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            ),
            None,
        )
        if closing_index is None:
            raise SkillParseError("SKILL.md 缺少 YAML Frontmatter 结束分隔符")

        try:
            raw_frontmatter = yaml.safe_load("\n".join(lines[1:closing_index]))
        except yaml.YAMLError as exc:
            raise SkillParseError("SKILL.md YAML Frontmatter 解析失败") from exc

        if not isinstance(raw_frontmatter, Mapping):
            raise SkillParseError("SKILL.md YAML Frontmatter 必须是对象")

        frontmatter: Mapping[str, Any] = raw_frontmatter
        name = self._required_text(frontmatter, "name")
        description = self._required_text(frontmatter, "description")
        instructions = "\n".join(lines[closing_index + 1 :]).strip()
        if not instructions:
            raise SkillParseError("SKILL.md Markdown 正文不能为空")

        metadata = frontmatter.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise SkillParseError("SKILL.md metadata 必须是对象")

        compatibility = frontmatter.get("compatibility")
        if compatibility is not None and not isinstance(compatibility, str):
            raise SkillParseError("SKILL.md compatibility 必须是字符串")

        return ParsedSkill(
            name=name,
            description=description,
            instructions=instructions,
            compatibility=compatibility,
            allowed_tools=self._allowed_tools(frontmatter.get("allowed-tools", ())),
            metadata=dict(metadata),
        )

    @staticmethod
    def _required_text(frontmatter: Mapping[str, Any], key: str) -> str:
        value = frontmatter.get(key)
        if not isinstance(value, str) or not value.strip():
            raise SkillParseError(f"SKILL.md Frontmatter 的 {key} 不能为空")
        return value.strip()

    @staticmethod
    def _allowed_tools(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(item for item in value.split() if item)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return tuple(item.strip() for item in value if item.strip())
        raise SkillParseError("SKILL.md allowed-tools 必须是字符串或字符串列表")
