"""Compile parsed Skill instructions and actually available Tools."""

from collections.abc import Sequence

from langchain_skill_runtime.models.skill import ParsedSkill
from langchain_skill_runtime.models.tool import ResolvedToolDefinition


class PromptCompiler:
    """Create a deterministic system-prompt fragment for one Skill."""

    def compile(
        self,
        parsed_skill: ParsedSkill,
        tools: Sequence[ResolvedToolDefinition],
    ) -> str:
        tool_lines = [f"- {tool.name}: {tool.description}" for tool in tools]
        if not tool_lines:
            tool_lines.append("- 无")

        return "\n\n".join(
            (
                f"# Skill: {parsed_skill.name}",
                f"## Description\n{parsed_skill.description}",
                f"## Instructions\n{parsed_skill.instructions}",
                "## Available Tools\n" + "\n".join(tool_lines),
            )
        )
