from langchain_skill_runtime.models.skill import ParsedSkill
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType
from langchain_skill_runtime.prompting.prompt_compiler import PromptCompiler


def test_prompt_lists_only_successfully_built_tools() -> None:
    parsed = ParsedSkill(
        name="database-name-is-authoritative",
        description="测试 Skill",
        instructions="只调用已经装配成功的工具。",
        allowed_tools=("add_numbers", "unavailable_tool"),
    )
    tool = ResolvedToolDefinition(
        id="tool-1",
        name="add_numbers",
        description="计算两个数字之和",
        tool_type=ToolType.PYTHON_FUNCTION,
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        execution_config={"registry_key": "math.add"},
        version="1.0.0",
    )

    prompt = PromptCompiler().compile(parsed, [tool])

    assert "# Skill: database-name-is-authoritative" in prompt
    assert "只调用已经装配成功的工具。" in prompt
    assert "- add_numbers: 计算两个数字之和" in prompt
    assert "unavailable_tool" not in prompt
    assert "registry_key" not in prompt
