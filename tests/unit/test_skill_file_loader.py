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


def test_file_loader_accepts_registered_mcp_server_reference(tmp_path: Path) -> None:
    skill_path = write_skill(
        tmp_path,
        """name: safe-mcp
description: 引用宿主预注册 MCP 服务
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
      server_ref: mcp/hotel""",
    )

    document = SkillFileLoader().load(skill_path)

    assert document.tools[0].execution_config == {
        "server_name": "hotel",
        "tool_name": "query_hotel",
        "server_ref": "mcp/hotel",
    }


@pytest.mark.parametrize("tool_name", ["", None, 123])
def test_file_loader_rejects_invalid_legacy_mcp_tool_name(
    tmp_path: Path,
    tool_name: object,
) -> None:
    rendered_tool_name = "null" if tool_name is None else repr(tool_name)
    skill_path = write_skill(
        tmp_path,
        f"""name: invalid-mcp-tool-name
description: 非法旧版 MCP Tool 名称
tools:
  - name: query_hotel
    description: 查询酒店数据
    type: MCP
    input_schema:
      type: object
      properties: {{}}
    execution:
      server_name: hotel
      tool_name: {rendered_tool_name}
      server_ref: mcp/hotel""",
    )

    with pytest.raises(ToolDefinitionError, match="tool_name"):
        SkillFileLoader().load(skill_path)


def test_file_loader_accepts_inline_mcp_tool_collection(tmp_path: Path) -> None:
    skill_path = write_skill(
        tmp_path,
        """name: amap-skill
description: 高德 MCP 工具集合
tools:
  - name: amap_maps
    description: 高德地图 MCP 提供的全部工具
    type: MCP
    input_schema:
      type: object
      properties: {}
      additionalProperties: false
    execution:
      server_name: amap
      server:
        transport: streamable_http
        url: https://mcp.amap.com/mcp
        query:
          key:
            env: AMAP_MAPS_API_KEY""",
    )

    document = SkillFileLoader().load(skill_path)

    assert document.tools[0].name == "amap_maps"
    assert document.tools[0].tool_type is ToolType.MCP
    assert document.tools[0].execution_config == {
        "server_name": "amap",
        "server": {
            "transport": "streamable_http",
            "url": "https://mcp.amap.com/mcp",
            "query": {"key": {"env": "AMAP_MAPS_API_KEY"}},
        },
    }


def test_file_loader_rejects_plaintext_mcp_collection_query_key(
    tmp_path: Path,
) -> None:
    skill_path = write_skill(
        tmp_path,
        """name: unsafe-amap-skill
description: 包含明文 Key 的 MCP 工具集合
tools:
  - name: amap_maps
    description: 高德地图 MCP 提供的全部工具
    type: MCP
    input_schema:
      type: object
      properties: {}
    execution:
      server_name: amap
      server:
        transport: streamable_http
        url: https://mcp.amap.com/mcp
        query:
          key: plaintext-amap-key""",
    )

    with pytest.raises(ToolDefinitionError, match="env 或 secret_ref"):
        SkillFileLoader().load(skill_path)


def test_file_loader_rejects_inline_stdio_mcp_collection(tmp_path: Path) -> None:
    skill_path = write_skill(
        tmp_path,
        """name: unsafe-stdio-collection
description: 尝试启动本地进程的 MCP 工具集合
tools:
  - name: local_tools
    description: 本地 MCP 工具集合
    type: MCP
    input_schema:
      type: object
      properties: {}
    execution:
      server_name: local
      server:
        transport: stdio
        command: /bin/sh
        args: [-c, whoami]""",
    )

    with pytest.raises(ToolDefinitionError, match="server_ref"):
        SkillFileLoader().load(skill_path)


@pytest.mark.parametrize("transport", ["websocket", "unknown"])
def test_file_loader_rejects_unsupported_inline_mcp_transport(
    tmp_path: Path,
    transport: str,
) -> None:
    skill_path = write_skill(
        tmp_path,
        f"""name: unsafe-inline-transport
description: 不支持的内联 MCP 协议
tools:
  - name: unsafe_tools
    description: 非法 MCP 工具集合
    type: MCP
    input_schema:
      type: object
      properties: {{}}
    execution:
      server_name: unsafe
      server:
        transport: {transport}
        url: https://mcp.example.com/mcp""",
    )

    with pytest.raises(ToolDefinitionError, match="transport"):
        SkillFileLoader().load(skill_path)


@pytest.mark.parametrize(
    "url",
    [
        "http://mcp.example.com/mcp",
        "file:///etc/passwd",
        "https://127.0.0.1/mcp",
        "https://[::1]/mcp",
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.8/mcp",
    ],
)
def test_file_loader_rejects_non_public_https_inline_mcp_urls(
    tmp_path: Path,
    url: str,
) -> None:
    skill_path = write_skill(
        tmp_path,
        f"""name: unsafe-inline-url
description: 非安全 MCP URL
tools:
  - name: unsafe_tools
    description: 非安全 MCP 工具集合
    type: MCP
    input_schema:
      type: object
      properties: {{}}
    execution:
      server_name: unsafe
      server:
        transport: streamable_http
        url: {url}""",
    )

    with pytest.raises(ToolDefinitionError, match="公网 HTTPS"):
        SkillFileLoader().load(skill_path)


def test_file_loader_accepts_registered_mcp_tool_collection(tmp_path: Path) -> None:
    skill_path = write_skill(
        tmp_path,
        """name: registered-mcp-collection
description: 宿主预注册 MCP 工具集合
tools:
  - name: internal_tools
    description: 内部 MCP Server 提供的全部工具
    type: MCP
    input_schema:
      type: object
      properties: {}
    execution:
      server_name: internal
      server_ref: mcp/internal""",
    )

    document = SkillFileLoader().load(skill_path)

    assert document.tools[0].execution_config == {
        "server_name": "internal",
        "server_ref": "mcp/internal",
    }


@pytest.mark.parametrize(
    "server_config",
    [
        """server:
        transport: stdio
        command: /bin/sh
        args: [-c, whoami]""",
        """server:
        transport: http
        url: https://mcp.example.com/mcp
        headers:
          Authorization: Bearer plaintext-token""",
        """server:
        transport: stdio
        command: python
        env:
          API_KEY: plaintext-token""",
    ],
)
def test_file_loader_rejects_inline_mcp_server_configuration(
    tmp_path: Path,
    server_config: str,
) -> None:
    skill_path = write_skill(
        tmp_path,
        f"""name: unsafe-mcp
description: 包含未审核 MCP 服务配置
tools:
  - name: query_hotel
    description: 查询酒店数据
    type: MCP
    input_schema:
      type: object
      properties: {{}}
    execution:
      server_name: hotel
      tool_name: query_hotel
      {server_config}""",
    )

    with pytest.raises(ToolDefinitionError, match="server_ref"):
        SkillFileLoader().load(skill_path)


@pytest.mark.parametrize(
    ("tool_type", "execution", "error_pattern"),
    [
        ("PYTHON_FUNCTION", "{}", "registry_key"),
        ("CLIENT_JAVASCRIPT", "{}", "tool_key"),
        ("MCP", "{server_name: hotel, tool_name: query_hotel}", "server_ref"),
    ],
)
def test_file_loader_rejects_missing_execution_configuration(
    tmp_path: Path,
    tool_type: str,
    execution: str,
    error_pattern: str,
) -> None:
    skill_path = write_skill(
        tmp_path,
        f"""name: invalid-execution
description: 缺少执行配置
tools:
  - name: invalid_tool
    description: 非法工具
    type: {tool_type}
    input_schema:
      type: object
      properties: {{}}
    execution: {execution}""",
    )

    with pytest.raises(ToolDefinitionError, match=error_pattern):
        SkillFileLoader().load(skill_path)


def test_file_loader_rejects_invalid_input_schema(tmp_path: Path) -> None:
    skill_path = write_skill(
        tmp_path,
        """name: invalid-schema
description: 非法输入 Schema
tools:
  - name: invalid_tool
    description: 非法工具
    type: PYTHON_FUNCTION
    input_schema:
      type: string
    execution:
      registry_key: invalid.tool""",
    )

    with pytest.raises(ToolDefinitionError, match="input_schema"):
        SkillFileLoader().load(skill_path)


def test_file_loader_rejects_invalid_output_schema(tmp_path: Path) -> None:
    skill_path = write_skill(
        tmp_path,
        """name: invalid-output-schema
description: 非法输出 Schema
tools:
  - name: invalid_tool
    description: 非法工具
    type: PYTHON_FUNCTION
    input_schema:
      type: object
      properties: {}
    output_schema:
      type: not-a-json-schema-type
    execution:
      registry_key: invalid.tool""",
    )

    with pytest.raises(ToolDefinitionError, match="output_schema"):
        SkillFileLoader().load(skill_path)
