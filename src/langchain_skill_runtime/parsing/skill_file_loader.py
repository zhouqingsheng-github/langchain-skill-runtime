"""Load one SKILL.md file into the runtime's normalized document model."""

from collections.abc import Mapping
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from pydantic import ValidationError

from langchain_skill_runtime.errors import (
    SkillFileNotFoundError,
    SkillReadError,
    ToolDefinitionError,
)
from langchain_skill_runtime.models.skill import SkillDocument
from langchain_skill_runtime.models.tool import ResolvedToolDefinition, ToolType
from langchain_skill_runtime.parsing.skill_parser import SkillParser
from langchain_skill_runtime.schemas.json_schema import JsonSchemaModelFactory


class SkillFileLoader:
    """Read UTF-8 SKILL.md files and normalize their Tool declarations."""

    def __init__(
        self,
        parser: SkillParser | None = None,
        schema_factory: JsonSchemaModelFactory | None = None,
    ) -> None:
        self._parser = parser or SkillParser()
        self._schema_factory = schema_factory or JsonSchemaModelFactory()

    def load(self, path: str | Path) -> SkillDocument:
        skill_path = Path(path)
        try:
            content = skill_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise SkillFileNotFoundError("SKILL.md 文件不存在") from None
        except (OSError, UnicodeError):
            raise SkillReadError("SKILL.md 文件读取失败") from None

        parsed = self._parser.parse(content)
        source_root = skill_path.resolve().parent
        tools = tuple(
            self._normalize_tool(raw, parsed.version, index, source_root)
            for index, raw in enumerate(parsed.tool_declarations)
        )
        return SkillDocument(
            id=parsed.id or parsed.name,
            name=parsed.name,
            description=parsed.description,
            version=parsed.version,
            instructions=parsed.instructions,
            allowed_tools=parsed.allowed_tools or tuple(tool.name for tool in tools),
            tools=tools,
            source_root=source_root,
            metadata=parsed.metadata,
        )

    def _normalize_tool(
        self,
        raw: Mapping[str, Any],
        skill_version: str,
        index: int,
        source_root: Path,
    ) -> ResolvedToolDefinition:
        name = self._required_text(raw, "name")
        description = self._required_text(raw, "description")
        raw_type = raw.get("type")
        if not isinstance(raw_type, str):
            raise ToolDefinitionError("SKILL.md Tool 类型不受支持")
        try:
            tool_type = ToolType(raw_type)
        except ValueError:
            raise ToolDefinitionError("SKILL.md Tool 类型不受支持") from None

        input_schema = raw.get("input_schema")
        if not isinstance(input_schema, dict):
            raise ToolDefinitionError("SKILL.md Tool input_schema 必须是对象")
        output_schema = raw.get("output_schema")
        if output_schema is not None and not isinstance(output_schema, dict):
            raise ToolDefinitionError("SKILL.md Tool output_schema 必须是对象")
        execution = raw.get("execution", {})
        if not isinstance(execution, Mapping):
            raise ToolDefinitionError("SKILL.md Tool execution 必须是对象")
        execution_config = dict(execution)
        if tool_type is ToolType.SERVER_SCRIPT:
            execution_config = self._resolve_script_execution(
                execution_config,
                source_root,
            )
        elif tool_type is ToolType.MCP:
            self._validate_file_mcp_execution(execution_config)

        self._validate_schemas(name, input_schema, output_schema)
        self._validate_execution_config(tool_type, execution_config)

        try:
            return ResolvedToolDefinition(
                id=self._optional_text(raw, "id") or name,
                name=name,
                description=description,
                tool_type=tool_type,
                input_schema=input_schema,
                output_schema=output_schema,
                execution_config=execution_config,
                timeout_seconds=raw.get("timeout_seconds", 30.0),
                max_output_bytes=raw.get("max_output_bytes", 1_048_576),
                version=self._optional_text(raw, "version") or skill_version,
                required=raw.get("required", True),
                enabled=raw.get("enabled", True),
                sort_order=raw.get("sort_order", index),
            )
        except ValidationError:
            raise ToolDefinitionError("SKILL.md Tool 定义非法") from None

    @staticmethod
    def _resolve_script_execution(
        execution: dict[str, Any],
        source_root: Path,
    ) -> dict[str, Any]:
        entry = execution.pop("entry", None)
        if not isinstance(entry, str) or not entry.strip():
            raise ToolDefinitionError("SKILL.md 脚本 Tool 必须配置 entry")
        relative_entry = Path(entry)
        if relative_entry.is_absolute():
            raise ToolDefinitionError("SKILL.md 脚本路径必须位于 Skill 根目录")
        resolved_entry = (source_root / relative_entry).resolve()
        try:
            resolved_entry.relative_to(source_root)
        except ValueError:
            raise ToolDefinitionError(
                "SKILL.md 脚本路径必须位于 Skill 根目录"
            ) from None
        return {**execution, "artifact_id": str(resolved_entry)}

    @staticmethod
    def _validate_file_mcp_execution(execution: Mapping[str, Any]) -> None:
        has_tool_name = "tool_name" in execution
        tool_name = execution.get("tool_name")
        server = execution.get("server")
        server_ref = execution.get("server_ref")
        if has_tool_name:
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ToolDefinitionError("SKILL.md MCP Tool tool_name 不能为空")
            if server is not None:
                raise ToolDefinitionError(
                    "SKILL.md 单个 MCP Tool 只能使用宿主预注册的 server_ref"
                )
            if not isinstance(server_ref, str) or not server_ref.strip():
                raise ToolDefinitionError("SKILL.md MCP Tool 必须配置 server_ref")
            return

        if server is not None and server_ref is not None:
            raise ToolDefinitionError("MCP 工具集合不能同时配置 server 和 server_ref")
        if server_ref is not None:
            if not isinstance(server_ref, str) or not server_ref.strip():
                raise ToolDefinitionError("SKILL.md MCP Tool server_ref 不能为空")
            return
        if not isinstance(server, Mapping):
            raise ToolDefinitionError(
                "SKILL.md MCP 工具集合必须配置 server 或 server_ref"
            )
        SkillFileLoader._validate_inline_mcp_server(server)

    @classmethod
    def _validate_inline_mcp_server(cls, server: Mapping[str, Any]) -> None:
        transport = server.get("transport")
        if not isinstance(transport, str) or not transport.strip():
            raise ToolDefinitionError("MCP Server 必须配置 transport")
        if transport == "stdio":
            raise ToolDefinitionError(
                "SKILL.md 内联 stdio MCP 必须改用宿主预注册的 server_ref"
            )
        if transport not in {"sse", "streamable_http", "streamable-http", "http"}:
            raise ToolDefinitionError("内联 MCP Server transport 不受支持")

        url = server.get("url")
        if not isinstance(url, str):
            raise ToolDefinitionError("内联 MCP Server 必须配置公网 HTTPS URL")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ToolDefinitionError("内联 MCP Server 必须使用公网 HTTPS URL")
        try:
            address = ip_address(parsed.hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ToolDefinitionError("内联 MCP Server 必须使用公网 HTTPS URL")
        if (
            parsed.username
            or parsed.password
            or any(
                cls._is_sensitive_name(name)
                for name, _ in parse_qsl(parsed.query, keep_blank_values=True)
            )
        ):
            raise ToolDefinitionError("MCP Server URL 不允许包含明文凭据")
        cls._validate_inline_mcp_values(server)

    @classmethod
    def _validate_inline_mcp_values(cls, value: Any) -> None:
        if cls._is_reference(value):
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if cls._is_sensitive_name(key) and not cls._is_reference(item):
                    raise ToolDefinitionError(
                        "MCP Server 敏感配置必须使用 env 或 secret_ref"
                    )
                cls._validate_inline_mcp_values(item)
        elif isinstance(value, list):
            for item in value:
                cls._validate_inline_mcp_values(item)

    @staticmethod
    def _is_reference(value: Any) -> bool:
        return (
            isinstance(value, Mapping)
            and len(value) == 1
            and next(iter(value), None) in {"env", "secret_ref"}
            and isinstance(next(iter(value.values()), None), str)
            and bool(next(iter(value.values()), "").strip())
        )

    @staticmethod
    def _is_sensitive_name(value: Any) -> bool:
        normalized = "".join(
            character for character in str(value).casefold() if character.isalnum()
        )
        return normalized == "key" or any(
            marker in normalized
            for marker in (
                "apikey",
                "authorization",
                "cookie",
                "credential",
                "password",
                "secret",
                "token",
            )
        )

    def _validate_schemas(
        self,
        name: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any] | None,
    ) -> None:
        self._schema_factory.create(
            f"{name.title().replace('_', '')}Input",
            input_schema,
        )
        if output_schema is None:
            return
        try:
            Draft202012Validator.check_schema(output_schema)
        except SchemaError:
            raise ToolDefinitionError(
                "Tool output_schema 不是合法 JSON Schema"
            ) from None

    @staticmethod
    def _validate_execution_config(
        tool_type: ToolType,
        execution: Mapping[str, Any],
    ) -> None:
        required_keys: dict[ToolType, tuple[str, ...]] = {
            ToolType.PYTHON_FUNCTION: ("registry_key",),
            ToolType.SERVER_SCRIPT: ("artifact_id",),
            ToolType.CLIENT_JAVASCRIPT: ("tool_key",),
            ToolType.MCP: ("server_name",),
        }
        for key in required_keys[tool_type]:
            value = execution.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ToolDefinitionError(
                    f"SKILL.md {tool_type.value} Tool 必须配置 {key}"
                )

    @staticmethod
    def _required_text(raw: Mapping[str, Any], key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ToolDefinitionError(f"SKILL.md Tool 的 {key} 不能为空")
        return value.strip()

    @staticmethod
    def _optional_text(raw: Mapping[str, Any], key: str) -> str | None:
        value = raw.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ToolDefinitionError(f"SKILL.md Tool 的 {key} 必须是非空字符串")
        return value.strip()
