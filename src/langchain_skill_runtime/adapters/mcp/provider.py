"""LangChain MCP client provider and secure connection preparation."""

import asyncio
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, Protocol, Self, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langchain_core.tools import BaseTool

from langchain_skill_runtime.adapters.mcp.protocols import (
    McpServerConfigProvider,
    McpUrlPolicy,
)
from langchain_skill_runtime.adapters.mcp.url_policy import PublicHttpsMcpUrlPolicy
from langchain_skill_runtime.errors import ToolDefinitionError, ToolUnavailableError
from langchain_skill_runtime.models.context import CompileContext
from langchain_skill_runtime.secrets import SecretProvider


class _McpClient(Protocol):
    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]: ...


McpClientFactory = Callable[[dict[str, Mapping[str, Any]]], _McpClient]
McpSessionToolLoader = Callable[[Any], Awaitable[list[BaseTool]]]


class LangChainMcpToolProvider:
    """Use langchain-mcp-adapters while keeping it an optional dependency."""

    def __init__(
        self,
        secret_provider: SecretProvider | None = None,
        client_factory: McpClientFactory | None = None,
        *,
        server_config_provider: McpServerConfigProvider | None = None,
        url_policy: McpUrlPolicy | None = None,
        session_tool_loader: McpSessionToolLoader | None = None,
    ) -> None:
        self._secret_provider = secret_provider
        self._server_config_provider = server_config_provider
        self._url_policy = url_policy or PublicHttpsMcpUrlPolicy()
        self._has_explicit_url_policy = url_policy is not None
        self._client_factory = client_factory or self._default_client_factory
        self._session_tool_loader = (
            session_tool_loader or self._default_session_tool_loader
        )
        self._session_stack: AsyncExitStack | None = None
        self._session_tools: dict[str, list[BaseTool]] = {}
        self._session_lock: asyncio.Lock | None = None

    async def __aenter__(self) -> Self:
        """Open one host-managed scope for stateful MCP tool calls."""

        if self._session_stack is not None:
            raise RuntimeError("MCP Provider 已在运行作用域中")
        stack = AsyncExitStack()
        await stack.__aenter__()
        self._session_stack = stack
        self._session_tools = {}
        self._session_lock = asyncio.Lock()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Close every MCP Session opened in the current scope."""

        stack = self._session_stack
        if stack is None:
            return False
        try:
            return await stack.__aexit__(exc_type, exc, traceback)
        finally:
            self._session_stack = None
            self._session_tools = {}
            self._session_lock = None

    async def get_tool(
        self,
        server_name: str,
        tool_name: str,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> BaseTool | None:
        tools = await self.get_tools(server_name, server_config, context)
        return next((item for item in tools if item.name == tool_name), None)

    async def get_tools(
        self,
        server_name: str,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> list[BaseTool]:
        """Resolve one MCP connection and discover all of its tools."""

        uses_server_ref = "server_ref" in server_config
        trusted_config = await self._resolve_server_config(server_config, context)
        self._validate_secret_references(trusted_config)
        try:
            resolved = await self._resolve_value(trusted_config, context)
            prepared = self._prepare_connection_config(resolved)
            if (
                not uses_server_ref
                and not self._has_explicit_url_policy
                and prepared.get("transport")
                in {"sse", "streamable_http", "streamable-http", "http"}
            ):
                raise ToolDefinitionError("内联 MCP Server 必须配置宿主 McpUrlPolicy")
            if not uses_server_ref and self._has_explicit_url_policy:
                await self._authorize_remote_connection(prepared, context)
        except (ToolDefinitionError, ToolUnavailableError):
            raise
        except Exception:  # noqa: BLE001 - sanitize environment/Secret failures
            raise ToolUnavailableError("MCP 凭据解析失败") from None

        try:
            if self._session_stack is not None:
                return await self._get_session_tools(server_name, prepared)
            client = self._client_factory({server_name: prepared})
            return await client.get_tools(server_name=server_name)
        except Exception:  # noqa: BLE001 - sanitize arbitrary MCP client failures
            raise ToolUnavailableError("MCP 工具发现失败") from None

    async def _get_session_tools(
        self,
        server_name: str,
        prepared: Mapping[str, Any],
    ) -> list[BaseTool]:
        cached = self._session_tools.get(server_name)
        if cached is not None:
            return cached
        lock = self._session_lock
        stack = self._session_stack
        if lock is None or stack is None:
            raise RuntimeError("MCP Provider 作用域已结束")
        async with lock:
            cached = self._session_tools.get(server_name)
            if cached is not None:
                return cached
            client = self._client_factory({server_name: prepared})
            session_factory = getattr(client, "session", None)
            if session_factory is None:
                raise RuntimeError("MCP Client 不支持显式 Session")
            candidate = AsyncExitStack()
            await candidate.__aenter__()
            try:
                session = await candidate.enter_async_context(
                    session_factory(server_name)
                )
                tools = await self._session_tool_loader(session)
            except BaseException:
                await candidate.aclose()
                raise
            stack.push_async_callback(candidate.aclose)
            self._session_tools[server_name] = tools
            return tools

    @staticmethod
    async def _default_session_tool_loader(session: Any) -> list[BaseTool]:
        try:
            from langchain_mcp_adapters.tools import load_mcp_tools
        except ImportError:
            raise ToolUnavailableError(
                "使用 MCP Tool 需要安装 langchain-skill-runtime[mcp]"
            ) from None
        return cast(list[BaseTool], await load_mcp_tools(session))

    async def _authorize_remote_connection(
        self,
        connection: dict[str, Any],
        context: CompileContext,
    ) -> None:
        transport = connection.get("transport")
        if transport not in {"sse", "streamable_http", "streamable-http", "http"}:
            return
        url = connection.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ToolDefinitionError("MCP Server 缺少 URL")
        addresses = await self._url_policy.validate(url, context)
        connection["httpx_client_factory"] = self._bound_http_client_factory(
            url,
            addresses,
        )

    @staticmethod
    def _bound_http_client_factory(
        approved_url: str,
        addresses: tuple[str, ...],
    ) -> Callable[..., Any]:
        parsed = urlsplit(approved_url)
        approved_host = parsed.hostname or ""
        approved_port = parsed.port or 443
        try:
            import httpx
        except ImportError:
            raise ToolUnavailableError(
                "使用远程 MCP Tool 需要安装 langchain-skill-runtime[mcp]"
            ) from None

        class BoundMcpTransport(httpx.AsyncBaseTransport):
            def __init__(self) -> None:
                self._transport = httpx.AsyncHTTPTransport()

            async def handle_async_request(self, request: Any) -> Any:
                request_host = request.url.host
                request_port = request.url.port or 443
                if request_host.casefold() != approved_host.casefold() or (
                    request_port != approved_port
                ):
                    raise ToolUnavailableError("MCP 请求目标发生变化")
                approved_address = addresses[0]
                target = request.url.copy_with(host=approved_address)
                headers = request.headers.copy()
                default_port = 443 if parsed.scheme == "https" else 80
                headers["host"] = (
                    approved_host
                    if approved_port == default_port
                    else f"{approved_host}:{approved_port}"
                )
                extensions = dict(request.extensions)
                extensions["sni_hostname"] = approved_host
                bound_request = httpx.Request(
                    request.method,
                    target,
                    headers=headers,
                    stream=request.stream,
                    extensions=extensions,
                )
                return await self._transport.handle_async_request(bound_request)

            async def aclose(self) -> None:
                await self._transport.aclose()

        def factory(
            headers: dict[str, str] | None = None,
            timeout: Any = None,
            auth: Any = None,
        ) -> Any:
            return httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                auth=auth,
                follow_redirects=False,
                transport=BoundMcpTransport(),
            )

        return factory

    async def _resolve_server_config(
        self,
        server_config: Mapping[str, Any],
        context: CompileContext,
    ) -> dict[str, Any]:
        if "server_ref" not in server_config:
            return dict(server_config)
        if set(server_config) != {"server_ref"}:
            raise ToolDefinitionError("MCP server_ref 配置不能包含其他字段")

        reference = server_config.get("server_ref")
        if not isinstance(reference, str) or not reference.strip():
            raise ToolDefinitionError("MCP server_ref 不能为空")
        if self._server_config_provider is None:
            raise ToolDefinitionError("MCP server_ref 缺少 McpServerConfigProvider")
        try:
            resolved = await self._server_config_provider.resolve(reference, context)
        except Exception:  # noqa: BLE001 - sanitize external provider failures
            raise ToolUnavailableError("MCP Server 配置解析失败") from None
        if not isinstance(resolved, Mapping):
            raise ToolDefinitionError("MCP Server 配置必须是对象")
        return dict(resolved)

    @classmethod
    def _validate_secret_references(cls, server_config: Mapping[str, Any]) -> None:
        url = server_config.get("url")
        if url is not None and not cls._is_reference(url):
            if not isinstance(url, str):
                raise ToolDefinitionError("MCP URL 配置非法")
            parsed = urlsplit(url)
            sensitive_query = any(
                cls._is_sensitive_name(name)
                for name, _ in parse_qsl(parsed.query, keep_blank_values=True)
            )
            sensitive_fragment = cls._is_sensitive_name(parsed.fragment)
            if (
                parsed.username
                or parsed.password
                or sensitive_query
                or sensitive_fragment
            ):
                raise ToolDefinitionError("MCP URL 不允许包含明文凭据")

        headers = server_config.get("headers")
        if headers is not None:
            if not isinstance(headers, Mapping):
                raise ToolDefinitionError("MCP headers 配置必须是对象")
            for name, value in headers.items():
                if cls._is_sensitive_name(name) and not cls._is_reference(value):
                    raise ToolDefinitionError(
                        "MCP Header 凭据不允许使用明文，必须配置 env 或 secret_ref"
                    )

        environment = server_config.get("env")
        if environment is not None:
            if not isinstance(environment, Mapping):
                raise ToolDefinitionError("MCP env 配置必须是对象")
            for name, value in environment.items():
                if cls._is_sensitive_name(name) and not cls._is_reference(value):
                    raise ToolDefinitionError(
                        "MCP 环境变量凭据不允许使用明文，必须配置 env 或 secret_ref"
                    )

        cls._validate_nested_sensitive_values(server_config)

    @classmethod
    def _validate_nested_sensitive_values(cls, value: Any) -> None:
        if cls._is_reference(value):
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if cls._is_sensitive_name(key) and not cls._is_reference(item):
                    raise ToolDefinitionError(
                        "MCP 敏感配置不允许使用明文，必须配置 env 或 secret_ref"
                    )
                cls._validate_nested_sensitive_values(item)
        elif isinstance(value, list):
            for item in value:
                cls._validate_nested_sensitive_values(item)

    @staticmethod
    def _is_secret_ref(value: Any) -> bool:
        return (
            isinstance(value, Mapping)
            and set(value) == {"secret_ref"}
            and isinstance(value.get("secret_ref"), str)
            and bool(value["secret_ref"].strip())
        )

    @staticmethod
    def _is_env_ref(value: Any) -> bool:
        return (
            isinstance(value, Mapping)
            and set(value) == {"env"}
            and isinstance(value.get("env"), str)
            and bool(value["env"].strip())
        )

    @classmethod
    def _is_reference(cls, value: Any) -> bool:
        return cls._is_secret_ref(value) or cls._is_env_ref(value)

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

    async def _resolve_value(self, value: Any, context: CompileContext) -> Any:
        if isinstance(value, Mapping):
            if set(value) == {"secret_ref"}:
                reference = value.get("secret_ref")
                if not isinstance(reference, str) or not reference.strip():
                    raise ToolDefinitionError("MCP secret_ref 不能为空")
                if self._secret_provider is None:
                    raise ToolDefinitionError("MCP Secret 引用缺少 SecretProvider")
                try:
                    return await self._secret_provider.resolve(reference, context)
                except Exception:  # noqa: BLE001 - sanitize external provider failures
                    raise ToolUnavailableError("MCP Secret 解析失败") from None
            if set(value) == {"env"}:
                variable = value.get("env")
                if (
                    not isinstance(variable, str)
                    or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable) is None
                ):
                    raise ToolDefinitionError("MCP env 引用非法")
                resolved = os.getenv(variable)
                if resolved is None or not resolved.strip():
                    raise ToolUnavailableError("MCP 环境变量未配置")
                return resolved
            return {
                str(key): await self._resolve_value(item, context)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [await self._resolve_value(item, context) for item in value]
        return value

    @staticmethod
    def _prepare_connection_config(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ToolDefinitionError("MCP server 配置必须是对象")
        prepared = dict(value)
        query = prepared.pop("query", None)
        if query is None:
            return prepared
        url = prepared.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ToolDefinitionError("MCP query 配置必须配合 URL 使用")
        if not isinstance(query, Mapping):
            raise ToolDefinitionError("MCP query 配置必须是对象")
        if not all(isinstance(name, str) for name in query):
            raise ToolDefinitionError("MCP query 参数名必须是字符串")
        parsed = urlsplit(url)
        pairs = list(parse_qsl(parsed.query, keep_blank_values=True))
        pairs.extend((str(name), str(item)) for name, item in query.items())
        prepared["url"] = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(pairs),
                parsed.fragment,
            )
        )
        return prepared

    @staticmethod
    def _default_client_factory(
        connections: dict[str, Mapping[str, Any]],
    ) -> _McpClient:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            raise ToolUnavailableError(
                "使用 MCP Tool 需要安装 langchain-skill-runtime[mcp]"
            ) from None
        return cast(_McpClient, MultiServerMCPClient(cast(Any, connections)))
