"""URL authorization policies for remote MCP connections."""

import asyncio
import socket
from collections.abc import Mapping
from ipaddress import ip_address
from urllib.parse import urlsplit

from langchain_skill_runtime.adapters.mcp.protocols import AddressResolver
from langchain_skill_runtime.errors import ToolDefinitionError, ToolUnavailableError
from langchain_skill_runtime.models.context import CompileContext


class PublicHttpsMcpUrlPolicy:
    """Allow HTTPS URLs only when every resolved address is public."""

    def __init__(self, resolver: AddressResolver | None = None) -> None:
        self._resolver = resolver or self._resolve_addresses

    async def validate(
        self,
        url: str,
        context: CompileContext,
    ) -> tuple[str, ...]:
        del context
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ToolDefinitionError("MCP URL 必须使用公网 HTTPS 地址")
        try:
            port = parsed.port or 443
        except ValueError:
            raise ToolDefinitionError("MCP URL 端口非法") from None

        try:
            literal = ip_address(parsed.hostname)
        except ValueError:
            literal = None
        addresses: tuple[str, ...]
        if literal is not None:
            addresses = (str(literal),)
        else:
            try:
                addresses = await self._resolver(parsed.hostname, port)
            except Exception:  # noqa: BLE001 - sanitize DNS failures
                raise ToolUnavailableError("MCP Server 地址解析失败") from None
        if not addresses:
            raise ToolUnavailableError("MCP Server 地址解析失败")
        if any(not self._is_public_unicast(address) for address in addresses):
            raise ToolDefinitionError("MCP URL 必须解析到公网地址")
        return addresses

    @staticmethod
    def _is_public_unicast(value: str) -> bool:
        address = ip_address(value)
        mapped = getattr(address, "ipv4_mapped", None)
        if mapped is not None:
            address = mapped
        return (
            address.is_global
            and not address.is_multicast
            and not address.is_reserved
            and not address.is_unspecified
        )

    @staticmethod
    async def _resolve_addresses(host: str, port: int) -> tuple[str, ...]:
        loop = asyncio.get_running_loop()
        resolved = await loop.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
        )
        return tuple(dict.fromkeys(str(item[4][0]) for item in resolved))


class AllowHostsMcpUrlPolicy:
    """Allow only host names explicitly approved by the embedding application."""

    def __init__(
        self,
        hosts: Mapping[str, int | None] | set[str] | frozenset[str],
        resolver: AddressResolver | None = None,
    ) -> None:
        if isinstance(hosts, Mapping):
            self._hosts = {str(host).casefold(): port for host, port in hosts.items()}
        else:
            self._hosts = {str(host).casefold(): None for host in hosts}
        self._resolver = resolver or PublicHttpsMcpUrlPolicy._resolve_addresses

    async def validate(
        self,
        url: str,
        context: CompileContext,
    ) -> tuple[str, ...]:
        del context
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ToolDefinitionError("MCP URL 必须使用 HTTPS 地址")
        host = parsed.hostname.casefold() if parsed.hostname else ""
        if host not in self._hosts:
            raise ToolDefinitionError("MCP URL 未经宿主允许")
        configured_port = self._hosts[host]
        try:
            actual_port = parsed.port or 443
        except ValueError:
            raise ToolDefinitionError("MCP URL 端口非法") from None
        if configured_port is not None and actual_port != configured_port:
            raise ToolDefinitionError("MCP URL 端口未经宿主允许")

        try:
            literal = ip_address(parsed.hostname)
        except ValueError:
            literal = None
        addresses: tuple[str, ...]
        if literal is not None:
            addresses = (str(literal),)
        else:
            try:
                addresses = await self._resolver(parsed.hostname, actual_port)
            except Exception:  # noqa: BLE001 - sanitize DNS failures
                raise ToolUnavailableError("MCP Server 地址解析失败") from None
        if not addresses:
            raise ToolUnavailableError("MCP Server 地址解析失败")
        return addresses
