"""MCP 子包结构与公开导出兼容性。"""

from langchain_skill_runtime.adapters import (
    AllowHostsMcpUrlPolicy,
    LangChainMcpToolProvider,
    McpToolAdapter,
)
from langchain_skill_runtime.adapters.mcp.adapter import (
    McpToolAdapter as PackageMcpToolAdapter,
)
from langchain_skill_runtime.adapters.mcp.provider import (
    LangChainMcpToolProvider as PackageLangChainMcpToolProvider,
)
from langchain_skill_runtime.adapters.mcp.url_policy import (
    AllowHostsMcpUrlPolicy as PackageAllowHostsMcpUrlPolicy,
)


def test_mcp_submodules_preserve_public_adapter_exports() -> None:
    assert PackageMcpToolAdapter is McpToolAdapter
    assert PackageLangChainMcpToolProvider is LangChainMcpToolProvider
    assert PackageAllowHostsMcpUrlPolicy is AllowHostsMcpUrlPolicy
