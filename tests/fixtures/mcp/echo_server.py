from mcp.server.fastmcp import FastMCP

server = FastMCP("langchain-skill-runtime-test")


@server.tool()
async def mcp_echo(text: str) -> str:
    """Echo text for local integration testing."""

    return f"mcp:{text}"


@server.tool()
async def mcp_text_length(text: str) -> int:
    """Return the number of characters in text for collection discovery testing."""

    return len(text)


if __name__ == "__main__":
    server.run(transport="stdio")
