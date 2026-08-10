from mcp.server.fastmcp import FastMCP

server = FastMCP("langchain-skill-runtime-test")


@server.tool()
async def mcp_echo(text: str) -> str:
    """Echo text for local integration testing."""

    return f"mcp:{text}"


if __name__ == "__main__":
    server.run(transport="stdio")
