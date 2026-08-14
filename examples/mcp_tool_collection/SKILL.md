---
id: remote-mcp-tool-collection
name: remote-mcp-tool-collection
description: 从一个标准 MCP Server 动态发现全部工具
version: 1.0.0
tools:
  - id: example-mcp-server
    name: example_mcp_server
    description: 示例 MCP Server 提供的全部工具
    type: MCP
    input_schema:
      type: object
      properties: {}
      additionalProperties: false
    execution:
      server_name: example
      server:
        transport: streamable_http
        url: https://mcp.example.com/mcp
        headers:
          Authorization:
            env: EXAMPLE_MCP_AUTHORIZATION
---

# MCP 工具集

根据用户任务，从 MCP Server 通过 `tools/list` 返回的全部工具中选择合适工具。
