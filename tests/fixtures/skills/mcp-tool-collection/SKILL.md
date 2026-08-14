---
id: mcp-tool-collection
name: mcp-tool-collection
description: 验证一个 MCP 工具对象展开为 MCP Server 的全部工具
version: 1.0.0
tools:
  - id: maps-server
    name: maps_server
    description: 地图 MCP Server 提供的全部工具
    type: MCP
    input_schema:
      type: object
      properties: {}
      additionalProperties: false
    execution:
      server_name: maps
      server:
        transport: streamable_http
        url: https://maps.example.invalid/mcp
        query:
          key:
            env: TEST_MAPS_MCP_KEY
---

# MCP 工具集合测试技能

根据用户需求，从地图 MCP Server 动态发现的全部工具中选择合适工具。
