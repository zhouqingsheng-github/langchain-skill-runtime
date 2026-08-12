---
id: heterogeneous-skill
name: heterogeneous-tool-test
description: 验证一个技能同时使用 Function、脚本、客户端和 MCP 工具
version: 1.0.0
allowed-tools:
  - add_numbers
  - generate_report
  - export_client_file
  - mcp_echo
tools:
  - id: function-add
    name: add_numbers
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
      registry_key: math.add

  - id: script-report
    name: generate_report
    description: 生成测试报告
    type: SERVER_SCRIPT
    timeout_seconds: 15
    input_schema:
      type: object
      properties:
        title: {type: string}
      required: [title]
      additionalProperties: false
    execution:
      entry: scripts/generate_report.py

  - id: client-export
    name: export_client_file
    description: 在客户端导出测试文件
    type: CLIENT_JAVASCRIPT
    timeout_seconds: 10
    input_schema:
      type: object
      properties:
        format: {type: string, enum: [xlsx]}
      required: [format]
      additionalProperties: false
    execution:
      tool_key: client.export.file

  - id: mcp-echo
    name: mcp_echo
    description: 通过 MCP 回显文本
    type: MCP
    input_schema:
      type: object
      properties:
        text: {type: string}
      required: [text]
      additionalProperties: false
    execution:
      server_name: echo
      tool_name: mcp_echo
      server_ref: demo.echo
metadata:
  category: integration-test
---

# 异构工具测试技能

根据用户任务选择适合的工具：计算使用 `add_numbers`，生成报告使用
`generate_report`，客户端导出使用 `export_client_file`，MCP 回显使用
`mcp_echo`。不得创建或调用未绑定的工具。
