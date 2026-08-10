---
name: heterogeneous-tool-test
description: 验证一个技能同时使用 Function、脚本、客户端和 MCP 工具
allowed-tools:
  - add_numbers
  - generate_report
  - export_client_file
  - mcp_echo
metadata:
  category: integration-test
---

# 异构工具测试技能

根据用户任务选择适合的工具：计算使用 `add_numbers`，生成报告使用
`generate_report`，客户端导出使用 `export_client_file`，MCP 回显使用
`mcp_echo`。不得创建或调用未绑定的工具。
