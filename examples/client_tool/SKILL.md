---
id: client-file-export
name: client-file-export
description: 将文件导出委派给当前在线客户端
version: 1.0.0
tools:
  - id: export-file
    name: export_client_file
    description: 请求客户端导出文件
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
---

# 客户端文件导出

用户要求导出文件时，调用 `export_client_file`。
