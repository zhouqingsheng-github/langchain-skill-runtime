---
id: calculator
name: calculator
description: 通过宿主 Python Function 计算两个整数之和
version: 1.0.0
tools:
  - id: add-numbers
    name: add_numbers
    description: 计算两个整数之和
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
---

# 计算器

用户要求计算两个整数之和时，调用 `add_numbers`。
