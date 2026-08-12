"""测试夹具：真实项目中由业务 Script Executor 运行该入口。"""


def generate_report(title: str) -> dict[str, str]:
    return {"status": "generated", "title": title}
