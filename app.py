"""入口：跑一次完整的差异表达分析（含人工确认 + 长期记忆）。

用法（在项目根目录执行）：
    python data/generate_data.py   # 先造数据
    python app.py                  # 再跑分析（质检后会暂停等你确认）
    python app.py --auto           # 自动确认，跳过人工确认（演示用）

启用 PostgreSQL 持久化（检查点 + 长期记忆，重启不丢）：
    在 .env 里设 USE_POSTGRES=true（连接信息见 .env.example）
"""

import sys

from langgraph.types import Command

from omics_agent.graph import build_graph


def _ask_confirm(info: dict) -> bool:
    """把质检结果给用户看，问是否继续。返回 True/False。"""
    print("\n" + "=" * 60)
    print("【人工确认】", info.get("message", ""))
    print("-" * 60)
    print(info.get("qc_report", ""))
    print("-" * 60)
    print(f"A 组：{info.get('group_a')}，B 组：{info.get('group_b')}")
    print("=" * 60)
    ans = input("是否继续分析？(y/n，回车默认 y): ").strip().lower()
    return ans in ("", "y", "yes")


def main():
    auto = "--auto" in sys.argv
    graph = build_graph()

    # thread_id 是"会话编号"，同一个编号会复用同一段检查点历史
    config = {"configurable": {"thread_id": "demo-1"}}

    initial = {
        "user_request": "帮我找两组样本的差异表达基因，并解读结果",
        "dataset_path": "data",
        "user_id": "researcher-1",  # 长期记忆按这个区分不同研究者
    }

    print("=" * 60)
    print("分析请求：", initial["user_request"])
    print("=" * 60)

    # 第一次调用：会跑到"人工确认"节点暂停
    result = graph.invoke(initial, config=config)

    # 处理人工确认（HITL）：只要还挂着中断，就一直处理
    while result.get("__interrupt__"):
        info = result["__interrupt__"][0].value
        approved = True if auto else _ask_confirm(info)
        result = graph.invoke(Command(resume={"approved": approved}), config=config)

    print("\n" + "=" * 60)
    print(result["final_report"])
    print("=" * 60)


if __name__ == "__main__":
    main()
