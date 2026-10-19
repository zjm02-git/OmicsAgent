"""入口：跑一次完整的差异表达分析（含人工确认 + 长期记忆 + 回溯重跑）。

用法（在项目根目录执行）：
    python data/generate_data.py   # 先造数据（模拟数据）
    python app.py                  # 跑分析（质检后暂停等你确认，跑完可回溯改参重跑）
    python app.py --auto           # 自动确认，跳过人工确认（演示用，也跳过回溯提问）

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


def _describe(values: dict) -> str:
    """根据 State 里已经填了哪些字段，判断这个检查点跑到哪一步。"""
    if values.get("final_report"):
        return "⑧ 报告（跑完）"
    if values.get("interpretation"):
        return "⑥ 解读之后"
    if values.get("enrichment_summary"):
        return "⑤ 富集之后"
    if values.get("diff_summary"):
        return "④ 差异分析之后"
    if values.get("qc_passed") is not None:
        return "② 质检之后"
    if values.get("analysis_type"):
        return "① 需求解析之后"
    return "起点"


def _run_to_end(graph, config: dict, auto: bool, initial=None):
    """从初始状态（或某个历史检查点）跑到出报告；遇到人工确认就处理。

    initial=None 时表示不从起点跑，而是从 config 指定的检查点继续（回溯用）。
    """
    result = graph.invoke(initial, config=config)
    # __interrupt__ 是挂起时的中断信息；[0].value 就是 human_review 里
    # interrupt() 抛出的那个 dict（含 message / qc_report / group_a / group_b）
    while result.get("__interrupt__"):
        info = result["__interrupt__"][0].value
        approved = True if auto else _ask_confirm(info)
        result = graph.invoke(Command(resume={"approved": approved}), config=config)
    return result


def _backtrack(graph, config: dict, auto: bool):
    """时间旅行：列出历史检查点，回到某一步换参数重跑，不从头来。"""
    history = list(graph.get_state_history(config))
    print("\n历史检查点（从新到旧，每个节点跑完自动存档）：")
    for i, snap in enumerate(history):
        desc = _describe(snap.values)
        cid = snap.config["configurable"]["checkpoint_id"][:8]
        print(f"  [{i}] {desc}  checkpoint_id={cid}…")

    # 默认回到"质检之后"：数据读取和质检不重算，只重跑差异分析及以后
    default_idx = next(
        (i for i, s in enumerate(history) if _describe(s.values) == "② 质检之后"), 0
    )
    ans = input(f"\n回到哪个检查点？（默认 {default_idx}=质检之后）: ").strip()
    idx = int(ans) if ans.isdigit() else default_idx
    target = history[idx]

    fc = input("新的 fc_threshold（|log2FC| 阈值，回车保持默认 1.0）: ").strip()
    fdr = input("新的 fdr_threshold（FDR 阈值，回车保持默认 0.05）: ").strip()
    update = {}
    if fc:
        update["fc_threshold"] = float(fc)
    if fdr:
        update["fdr_threshold"] = float(fdr)

    # update_state：把新参数写进那个历史检查点，得到"从那里继续"的新配置
    new_config = graph.update_state(target.config, update)
    print(f"\n回溯到「{_describe(target.values)}」，改参数 {update or '无'}，重跑……")
    return _run_to_end(graph, new_config, auto)


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

    result = _run_to_end(graph, config, auto, initial)

    print("\n" + "=" * 60)
    print(result["final_report"])
    print("=" * 60)

    # 回溯重跑：跑完不满意（比如阈值太松/太严），回到历史检查点换参数重跑
    if not auto:
        while True:
            ans = input("\n是否回溯改参数重跑？(y/n，回车默认 n): ").strip().lower()
            if ans not in ("y", "yes"):
                break
            result = _backtrack(graph, config, auto)
            print("\n" + "=" * 60)
            print(result["final_report"])
            print("=" * 60)


if __name__ == "__main__":
    main()
