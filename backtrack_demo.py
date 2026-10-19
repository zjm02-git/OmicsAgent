# -*- coding: utf-8 -*-
"""回溯（时间旅行）功能演示。

Checkpointer 不只存"最后一步"，而是**每个节点跑完都存一份快照**。
利用 get_state_history 可以回到任意历史检查点，换参数重跑，不用从头来。

用法（在项目根目录执行）：
    python backtrack_demo.py

演示内容：
    1. 第一次完整跑，得到一批显著差异基因（fc 阈值 1.0）；
    2. 用 get_state_history 列出所有历史检查点；
    3. 回到"质检之后"那个检查点，把 fc_threshold 改成 2.0；
    4. 从这个检查点继续重跑，显著基因变少（阈值更严，证明没从头跑）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["USE_POSTGRES"] = "false"

from langgraph.types import Command
from omics_agent.graph import build_graph


def _describe(values: dict) -> str:
    """根据 State 里已经有哪几步的字段，判断这个检查点跑到哪一步了。"""
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


def _run(graph, config, initial=None):
    """从某个状态跑到出报告；遇到人工确认（interrupt）就自动批准。"""
    result = graph.invoke(initial, config=config)  # initial=None 时 = 从该检查点继续
    while result.get("__interrupt__"):
        result = graph.invoke(Command(resume={"approved": True}), config=config)
    return result


if __name__ == "__main__":
    g = build_graph()
    config = {"configurable": {"thread_id": "backtrack-demo"}}

    # ── 1. 第一次完整跑 ──
    initial = {
        "user_request": "帮我找两组样本的差异表达基因，并解读",
        "dataset_path": "data",
        "user_id": "researcher-1",
    }
    r1 = _run(g, config, initial)
    print(f"第一次跑完：显著基因 {r1['significant_count']} 个，fc 阈值 {r1['fc_threshold']}")

    # ── 2. 列出所有历史检查点（从新到旧）──
    history = list(g.get_state_history(config))
    print(f"\n共 {len(history)} 个检查点（每个节点跑完自动存档）：")
    for i, snap in enumerate(history):
        desc = _describe(snap.values)
        cid = snap.config["configurable"]["checkpoint_id"][:8]
        print(f"  [{i}] {desc}  checkpoint_id={cid}…")

    # ── 3. 回溯到"质检之后"，把阈值改严再重跑 ──
    target = next((s for s in history if _describe(s.values) == "② 质检之后"), None)
    assert target is not None, "没找到质检之后的检查点"

    print(f"\n回溯到「② 质检之后」，把 fc_threshold 从 1.0 改成 2.0，重跑……")
    new_config = g.update_state(target.config, {"fc_threshold": 2.0})
    r2 = _run(g, new_config)  # 从更新后的检查点继续，不传 initial

    print(f"回溯重跑：显著基因 {r2['significant_count']} 个，fc 阈值 {r2['fc_threshold']}")
    print("\n结论：没从头跑，直接从历史检查点换了参数继续；阈值更严（2 倍→4 倍）→ 显著基因大幅减少。")
