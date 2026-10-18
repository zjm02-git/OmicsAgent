# -*- coding: utf-8 -*-
"""冒烟测试：内存版完整流程（Human-in-the-loop + 长期记忆复用）。

用法：在项目根目录运行
    python tests/test_pipeline.py

验证点：
1. 第一次跑，阈值用默认（prefs_from_memory=False）；
2. 第二次跑，能从长期记忆读到上次的阈值（prefs_from_memory=True）。
"""
import os
import sys

# 把项目根目录加入 sys.path（tests 的上一级），不依赖硬编码路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["USE_POSTGRES"] = "false"

from langgraph.types import Command
from omics_agent.graph import build_graph


def run(graph, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "user_request": "帮我找两组样本的差异表达基因",
        "dataset_path": "data",
        "user_id": "researcher-1",
    }
    result = graph.invoke(initial, config=config)
    # 处理人工确认：只要还挂着 interrupt，就一直批准继续
    while result.get("__interrupt__"):
        result = graph.invoke(Command(resume={"approved": True}), config=config)
    return result


if __name__ == "__main__":
    g = build_graph()
    r1 = run(g, "t1")
    print("第1次  prefs_from_memory =", r1.get("prefs_from_memory"), "| fc =", r1.get("fc_threshold"))
    r2 = run(g, "t2")
    print("第2次  prefs_from_memory =", r2.get("prefs_from_memory"), "| fc =", r2.get("fc_threshold"))

    assert r1.get("prefs_from_memory") is False, "第一次不该有记忆"
    assert r2.get("prefs_from_memory") is True, "第二次应该复用记忆"
    assert r2.get("fc_threshold") == 1.0, "复用到的阈值应该是 1.0"
    print("长期记忆复用 OK")
