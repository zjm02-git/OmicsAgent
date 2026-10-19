"""把节点拼成一张图，加上条件路由、检查点、长期记忆。

图的结构：
  START → parse_intent
              ├─ analysis_type=other → report（暂不支持）
              └─ analysis_type=differential → load_qc
                        ├─ 质检不合格 → report（质检失败）
                        └─ 质检合格 → human_review（人工确认）
                                    ├─ 用户取消 → report
                                    └─ 用户批准 → diff_analysis → enrichment
                                                  → interpret → save_memory → report → END

持久化：默认内存（InMemorySaver + InMemoryStore），设 USE_POSTGRES=true 切 PostgreSQL。
"""

from langgraph.graph import END, START, StateGraph

from omics_agent import config, nodes
from omics_agent.state import AnalysisState


def route_after_intent(state: dict) -> str:
    """① 之后的分叉：只支持差异分析，其他直接出报告。"""
    return "load_qc" if state.get("analysis_type") == "differential" else "report"


def route_after_qc(state: dict) -> str:
    """② 质检之后的分叉：合格先进人工确认，不合格直接出报告。"""
    return "human_review" if state.get("qc_passed", False) else "report"


def route_after_review(state: dict) -> str:
    """③ 人工确认之后的分叉：批准继续分析，取消直接出报告。"""
    return "diff_analysis" if state.get("qc_passed", False) else "report"


def build_graph():
    builder = StateGraph(AnalysisState)

    # 加节点
    builder.add_node("parse_intent", nodes.parse_intent_node)
    builder.add_node("load_qc", nodes.load_qc_node)
    builder.add_node("human_review", nodes.human_review_node)
    builder.add_node("diff_analysis", nodes.diff_analysis_node)
    builder.add_node("enrichment", nodes.enrichment_node)
    builder.add_node("interpret", nodes.interpret_node)
    builder.add_node("save_memory", nodes.save_memory_node)
    builder.add_node("report", nodes.report_node)

    # 加边
    builder.add_edge(START, "parse_intent")

    # 条件边 1：按分析类型路由
    # add_conditional_edges 三参数：源节点 / 路由函数（返回字符串）/ 路由返回值→目标节点的映射
    builder.add_conditional_edges(
        "parse_intent", route_after_intent,
        {"load_qc": "load_qc", "report": "report"},
    )

    # 条件边 2：质检合格 → 人工确认，否则 → 报告
    builder.add_conditional_edges(
        "load_qc", route_after_qc,
        {"human_review": "human_review", "report": "report"},
    )

    # 条件边 3：人工确认批准 → 分析，取消 → 报告
    builder.add_conditional_edges(
        "human_review", route_after_review,
        {"diff_analysis": "diff_analysis", "report": "report"},
    )

    builder.add_edge("diff_analysis", "enrichment")
    builder.add_edge("enrichment", "interpret")
    builder.add_edge("interpret", "save_memory")
    builder.add_edge("save_memory", "report")
    builder.add_edge("report", END)

    # 检查点 + 长期记忆：默认内存，设 USE_POSTGRES=true 切到 PostgreSQL
    return builder.compile(
        checkpointer=config.get_checkpointer(),
        store=config.get_store(),
    )
