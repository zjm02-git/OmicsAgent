"""8 个节点函数：每个节点只干一件事，从 State 读、往 State 写。

分工原则：
  - LLM 干"懂"的活（①理解需求、⑥解读结果）
  - Python 干"算"的活（②质检、④差异统计、⑤通路富集、⑧拼报告）
  - 人工介入：③人工确认；记忆：②读偏好、⑦写偏好
"""

import os
from typing import Literal

import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from omics_agent.config import get_model
from omics_agent.diff import differential_expression
from omics_agent.enrichment import enrichment_analysis

# 默认数据目录、分组列名
DATASET_PATH = "data"
GROUP_COL = "condition"

# 长期记忆的命名空间和 key（记忆按"研究者"区分，存各自的分析偏好）
MEMORY_NS = ("omics",)
PREFS_KEY = "preferences"


# ══════════ ① 需求解析 ══════════
class IntentOutput(BaseModel):
    """让 LLM 输出"确定的值"，而不是自由文本，路由才可靠。"""

    analysis_type: Literal["differential", "other"] = Field(
        description="要做的分析类型。differential=差异表达分析，other=其他/不支持"
    )


def parse_intent_node(state: dict) -> dict:
    """结构化输出：判断用户想做什么分析。"""
    model = get_model().with_structured_output(IntentOutput)
    res = model.invoke([HumanMessage(state["user_request"])])
    return {"analysis_type": res.analysis_type}


# ══════════ ② 加载 + 质检 ══════════
def load_qc_node(state: dict, runtime: Runtime) -> dict:
    """读数据 + 质检，质检通过后从长期记忆读这个用户上次的分析偏好。

    runtime 是 LangGraph 自动注入的：节点函数签名里声明了 Runtime 参数，
    就能拿到它来读写长期记忆（runtime.store），不需要自己手动传。
    """
    dataset_path = state.get("dataset_path", DATASET_PATH)
    expr_file = os.path.join(dataset_path, "expression.csv")
    meta_file = os.path.join(dataset_path, "metadata.csv")

    # 1. 文件在不在
    try:
        expr_df = pd.read_csv(expr_file)
        metadata_df = pd.read_csv(meta_file)
    except FileNotFoundError as e:
        return {"qc_passed": False, "qc_report": f"数据文件缺失：{e}"}

    # 2. 分组列在不在
    if GROUP_COL not in metadata_df.columns:
        return {"qc_passed": False, "qc_report": f"metadata.csv 缺少分组列 {GROUP_COL}"}

    # 3. 必须恰好两组（自动识别 A/B 两组名字）
    groups = metadata_df[GROUP_COL].unique().tolist()
    if len(groups) != 2:
        return {"qc_passed": False, "qc_report": f"需要恰好 2 组，实际 {len(groups)} 组"}
    group_a, group_b = groups[0], groups[1]

    # 4. 每组样本数不能太少（t 检验至少要有几个样本才有意义）
    n_a = int((metadata_df[GROUP_COL] == group_a).sum())
    n_b = int((metadata_df[GROUP_COL] == group_b).sum())
    if n_a < 2 or n_b < 2:
        return {"qc_passed": False, "qc_report": f"每组至少 2 个样本，实际 A={n_a} / B={n_b}"}

    # 5. 缺失值不能太多
    na_ratio = float(expr_df.isna().mean().mean())
    if na_ratio > 0.1:
        return {"qc_passed": False, "qc_report": f"缺失值过多：{na_ratio:.1%}"}

    qc_report = (
        f"质检通过：{len(expr_df)} 个基因，"
        f"A 组({group_a}) {n_a} 样本，B 组({group_b}) {n_b} 样本，缺失值 {na_ratio:.1%}"
    )

    # 6. 读长期记忆：这个研究者上次做过分析，就复用他上次的阈值偏好
    user_id = state.get("user_id", "default")
    fc_threshold, fdr_threshold = 1.0, 0.05
    prefs_from_memory = False
    try:
        item = runtime.store.get((*MEMORY_NS, user_id), PREFS_KEY)
        if item and item.value:
            fc_threshold = float(item.value.get("fc_threshold", 1.0))
            fdr_threshold = float(item.value.get("fdr_threshold", 0.05))
            prefs_from_memory = True
    except Exception:
        pass  # 记忆读不到就用默认阈值，不影响主流程

    return {
        "qc_passed": True,
        "qc_report": qc_report,
        "group_a": group_a,
        "group_b": group_b,
        "fc_threshold": fc_threshold,
        "fdr_threshold": fdr_threshold,
        "prefs_from_memory": prefs_from_memory,
    }


# ══════════ ③ 人工确认（HITL） ══════════
def human_review_node(state: dict) -> dict:
    """质检通过后暂停，把结果给人看，等人确认后才继续分析。"""
    # interrupt() 会让图在这里暂停；返回值 = 外部 resume 回来的值
    decision = interrupt({
        "message": "质检通过，是否继续差异分析？",
        "qc_report": state.get("qc_report", ""),
        "group_a": state.get("group_a"),
        "group_b": state.get("group_b"),
    })
    if isinstance(decision, dict) and decision.get("approved") is False:
        return {"qc_passed": False, "qc_report": "已由用户取消分析"}
    return {}


# ══════════ ④ 差异表达分析 ══════════
def diff_analysis_node(state: dict) -> dict:
    """调 diff.py 做真实统计，把结果整理成摘要文本。"""
    dataset_path = state.get("dataset_path", DATASET_PATH)
    expr_df = pd.read_csv(os.path.join(dataset_path, "expression.csv"))
    metadata_df = pd.read_csv(os.path.join(dataset_path, "metadata.csv"))

    res = differential_expression(
        expr_df, metadata_df, GROUP_COL,
        state["group_a"], state["group_b"],
        fc_threshold=state.get("fc_threshold", 1.0),
        fdr_threshold=state.get("fdr_threshold", 0.05),
    )

    sig = res[res["significant"]]
    significant_count = int(len(sig))
    sig_genes_up = sig[sig["log2fc"] > 0]["gene"].tolist()
    sig_genes_down = sig[sig["log2fc"] < 0]["gene"].tolist()

    # 取 top 10 显著基因，做成 markdown 表格
    top = sig.head(10)
    if top.empty:
        diff_summary = "未找到显著差异表达基因。"
    else:
        lines = [
            "| 基因 | log2FC | FDR | 变化方向 |",
            "| --- | --- | --- | --- |",
        ]
        for _, row in top.iterrows():
            direction = "上调" if row["log2fc"] > 0 else "下调"
            lines.append(
                f"| {row['gene']} | {row['log2fc']:.2f} | {row['fdr']:.2e} | {direction} |"
            )
        diff_summary = "\n".join(lines)

    return {
        "diff_summary": diff_summary,
        "significant_count": significant_count,
        "sig_genes_up": sig_genes_up,
        "sig_genes_down": sig_genes_down,
    }


# ══════════ ⑤ 通路富集分析 ══════════
def enrichment_node(state: dict) -> dict:
    """对显著基因做 GO 通路富集，判断它们是不是扎堆在某些通路上。"""
    dataset_path = state.get("dataset_path", DATASET_PATH)
    ann_file = os.path.join(dataset_path, "gene_ontology.csv")

    # 注释文件缺失时优雅降级，不阻断整个流程
    if not os.path.exists(ann_file):
        return {"enrichment_summary": "（缺少 gene_ontology.csv，跳过富集分析）"}

    ann_df = pd.read_csv(ann_file)
    expr_df = pd.read_csv(os.path.join(dataset_path, "expression.csv"))
    background = expr_df.iloc[:, 0].tolist()  # 所有基因 = 背景集

    up = state.get("sig_genes_up", [])
    down = state.get("sig_genes_down", [])

    parts = []
    for label, genes in [("上调基因", up), ("下调基因", down)]:
        if not genes:
            continue
        res = enrichment_analysis(genes, ann_df, background)
        if res.empty:
            continue
        sig_terms = res[res["significant"]]
        if sig_terms.empty:
            parts.append(f"**{label}**：未发现显著富集的通路。")
            continue
        lines = [f"**{label}** 显著富集的通路：", ""]
        for _, row in sig_terms.head(5).iterrows():
            lines.append(
                f"- {row['term']}（显著基因 {row['sig_count']}/{row['bg_count']}，"
                f"FDR={row['fdr']:.2e}）"
            )
        parts.append("\n".join(lines))

    summary = "\n\n".join(parts) if parts else "未发现显著富集的通路。"
    return {"enrichment_summary": summary}


# ══════════ ⑥ LLM 解读 ══════════
def interpret_node(state: dict) -> dict:
    """LLM 把一堆基因数字翻译成人话。"""
    model = get_model()
    prompt = (
        "你是一名生物信息学专家。下面是差异表达分析和通路富集分析的结果，"
        "请用通俗的中文解读：哪些基因显著上/下调、显著富集在哪些通路，"
        "这些通路可能提示什么生物学意义。\n\n"
        "【差异表达】\n"
        f"{state['diff_summary']}\n\n"
        f"（共 {state['significant_count']} 个显著差异基因）\n\n"
        "【通路富集】\n"
        f"{state.get('enrichment_summary', '无')}"
    )
    res = model.invoke([HumanMessage(prompt)])
    return {"interpretation": res.content}


# ══════════ ⑦ 写长期记忆 ══════════
def save_memory_node(state: dict, runtime: Runtime) -> dict:
    """把这次分析用的参数写进长期记忆，下次自动复用。"""
    user_id = state.get("user_id", "default")
    try:
        runtime.store.put(
            (*MEMORY_NS, user_id),
            PREFS_KEY,
            {
                "fc_threshold": state.get("fc_threshold", 1.0),
                "fdr_threshold": state.get("fdr_threshold", 0.05),
                "group_a": state.get("group_a", ""),
                "group_b": state.get("group_b", ""),
            },
        )
    except Exception:
        pass  # 记忆写失败不影响主流程
    return {}


# ══════════ ⑧ 汇总报告 ══════════
def report_node(state: dict) -> dict:
    """把所有结果拼成一份最终报告。要能处理三种情况。"""
    # 情况1：不支持的分析类型（没进质检，直接到这）
    if state.get("analysis_type") == "other":
        return {
            "final_report": "# 分析报告\n\n暂不支持该分析类型，当前只支持差异表达分析（differential）。"
        }

    # 情况2：质检失败（或人工确认时取消）
    if not state.get("qc_passed", False):
        return {
            "final_report": "# 分析报告\n\n## 质检未通过\n\n" + state.get("qc_report", "")
        }

    # 情况3：正常完整报告
    param_src = "来自长期记忆" if state.get("prefs_from_memory") else "默认"
    lines = [
        "# 多组学数据分析报告",
        "",
        "## 一、质检结果",
        state.get("qc_report", ""),
        "",
        "## 二、差异表达分析",
        f"显著差异基因数：{state.get('significant_count', 0)}",
        f"参数：|log2FC| > {state.get('fc_threshold', 1.0)}，"
        f"FDR < {state.get('fdr_threshold', 0.05)}（{param_src}）",
        "",
        state.get("diff_summary", ""),
        "",
        "## 三、通路富集分析",
        state.get("enrichment_summary", "无显著富集通路"),
        "",
        "## 四、结果解读",
        state.get("interpretation", ""),
    ]
    return {"final_report": "\n".join(lines)}
