"""差异表达分析：纯 Python 统计计算（不依赖 LLM）。

核心思想：对每个基因，比较 A 组和 B 组的表达量，判断它是不是"显著差异"。
三个指标：
  1. log2FC（fold change，倍数变化）—— 表达量差了多少倍，取 log2 为了对称；
  2. p 值（t 检验）—— 差异是不是"碰巧"出现的概率；
  3. FDR（BH 校正后的 p 值）—— 一次比较成千上万个基因，
     必须做多重检验校正，否则假阳性一大堆。

关于显著性检验：优先用 scipy 的 Welch t 检验（更专业）；
如果没装 scipy，自动退化为正态近似（z 检验），保证项目开箱即跑。
"""

import math

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _p_value(a: np.ndarray, b: np.ndarray) -> float:
    """两组样本的差异显著性 p 值（双尾）。

    a、b 是两个样本数组。优先用 scipy 的 Welch t 检验；
    没装 scipy 时退化为正态近似（z 检验）。
    """
    if _HAS_SCIPY:
        return float(_stats.ttest_ind(b, a, equal_var=False).pvalue)

    # ── 正态近似 fallback ──
    n1, n2 = len(a), len(b)
    v1 = float(np.var(a, ddof=1))
    v2 = float(np.var(b, ddof=1))
    se = math.sqrt(v1 / n1 + v2 / n2)  # 标准误
    if se == 0:
        return 1.0  # 两组完全没有方差差异，无法判断，视为不显著
    z = (float(np.mean(b)) - float(np.mean(a))) / se
    # 双尾 p = 2 * (1 - Φ(|z|))，Φ 用误差函数 erf 算
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def differential_expression(
    expr_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    fc_threshold: float = 1.0,
    fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    """对每个基因做两组差异表达分析。

    参数:
        expr_df:     表达矩阵，第一列是基因名，其余列是样本
        metadata_df: 样本信息表，含 sample_id 和分组列
        group_col:   分组列名（如 "condition"）
        group_a:     A 组名
        group_b:     B 组名
        fc_threshold:  log2FC 阈值，默认 |log2FC| > 1（即差 2 倍）
        fdr_threshold: FDR 阈值，默认 < 0.05

    返回:
        DataFrame，每行一个基因，含 mean/log2fc/pvalue/fdr/significant
    """
    # 1. 从 metadata 取出两组各自的样本名
    meta = metadata_df.set_index("sample_id")
    samples_a = meta[meta[group_col] == group_a].index.tolist()
    samples_b = meta[meta[group_col] == group_b].index.tolist()

    # 2. 表达矩阵：第一列是基因名，设成 index，剩下全是样本
    expr = expr_df.copy()
    expr = expr.set_index(expr.columns[0])

    # 3. 逐个基因算
    rows = []
    for gene in expr.index:
        vals_a = expr.loc[gene, samples_a].astype(float).values
        vals_b = expr.loc[gene, samples_b].astype(float).values

        mean_a = float(np.mean(vals_a))
        mean_b = float(np.mean(vals_b))

        # log2 fold change（+1e-6 防 log0）
        log2fc = float(np.log2((mean_b + 1e-6) / (mean_a + 1e-6)))

        p_value = _p_value(vals_a, vals_b)

        rows.append({
            "gene": gene,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "log2fc": log2fc,
            "pvalue": p_value,
        })

    res = pd.DataFrame(rows)

    # 4. BH 多重检验校正：按 p 值从小到大排序，逐步收紧阈值
    res = res.sort_values("pvalue").reset_index(drop=True)
    n = len(res)
    res["fdr"] = res["pvalue"] * n / (np.arange(n) + 1)
    # BH 要求 FDR 单调不减，从后往前取最小值
    res["fdr"] = res["fdr"][::-1].cummin()[::-1]
    res["fdr"] = res["fdr"].clip(upper=1.0)

    # 5. 显著：FDR 够小 且 变化倍数够大
    res["significant"] = (res["fdr"] < fdr_threshold) & (res["log2fc"].abs() > fc_threshold)

    return res.sort_values("fdr").reset_index(drop=True)
