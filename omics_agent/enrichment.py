"""通路富集分析（GO/通路）：判断显著基因是不是"扎堆"在某些通路里。

核心原理：超几何检验（hypergeometric test），生活比喻是"抽球"：
  - 袋子里有 N 个球（所有基因），其中 K 个是红球（属于某通路的基因）。
  - 你随手抽出 n 个球（显著基因），里面恰好有 k 个红球。
  - k 比"瞎抽"该有的数量多得多 → 显著基因偏爱这个通路 → 富集。

和差异分析一样，一次要查很多通路，所以也要做 BH 多重检验校正，
否则通路一多，假阳性也跟着多。
"""

import math

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _hypergeom_upper_pvalue(N: int, K: int, n: int, k: int) -> float:
    """超几何检验的上尾 p 值 P(X >= k)：至少抽到 k 个红球的概率。

    参数:
        N  总球数（背景基因总数）
        K  红球数（该通路在背景里的基因数）
        n  抽球数（显著基因数）
        k  抽到的红球数（显著基因里属于该通路的数）

    优先用 scipy；没装 scipy 时退化为纯 Python 对数递推实现（规模小足够精确）。
    """
    if k <= 0:
        return 1.0
    if _HAS_SCIPY:
        # sf(k-1) = P(X > k-1) = P(X >= k)
        return float(_stats.hypergeom.sf(k - 1, N, K, n))

    # ── 纯 Python fallback：对数空间算超几何 PMF，避免大组合数溢出 ──
    def _log_comb(a: int, b: int) -> float:
        """log(C(a, b))，用循环乘除算，避免先算巨大阶乘。"""
        if b < 0 or b > a:
            return -math.inf
        b = min(b, a - b)
        res = 0.0
        for i in range(b):
            res += math.log(a - i) - math.log(i + 1)
        return res

    max_i = min(K, n)
    if k > max_i:
        return 0.0  # 理论上不可能抽到这么多红球
    log_denom = _log_comb(N, n)
    total = 0.0
    for i in range(k, max_i + 1):
        # P(X=i) = C(K,i) * C(N-K,n-i) / C(N,n)
        log_pmf = _log_comb(K, i) + _log_comb(N - K, n - i) - log_denom
        total += math.exp(log_pmf)
    return min(total, 1.0)


def enrichment_analysis(
    sig_genes: list,
    annotation_df: pd.DataFrame,
    background_genes: list,
    fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    """对一组显著基因做通路富集，返回每个通路的检验结果。

    参数:
        sig_genes:        显著基因名列表
        annotation_df:    注释表，至少两列 gene / term（一个基因可属于多个通路）
        background_genes: 背景基因全集（通常 = 表达矩阵里的所有基因）
        fdr_threshold:    FDR 阈值，默认 < 0.05

    返回:
        DataFrame，每行一个通路：term / bg_count / sig_count /
        pvalue / fdr / significant / enriched_genes
    """
    background_set = set(background_genes)
    sig_set = set(sig_genes) & background_set

    N = len(background_set)
    n = len(sig_set)

    # 只看注释表里、且在背景里的那些基因
    ann = annotation_df[annotation_df["gene"].isin(background_set)]

    # 一次性按 term 分组（不要每个 term 都全表过滤一遍：真实数据注释几十万行，
    # 循环里反复过滤会慢几十倍）
    term_to_genes = ann.groupby("term")["gene"].agg(set)

    rows = []
    for term, genes_in_term in term_to_genes.items():
        K = len(genes_in_term)
        k = len(genes_in_term & sig_set)
        if K == 0 or k == 0:
            continue  # 该通路在显著基因里一个都没有，谈不上富集
        pvalue = _hypergeom_upper_pvalue(N, K, n, k)
        rows.append({
            "term": term,
            "bg_count": K,
            "sig_count": k,
            "pvalue": pvalue,
            "enriched_genes": ",".join(sorted(genes_in_term & sig_set)),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    # BH 多重检验校正（和 diff.py 同一套逻辑）
    res = res.sort_values("pvalue").reset_index(drop=True)
    m = len(res)
    res["fdr"] = res["pvalue"] * m / (np.arange(m) + 1)
    res["fdr"] = res["fdr"][::-1].cummin()[::-1]
    res["fdr"] = res["fdr"].clip(upper=1.0)
    res["significant"] = res["fdr"] < fdr_threshold

    return res.sort_values("fdr").reset_index(drop=True)
