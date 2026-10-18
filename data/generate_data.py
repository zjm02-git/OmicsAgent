"""生成模拟基因表达数据，供项目演示。

会生成三个文件（和真实生物信息数据格式一致）：
  expression.csv      —— 基因 × 样本的表达矩阵（第一列是基因名）
  metadata.csv        —— 样本 → 分组（control / treatment）
  gene_ontology.csv   —— 基因 → 通路注释（供富集分析用）

数据是模拟的：随机基线 + 人为给部分基因加上差异，
这样没有真实数据也能把全流程跑通。
以后换成 TCGA / GEO 等真实公开数据，格式不用改，直接替换文件即可。
"""

import os

import numpy as np
import pandas as pd

N_GENES = 100        # 基因数
N_PER_GROUP = 4      # 每组样本数
UP_GENES = range(0, 20)     # 前 20 个基因：treatment 组上调
DOWN_GENES = range(20, 30)  # 第 20~29 个：treatment 组下调


def main():
    rng = np.random.default_rng(42)  # 固定随机种子，结果可复现

    genes = [f"GENE{i:04d}" for i in range(N_GENES)]
    ctrl = [f"ctrl_{i + 1}" for i in range(N_PER_GROUP)]
    treat = [f"treat_{i + 1}" for i in range(N_PER_GROUP)]
    samples = ctrl + treat

    # 每个基因的基础表达量（服从正态分布）
    base = rng.normal(8.0, 1.0, N_GENES)

    # 造表达矩阵：基线 + 随机噪声
    expr = pd.DataFrame({"gene": genes})
    for s in samples:
        expr[s] = base + rng.normal(0, 0.5, N_GENES)

    # 人为制造差异：让部分基因在 treatment 组表达量翻 3 倍 / 降到 1/3
    # （这样 log2FC ≈ ±1.58，超过默认阈值 1.0，能被判为显著差异）
    for i in UP_GENES:
        expr.loc[i, treat] = expr.loc[i, treat] * 3.0
    for i in DOWN_GENES:
        expr.loc[i, treat] = expr.loc[i, treat] / 3.0

    here = os.path.dirname(os.path.abspath(__file__))
    expr.to_csv(os.path.join(here, "expression.csv"), index=False)

    meta = pd.DataFrame({
        "sample_id": samples,
        "condition": ["control"] * N_PER_GROUP + ["treatment"] * N_PER_GROUP,
    })
    meta.to_csv(os.path.join(here, "metadata.csv"), index=False)

    # 造通路注释（模拟 GO）：让上调基因扎堆在"细胞周期/炎症"，
    # 下调基因扎堆在"凋亡"，这样富集分析能找出这些通路。
    # 真实项目里换成 g:Profiler / 真实 GO 注释文件，格式一样，无需改代码。
    ann_rows = []
    for i in range(0, 10):                      # 上调前 10 个 → 细胞周期
        ann_rows.append((f"GENE{i:04d}", "GO:0007049 cell cycle"))
    for i in range(10, 20):                     # 上调后 10 个 → 炎症反应
        ann_rows.append((f"GENE{i:04d}", "GO:0006954 inflammatory response"))
    for i in range(20, 30):                     # 下调 10 个 → 凋亡
        ann_rows.append((f"GENE{i:04d}", "GO:0006915 apoptosis"))
    bg_terms = [
        "GO:0008152 metabolism",
        "GO:0006810 transport",
        "GO:0007165 signal transduction",
        "GO:0006351 transcription",
    ]
    for i in range(30, N_GENES):                # 其余背景基因分散到杂项通路
        ann_rows.append((f"GENE{i:04d}", bg_terms[i % len(bg_terms)]))
    ann = pd.DataFrame(ann_rows, columns=["gene", "term"])
    ann.to_csv(os.path.join(here, "gene_ontology.csv"), index=False)

    print(f"已生成 expression.csv：{N_GENES} 个基因 × {len(samples)} 个样本")
    print(f"已生成 metadata.csv：control {N_PER_GROUP} 个 / treatment {N_PER_GROUP} 个")
    print(f"已生成 gene_ontology.csv：{len(ann)} 条注释")


if __name__ == "__main__":
    main()
