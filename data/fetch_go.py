# -*- coding: utf-8 -*-
"""从 MSigDB 下载人类 GO 生物过程注释，转成项目 data/gene_ontology.csv。

数据源：MSigDB C5 GO:BP 基因集（symbols 版，GMT 格式，无需登录）
    https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c5.go.bp.v2024.1.Hs.symbols.gmt

生成文件（在 data/ 下）：
    - gene_ontology.csv : gene,term（gene=HUGO 符号，term=GO 生物过程术语，一个基因可属于多个术语）

用法（在项目根目录执行）：
    python data/fetch_go.py
"""
import csv
import os
import urllib.request

GMT_URL = ("https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
           "2024.1.Hs/c5.go.bp.v2024.1.Hs.symbols.gmt")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    req = urllib.request.Request(GMT_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        lines = [line.decode("utf-8") for line in r]

    out_path = os.path.join(OUT_DIR, "gene_ontology.csv")
    n_terms = 0
    n_assoc = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gene", "term"])
        for line in lines:
            cols = line.rstrip("\n").split("\t")
            term = cols[0]
            genes = [g for g in cols[2:] if g]
            n_terms += 1
            for g in genes:
                w.writerow([g, term])
                n_assoc += 1
    print(f"已写 gene_ontology.csv：{n_terms} 个 GO 术语，{n_assoc} 条基因-术语关联")


if __name__ == "__main__":
    main()
