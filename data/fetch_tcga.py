# -*- coding: utf-8 -*-
"""从 GDC 官方 API 下载 TCGA-BRCA 真实表达数据，转成项目 data/ 需要的格式。

数据源：GDC（Genomic Data Commons，TCGA 官方门户）
    - 表达文件：STAR - Counts 流程的 per-sample TSV（含 fpkm_uq 列）
    - 分组依据：样本类型 Primary Tumor=肿瘤 / Solid Tissue Normal=正常
    - 规模：取 BRCA 全部正常样本，肿瘤取同样数量保持两组平衡（约 113+113）

生成文件（在 data/ 下，直接替换模拟数据即可跑）：
    - expression.csv   : gene × 样本矩阵，值 = log2(fpkm_uq + 1)，首列 gene
    - metadata.csv     : sample_id,condition（tumor / normal）

用法（在项目根目录执行）：
    python data/fetch_tcga.py
"""
import csv
import json
import math
import os
import tempfile
import time
import urllib.request

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")


def _post(query: dict) -> dict:
    req = urllib.request.Request(
        GDC_FILES, data=json.dumps(query).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def query_samples(sample_type: str, n: int) -> list:
    """分页拉取最多 n 个指定样本类型的文件：[(file_id, submitter_id), ...]"""
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}},
        {"op": "in", "content": {"field": "cases.samples.sample_type", "value": [sample_type]}},
    ]}
    out = []
    for start in range(0, n, 100):
        q = {
            "filters": filters,
            "fields": "file_id,cases.samples.submitter_id",
            "expand": "cases.samples",
            "size": min(100, n - len(out)),
            "from": start,
        }
        hits = _post(q)["data"]["hits"]
        for h in hits:
            sub = h["cases"][0]["samples"][0]["submitter_id"]
            out.append((h["file_id"], sub))
        if len(hits) < 100:
            break
    return out


def download(file_id: str, dest: str) -> None:
    """下载一个样本文件，带重试。"""
    req = urllib.request.Request(GDC_DATA + "/" + file_id, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
                f.write(r.read())
            return
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"下载失败 {file_id}")


def parse_sample(path: str) -> dict:
    """解析一个样本 TSV，返回 {gene_name: fpkm_uq}，只留 protein_coding 且有符号名的。"""
    vals = {}
    with open(path, encoding="utf-8") as f:
        col = None
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if col is None:
                col = {name: i for i, name in enumerate(parts)}
                continue
            if parts[col["gene_type"]] != "protein_coding":
                continue
            name = parts[col["gene_name"]]
            if not name:
                continue
            raw = parts[col["fpkm_uq_unstranded"]]
            if raw in ("", "NA"):
                continue
            vals[name] = float(raw)
    return vals


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_dir = tempfile.mkdtemp(prefix="tcga_brca_")

    # 1. 取全部正常样本，肿瘤取同样数量保持两组平衡
    normal = query_samples("Solid Tissue Normal", 500)
    tumor = query_samples("Primary Tumor", len(normal))
    samples = [(fid, sub, "tumor") for fid, sub in tumor] + \
              [(fid, sub, "normal") for fid, sub in normal]
    print(f"共 {len(samples)} 个样本（{len(tumor)} 肿瘤 + {len(normal)} 正常）")

    # 2. 下载 + 解析，累积 gene -> {sample: value}
    expr = {}          # gene -> {sample_id: fpkm_uq}
    gene_order = None  # 用第一个文件的基因顺序，保证对齐
    for i, (fid, sub, cond) in enumerate(samples, 1):
        dest = os.path.join(raw_dir, fid + ".tsv")
        download(fid, dest)
        vals = parse_sample(dest)
        if gene_order is None:
            gene_order = list(vals.keys())
        for g, v in vals.items():
            expr.setdefault(g, {})[sub] = v
        print(f"  [{i}/{len(samples)}] {sub}（{cond}）{len(vals)} 个蛋白编码基因")
        time.sleep(0.1)

    # 3. 写 expression.csv（log2(fpkm_uq+1)）
    expr_path = os.path.join(OUT_DIR, "expression.csv")
    sample_ids = [s[1] for s in samples]
    with open(expr_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gene"] + sample_ids)
        for g in gene_order:
            row = [g]
            for sid in sample_ids:
                v = expr.get(g, {}).get(sid, 0.0)
                row.append(round(math.log2(v + 1.0), 6))
            w.writerow(row)
    print(f"已写 expression.csv：{len(gene_order)} 基因 × {len(sample_ids)} 样本")

    # 4. 写 metadata.csv
    meta_path = os.path.join(OUT_DIR, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "condition"])
        for _, sid, cond in samples:
            w.writerow([sid, cond])
    print(f"已写 metadata.csv：{len(samples)} 样本")


if __name__ == "__main__":
    main()
