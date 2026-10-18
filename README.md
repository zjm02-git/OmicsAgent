# 多组学数据分析智能体（OmicsAgent）

> 用 **LangGraph** 把「基因表达数据分析」这条多步骤流程，编排成一个**能持久化、能人工确认、能记住偏好、能自动解读结果**的 AI Agent。

---

## 一、这个项目是干嘛的

生物数据分析（比如差异表达分析）不是"问一句答一句"，而是一条**流水线**：

```
拿到表达数据 → 质检 → 差异分析 → 通路富集 → 解读结果 → 出报告
```

这条流水线有三大痛点：

1. **步骤多、容易断** —— 跑到一半出错，得从头再来；
2. **参数多、要试** —— fold change 阈值、p 值阈值选多少，不同选择结果不同；
3. **结果要解读** —— 一堆基因列表，得有人看懂它有什么生物学意义。

**OmicsAgent 做的事**：用 LangGraph 的「图」把这条流水线变成一个个**节点**，让大模型（LLM）负责"理解需求 + 解读结果"，让 Python 负责"质检 + 统计计算"，中间用**检查点**保存进度、用**条件边**做路由。这样一个不懂生物的人，说一句"帮我找差异表达基因"，就能拿到一份带解读的分析报告。

---

## 二、核心亮点（面试要讲的点）

| 亮点 | 用在哪 | 解决什么 |
| --- | --- | --- |
| **StateGraph 图编排** | 8 个节点串成流水线 | 把多步骤流程显式画成图，可读可维护 |
| **结构化输出路由** | 解析用户需求，判断分析类型 | 让 LLM 输出"确定的值"而不是自由文本，路由才可靠 |
| **条件边** | 质检不合格 → 走"失败报告"分支 | 流程能自己判断该走哪条路 |
| **Checkpointer 持久化** | InMemorySaver / PostgresSaver 可切换 | 检查点存库，中断接着跑、重启不丢 |
| **Human-in-the-loop** | 质检后 interrupt 暂停等确认 | 关键步骤人工把关，结果可审核 |
| **长期记忆** | InMemoryStore / PostgresStore | 记住用户上次的阈值，下次自动复用 |
| **真实统计计算** | t 检验 + BH 校正 + log2FC | 差异分析是真的在算，不是 LLM 瞎编 |
| **通路富集分析** | 超几何检验 + BH 校正 | 找出差异基因扎堆的功能通路 |
| **LLM 结果解读** | 把基因列表翻译成人话 | 结果可解释，不是冷冰冰的数字 |

---

## 三、架构图

```
用户："帮我找两组样本的差异表达基因，并解读"
        │
        ▼
┌─────────────────────────────────────────────┐
│  State（状态）：一条分析任务从头到尾共享的数据  │
└─────────────────────────────────────────────┘
        │
        ▼
[① parse_intent]   结构化输出 → 判断分析类型
        │
        ├── other ─────────► [⑧ report] 出"暂不支持"报告
        ▼ differential
[② load_qc]        加载数据 + 质检，读长期记忆的偏好阈值
        │
        ├── 质检不合格 ──► [⑧ report] 出"质检失败报告"
        │
        ▼ 质检合格
[③ human_review]   interrupt 暂停，等人工确认
        │
        ├── 用户取消 ──► [⑧ report] 出"已取消"报告
        │
        ▼ 用户批准
[④ diff_analysis]  差异表达分析（t 检验 + BH 校正）
        │
        ▼
[⑤ enrichment]     通路富集分析（超几何检验 + BH 校正）
        │
        ▼
[⑥ interpret]      LLM 把基因列表解读成人话
        │
        ▼
[⑦ save_memory]    把本次阈值偏好写回长期记忆
        │
        ▼
[⑧ report]         汇总成 Markdown 报告
        │
        ▼
       END
```

---

## 四、目录结构

```
多组学数据分析智能体/
├── app.py                  # 入口：跑一次完整分析
├── omics_agent/            # 核心代码包
│   ├── config.py           # 读配置、造模型、造检查点/记忆（工厂模式）
│   ├── state.py            # 定义 State（状态）
│   ├── diff.py             # 差异表达分析（纯 Python 统计）
│   ├── enrichment.py       # 通路富集分析（超几何检验）
│   ├── nodes.py            # 8 个节点函数（含人工确认、读写记忆）
│   └── graph.py            # 把节点拼成图 + 条件路由 + 检查点/记忆
├── data/
│   └── generate_data.py    # 生成模拟数据（可换真实数据）
├── docs/
│   ├── 项目总览.md          # 简洁完整版
│   ├── 项目详解.md          # 通俗易懂且完整
│   ├── 面试知识点.md        # 要背的面试知识点
│   └── 项目设计.md          # 历史文档入口（跳转到上面三份）
├── requirements.txt        # 依赖
├── .env.example            # 环境变量模板（真 key 别提交）
└── .gitignore
```

---

## 五、快速开始

### 1. 装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型 Key

把 `.env.example` 复制成 `.env`，填上你的 DeepSeek Key（或直接复用你 chapter03 目录里那份 `.env`）：

```
DEEPSEEK_API_KEY=你的key
```

### 3. 生成模拟数据

```bash
python data/generate_data.py
```

会生成 `data/expression.csv`（表达矩阵）、`data/metadata.csv`（样本分组）、`data/gene_ontology.csv`（通路注释）。

### 4. 跑一次完整分析

```bash
python app.py        # 质检后会暂停，等你确认（Human-in-the-loop）
python app.py --auto # 自动确认，跳过人工确认（演示用）
```

会看到 Agent 一步步跑：解析需求 → 质检 → 人工确认 → 差异分析 → 通路富集 → 解读 → 出报告。

### 5.（可选）启用 PostgreSQL 持久化

在 `.env` 里设 `USE_POSTGRES=true`，检查点和长期记忆就存进 PostgreSQL（重启不丢）。默认内存版开箱即跑，无需数据库。

---

## 六、后续规划（Roadmap）

- [ ] 换真实公开数据（TCGA / GEO）
- [x] 持久化换成 PostgresSaver（跨重启不丢）
- [ ] Fork 分叉：同一份数据用不同阈值跑两条分支对比
- [x] 富集分析（GO / KEGG）
- [x] Human-in-the-loop：关键步骤人工确认
- [x] 长期记忆：记住用户偏好
- [ ] Streamlit 网页界面

---

## 七、技术栈

Python · LangGraph · LangChain · DeepSeek · pandas · NumPy · SciPy · PostgreSQL
