# 运营智脑 · 跨境电商 Multi-Agent 运营自动化平台

> 开发周期：2026.02 – 2026.08
> 用 LangGraph 把 6 个角色 Agent 编成一张状态图，自动完成跨境电商运营全流程：
> 市场调研 → 竞品分析 → 评论洞察 → Listing 生成 → 广告关键词 → 运营周报。

> 项目重点放在可控和可度量上：每一步有状态约束，每次运行有评测结果。

---

## 一、5 分钟跑起来

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置（复制后填自己的 Key）
cp .env.example .env
#   LLM_BASE_URL / LLM_API_KEY / LLM_MODEL  ← 支持 DeepSeek / 通义 / 智谱 / OpenAI 兼容接口
#   没填 Key 也没关系，会自动降级 mock 模式，项目照样能跑通全链路

# 3. 先跑一次演示，确认环境 OK
python -m scripts.demo

# 4. 跑单元测试
python -m pytest tests/ -v

# 5. 启动服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 6. 打开浏览器
#    演示前端： http://localhost:8000
#    接口文档： http://localhost:8000/docs
```

**Docker 一键部署（单容器，默认 SQLite，无需 PG/Redis）：**
```bash
docker compose up -d --build
docker compose logs -f app
```
> 本项目默认存储是 SQLite、缓存是进程内 LRU，**不依赖 PostgreSQL / Redis**。
> 若需「重启不丢 checkpoint」，装 `langgraph-checkpoint-postgres` 并把 `DATABASE_URL`
> 指向 PG 即可（见 `app/memory/persistence.py` 的自动降级逻辑）。

**跑评测（生成报告）：**
```bash
python -m scripts.run_eval --limit 6 --no-judge   # 快速版，不进 LLM 裁判
python -m scripts.run_eval                        # 完整版，含 judge 抽样
# 报告输出：docs/eval_report.md
```

---

## 二、架构

```
                        ┌──────────────────────────────┐
   用户需求 ──────────▶ │  prepare  召回长期记忆        │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  planner  拆解为子任务 DAG     │  ← LLM + JSON Schema 强约束
                        └──────────────┬───────────────┘
                                       ▼
              ┌────────────────────────────────────────────┐
              │  execute  按批次并行执行子任务（循环边）      │
              │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
              │  │Researcher│ │ Analyst  │ │  Writer  │   │  ← 同批 asyncio.gather 并发
              │  │ 工具采集  │ │ 计算分析  │ │ 内容生成  │   │
              │  └──────────┘ └──────────┘ └──────────┘   │
              └───────────────┬────────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  writer  汇总成稿 │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐        不通过且未超轮次
                    │  critic  质量审核 │ ◀──────────────┐
                    └────────┬─────────┘                │
                             │ 通过                      │
                             │                  ┌───────────────┐
                             │                  │ revise 回灌重写│
                             │                  └───────────────┘
                             ▼
                    ┌──────────────────┐
                    │ finalize 定稿沉淀 │  越改越差 → 自动回滚 + 写入长期记忆
                    └──────────────────┘
```

### 六层模块

| 层 | 目录 | 职责 | 设计要点 |
|---|---|---|---|
| 配置层 | `app/core/config.py` | 环境变量、类型校验、密钥隔离 | 12-Factor App |
| 契约层 | `app/core/schemas.py` | Pydantic 数据模型、DAG 拓扑排序 | 结构化输出 |
| 观测层 | `app/core/observability.py` | Trace 树、成本记账、P50/P95 指标 | 全链路可观测 |
| 模型层 | `app/llm/client.py` | 模型路由、指数退避重试、结构化输出修复 | 成本控制 + 稳定性 |
| 记忆层 | `app/memory/` | L1 窗口 / L2 摘要 / L3 向量记忆、Checkpointer | 长短期记忆管理 |
| 工具层 | `app/tools/registry.py` | 12 个 MCP 风格工具、校验 / 重试 / 幂等 / 权限 | Function Calling |
| 智能体层 | `app/agents/roles.py` | 6 个角色的 prompt 与执行逻辑 | Multi-Agent 协作 |
| 编排层 | `app/graph/workflow.py` | LangGraph 状态图、条件边、循环反思 | 状态机编排 |
| 评测层 | `app/eval/harness.py` | 三层评测体系 + CI 门禁 | 结果可量化 |
| 接口层 | `app/api/routes.py` | FastAPI + SSE 流式 | 工程交付 |

---

## 三、12 个内置工具（MCP 风格）

| 工具 | 用途 | 特点 |
|---|---|---|
| `competitor_scraper` | 竞品榜单抓取 | 幂等缓存 |
| `review_insight` | 评论痛点/好评点抽取 | 幂等缓存 |
| `sql_query` | 只读 SQL 查询 | **安全四道闸**：只读校验+白名单+行数上限+超时 |
| `calculator` | 安全计算器 | AST 白名单，杜绝代码注入 |
| `keyword_extract` | 关键词抽取 | 用于 Listing 埋词 |
| `sentiment_analysis` | 批量情感分析 | 返回分布+样本 |
| `trend_analysis` | 搜索热度趋势 | 判断上升期/衰退期 |
| `price_history` | 价格曲线与波动率 | 判断价格战 |
| `image_generator` | 产品场景图生成 | 对接 ComfyUI 的位置 |
| `erp_inventory` | 库存与补货预警 | 业务系统集成 |
| `exchange_rate` | 汇率查询 | 多币种定价 |
| `web_search` | 联网检索 | 补充行业趋势 |

> 工具目前是**确定性模拟实现**（保证可复现、零外部依赖、CI 可跑）。
> 生产替换只需改 handler，Agent 侧代码零改动 —— 这是依赖倒置的价值。

---

## 四、评测体系（本项目核心竞争力）

```
L1 确定性断言  ── 快、免费、100% 稳定
   ├─ 产出非空 / 最小长度
   ├─ 关键词命中率
   ├─ 违禁词检测（合规红线）
   └─ 交付结构完整性

L2 Agent 行为指标  ── 针对 Agent 系统特有，可解释
   ├─ 子任务完成率
   ├─ 角色分工覆盖率
   ├─ 依赖引用合法性
   ├─ 反思有效性（打回后分数是否提升）
   └─ 单任务 token 成本

L3 LLM-as-Judge  ── 有偏差，仅抽样
   └─ 分维度打分：相关性 / 事实性 / 完整性 / 可执行性

L4 人工抽样校准  ── 金标准
   └─ 算 judge 与人工的 MAE，< 0.1 才认为 judge 可用
```

**20 条内置用例**覆盖：典型场景 10 条、全链路 2 条、边界场景 3 条、
高风险合规 3 条（critical，一票否决）、成本稳定性 2 条。

**CI 门禁规则**：critical 用例全通过 + 整体通过率 ≥ 80%。

---

## 五、关键设计取舍

| 决策 | 选择 | 理由 |
|---|---|---|
| 编排框架 | LangGraph | 需要两条循环边 + 并行分支 + 断点续跑，自研成本高于收益 |
| 通信模型 | 共享状态（黑板模式） | 新 Agent 接入不改老代码；代价是状态膨胀，用截断控制 |
| 上下文策略 | 依赖驱动最小注入 | 只注入 depends_on 的产出，token 降本主要来源 |
| 模型路由 | 规则打分（非分类器） | 冷启动无标注数据，先规则上线，再用真实失败案例训练分类器 |
| 记忆 L3 | 哈希词袋向量 | 演示零依赖；生产换 BGE-M3 + Milvus，接口不变 |
| 幻觉治理 | 结构化输出 + 引用约束 + Critic 拦截 | 三道闸，单靠一个都不够 |
| 反思循环 | 轮次上限 + 越改越差回滚 | 无限循环烧钱；LLM 修改可能把好的改坏 |

---

## 六、目录结构

```
xiaoshou-zhinao/
├── app/
│   ├── main.py                  # FastAPI 入口（lifespan 管理）
│   ├── core/
│   │   ├── config.py            # 配置层
│   │   ├── schemas.py           # 数据契约 + DAG 拓扑排序
│   │   └── observability.py     # Trace / 成本 / 指标
│   ├── llm/client.py            # 模型路由 + 重试 + 结构化输出
│   ├── memory/
│   │   ├── store.py             # L1/L2/L3 记忆
│   │   └── persistence.py       # Checkpointer + 任务仓库
│   ├── tools/registry.py        # 12 个工具 + 统一调用入口
│   ├── agents/roles.py          # 6 个 Agent
│   ├── graph/workflow.py        # LangGraph 状态图（心脏）
│   ├── eval/harness.py          # 评测体系
│   └── api/routes.py            # REST + SSE 接口
├── frontend/index.html          # 演示前端（零依赖，原生 JS）
├── scripts/
│   ├── demo.py                  # 一键演示
│   └── run_eval.py              # 评测入口
├── tests/test_core.py           # 43 个单测
├── data/
│   ├── traces/                  # Trace 归档（指标：token/耗时/成本/span 树）
│   ├── tasks/                   # 任务业务视图（含 archived_path）
│   └── deliverables/            # ★ 交付物正文归档（Markdown，可追溯）
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml     # CI（测试 + 评测门禁 + 镜像构建）
```

---

## 七、实测数据

> **数据可复现原则**：本节每个数字都能从 `data/traces/*.json` 复算。
> 一键复算：`python -m app.eval.metrics_report --out docs/measured_metrics.md`。
>
> ⚠️ **关于"提升率"的如实说明**：本项目**未做受控 A/B 实验**，
> 因此**不宣称任何"改造前后提升 X%"**。不同 run 之间的差异混杂了
> 反思轮次、是否重试等多个变量（详见 `docs/实测数据档案_三次真实运行.md` §一）。
> 我们只报告**可对账的绝对值**，以及**已确证的机制**。

### 真实模型运行记录（阿里云百炼 `qwen3.8-flash`，2026-10-10）

| # | trace_id | 目的 | 墙钟耗时 | token | 成本 |
|---|---|---|---|---|---|
| 1 | `tr_85e50d9c1118` | 首次端到端 | **1396.7 s（23.3 min）** | 130,159 | 未记录 † |
| 2 | `tr_094845f0d766` | 长文生成端到端 | **698.4 s（11.6 min）** | 57,958 | **$0.0177** |
| 3 | `tr_acb6eb6301c5` | 归档链路验收 | **224.1 s（3.7 min）** | 21,648 | $0.0060 |
| 4 | `tr_1624235a52c8` | 另一次长跑 | **1050.5 s（17.5 min）** | 98,865 | $0.0300 |

> † trace 1 运行时 `PRICE_TABLE` 尚未收录 `qwen3.8-flash`，`cost_usd` 记为 0（已修复，见踩坑记录）。
> 耗时口径：**墙钟** = `max(end_ts) − min(start_ts)`（即用户实际等待时长）。
>
> ⚠️ **一个值得说的真实 bug**：这些归档 trace 是**修复前**产生的，文件里的
> `wall_latency_ms` 字段存的是**旧的错误值**（取"最长单节点"而非端到端）。
> 例如 `tr_85e50d9c1118` 存的是 377.1s，而真实端到端是 **1396.7s**（差 3.7 倍）。
> 上表列的是**按修正公式重新复算**的值；`docs/measured_metrics.md` 也会重新算一遍
> （脚本刻意不信任存储字段）。这正说明"能复算"比"存了什么"更重要。

### 全量归档统计（来自 `docs/measured_metrics.md`，自动生成）

| 指标 | 值 | 口径 |
|---|---|---|
| 有效 trace 数 | 43 | 含 span 的归档 |
| 工具调用成功率 | **100%**（84/84）| `kind=tool` 且 `status=ok` |
| 工具重试次数 | **0** | `attempt>1` |
| 累计 token | 467,894 | 所有 trace 求和 |
| 累计成本 | $0.0537 | 4 次真实长跑中的 3 次有记账 |

### Run 2 逐节点实测（trace `tr_094845f0d766`，21 个 span）

| 节点 | 耗时 | 占墙钟 |
|---|---|---|
| `planner` | 8.95 s | 1.3% |
| `execute`（3 子任务串行）| 279.61 s | 40.0% |
| `critic`（2 轮）| 46.34 s | 6.6% |
| **`revise`（重写）** | **363.46 s** | **52.0%** |
| `finalize` | 0.00 s | 0.0% |

> **洞察**：`writer:revise` 一个节点吃掉该 trace **52% 的时间、51% 的成本**
> （$0.009032 / $0.017697）。这是"反思循环是最大成本黑洞"的**同 trace 内**证据，
> 解释了为什么必须设轮次上限 + 「越改越差回滚」——无上限自我批判会烧穿成本。

### 关键性能指标

| 指标 | 数值 | 来源 / 口径 |
|---|---|---|
| 工具调用成功率 | **100%** | `metrics_report`，84/84 |
| 单元测试 | **46/46 通过** | `python -m pytest tests/ -q` |
| 连通性自检 | 脚本验证 3 项 | `scripts/check_bailian.py`（不落盘，可重跑）|
| 短/长输出延迟 | 见 `docs/bench_latency_result.md` | `scripts/bench_latency.py` 实测 |

### mock 基线（用于 CI 快速门禁）

| 指标 | 数值 | 来源 |
|---|---|---|
| 端到端墙钟 | **~151 ms** | `data/traces/tr_41d4293846ac.json` |
| 评测用例通过率 | 见 `docs/eval_report.md` 顶部 | `docs/eval_report.md`（mock 运行会标注非真实测量）|
| token 总量 | **3,244** | `data/traces/tr_46398b0ee5bf.json` |


> ⚠️ **mock 通过率偏低不是 bug**：mock 返回固定占位文本，不含业务关键词，
> L1 关键词命中率断言必然部分失败 —— **这恰恰证明评测在正常工作**
> （它能检测出"输出里没有业务内容"）。具体通过率随用例数变化，以 `docs/eval_report.md`
> 顶部标注为准；该报告在 mock 运行时会在开头明确标注"非真实测量"。
> 简历请引用真实模型口径的数字。

### 关键性能洞察（关键洞察）

以 Run 2 的 trace 为准（21 个 span，全部 ok）：

```
planner      8.95s  ( 2.5%)
execute     279.6s  (40.0%)   ← 3 个子任务串行
critic       46.3s  ( 6.6%)   ← 2 轮评审
revise      363.5s  (52.0%)   ← 单点最大耗时
finalize      0.0s  ( 0.0%)
```

| 环节 | 耗时 | 占比 | 成本占比 |
|---|---|---|---|
| `revise` 重写 | 363.5 s | **52%** | **51%**（$0.0090 / $0.0177）|
| `execute` 子任务 | 279.6 s | 40% | 34% |
| `critic` 评审 | 46.3 s | 7% | 13% |
| `planner` 规划 | 9.0 s | 1% | 2% |

**洞察**：反思循环是**最大时间与成本黑洞**（时间 52%、成本 51% 都压在 `revise` 一个节点上），
说明 Critic 通过阈值（0.80）偏严。这引出设计权衡：阈值高 → 质量好但慢且贵；阈值低 → 快且省但质量差。

**更进一步**：Run 1 打了 2 轮（两次 revise），Run 2 只打了 1 轮。
这 1 轮之差让总耗时从 23.3 min 降到 11.6 min、token 从 130k 降到 58k —— **反思轮次是最敏感的成本杠杆**。
**正确做法是用评测数据找阈值拐点，而不是拍脑袋定。**

---

## 八、边界与失败模式（如实交代的"诚实清单"）

> 一个 Agent 项目值不值得信，不看它顺利时多漂亮，看它**失败时能不能说清怎么失败的**。
> 下面每一条都对应代码里的具体处理，而不是空话。

### 8.1 明确的边界

1. **工具是确定性模拟的** —— 用固定返回保证可复现；生产需替换 handler 并处理反爬/限流/鉴权。
2. **Checkpointer 默认内存态** —— 默认 `MemorySaver`，**重启即丢**；要「断点续跑」
   需装 `langgraph-checkpoint-postgres` 并把 `DATABASE_URL` 指向 PostgreSQL
   （`build_checkpointer()` 已写好自动降级，失败会回退内存并打 warning，不会崩）。
3. **单机、无多租户** —— 存储是 SQLite，缓存是进程内 LRU；真实 SaaS 需按租户分向量命名空间。
4. **Critic 存在自我偏好偏差** —— 同模型自评不可靠，建议换厂商交叉评审 + 人工抽样校准
   （`human_calibration()` 提供了 MAE/一致性计算脚手架）。
5. **成本为估算值** —— 按返回 usage 精算（Run 2 起），但生产应以账单对账为准。

### 8.2 已知失败模式与处理

| 失败模式 | 触发场景 | 代码里的处理 |
|---|---|---|
| **LLM 输出非法 JSON** | 模型偶发不遵守格式 | `extract_json()` 容错解析 + Pydantic 校验失败重试 |
| **工具参数非法/缺失** | 模型自作主张加参数 | 校验失败**不抛异常**，把"人话"错误回灌让模型自纠正 |
| **安全拦截后静默 fail-open** | 合规检查命中但流程照跑 | 已修复：拦截即返回 `ok=False` 并硬中断（曾有该缺陷，见 `踩坑记录.md`） |
| **长任务产物丢失** | 早期 TaskStore 纯内存，重启归零 | 已修复：落盘 `data/tasks/*.json`，重启自动恢复 |
| **并行分支写状态竞争** | 同批次子任务并发写共享 state | 子任务结果用 reducer 归并，避免覆盖 |
| **上下文无限膨胀** | 长任务把全部历史塞 prompt | 滑动窗口 + 触发式摘要压缩 + 只注入相关结论 |
| **Critic 打回后越改越差** | 反思循环反向优化 | 检测"修订后评分下降"则**回滚到上一版**（`workflow.py`） |
| **额度/网络超时** | 长文生成耗时远超预期 | `timeout=180s`；区分确定性/瞬时错误，仅后者退避重试 |
| **`sql_query` 无真实数据** | 模拟环境返回执行计划 | 模型**诚实标注 `$[待回填]` 而非编造数字**——这是事实性约束在生效 |

### 8.3 一句话一句话回答

> "这个项目的价值不在它有多完美，而在于**每一处失败我都能定位到根因并写进文档**。
> 比如我发现过三个真 bug：墙钟耗时算错、安全拦截 fail-open、长任务产物丢内存——
> 都在 `docs/踩坑记录.md` 里有定位过程和修复验证。"

---

## 九、文档

- [`docs/逐步实现指南_实战增强版.md`](docs/逐步实现指南_实战增强版.md) —— **从零实现的完整顺序 + 每步设计要点**
- [`docs/实测数据档案_三次真实运行.md`](docs/实测数据档案_三次真实运行.md) —— **★ 三次真实运行完整数据（可下载引用）**
- [`docs/measured_metrics.md`](docs/measured_metrics.md) —— **★ 从 trace 自动复算的指标（可重跑验证）**
- [`docs/真实模型实测报告.md`](docs/真实模型实测报告.md) —— **真实模型实测数据 + 3 个坑的定位与修复**
- [`docs/踩坑记录.md`](docs/踩坑记录.md) —— 6 个真实工程坑的深入复盘（含"长任务产物丢失"）
- [`docs/百炼模型选型指南.md`](docs/百炼模型选型指南.md) —— 阿里云百炼模型选型与接入
- [`docs/eval_report.md`](docs/eval_report.md) —— 评测报告（跑完 `run_eval` 生成，含 mock 标注）
- [`CONTRIBUTING.md`](CONTRIBUTING.md) —— 贡献指南（含"禁止编造数据"红线）

### 快速验证命令

```bash
bash scripts/reproduce.sh                             # ★ 一键复现所有文档指标（离线）
python -m pytest tests/ -q                            # 46 个单测，约 1 秒
python scripts/check_bailian.py --model qwen3.8-flash # 连通性自检，约 10 秒
python -m scripts.demo --mock                         # 离线端到端，约 0.2 秒
python -u -m scripts.demo --query "..."               # 真实模型端到端，约 5~25 分钟
python -m app.eval.metrics_report --out docs/measured_metrics.md  # 指标复算
```

**查看某次任务的归档交付物：**
```bash
curl -s http://localhost:8000/api/task/<task_id>/deliverable | python3 -m json.tool
# 或直接看文件： data/deliverables/<task_id>.md
```
