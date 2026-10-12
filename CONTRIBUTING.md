# 贡献指南（CONTRIBUTING）

感谢你关注「运营智脑」。本项目是一个**强调真实性与可复现性**的 Multi-Agent 工程示范，
所以在提 PR 前，请先读一下下面几条「红线」——它们比一般的代码风格更重要。

## 一、核心红线：不允许编造数据

这是本仓库的最高原则。任何出现在 README / docs / 代码注释里的**数字**，都必须：

1. **可追溯到来源** —— 例如 `data/traces/<trace_id>.json` 里有对应的实测字段；
2. **可被复算** —— 别人 clone 下来跑对应的脚本，能得到量级一致的数；
3. **有明确口径** —— "准确率"要说清分母是什么、谁标的、"提升率"要说清对照组是什么。

具体禁止：

- ❌ 写"效率提升 78%""准确率 0.71→0.93"这类**没有受控 A/B 就给出的提升率**；
- ❌ 用占位符/目标值充当实测值（如把"计划做到 200 条用例"写成"自建 200 条用例"）；
- ❌ 技术栈里写"配了但从没被 import 的技术"（如无引用的 Redis 配置）——这叫**名实不符**。

> 需要展示"机制有效"又拿不到受控数据时，正确做法是**只讲机制、不报数字**，
> 或补一个可复现的测量脚本 + 归档。参考 `scripts/bench_latency.py` 与
> `app/eval/metrics_report.py` 的做法。

## 二、开发环境

```bash
# 1. Python 3.11+
python -m venv .venv && source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（不配 Key 也能跑，会自动降级到 mock）
cp .env.example .env
#   编辑 .env 填入你的 LLM Key；或直接保持 mock 模式做离线开发
```

## 三、提交前必须通过

```bash
# 单元测试（mock 模式，不花钱）—— 必须全绿
LLM_PROVIDER=mock python -m pytest tests/ -q

# 离线评测（含发布门禁）
LLM_PROVIDER=mock python -m scripts.run_eval --limit 6 --no-judge

# 指标复算（确认你引用的数字能被重新算出来）
python -m app.eval.metrics_report --out /tmp/m.md
```

CI（`.github/workflows/ci.yml`）会在 push / PR 时自动跑上面这些，**且强制 mock 模式**
（绝不在 CI 里调用付费 API）。

## 四、代码约定

- **类型注解**：公开函数都要有完整注解；`from __future__ import annotations` 开头。
- **docstring**：重点解释「**为什么这么设计 / 踩了什么坑**」，而不是复述代码在做什么。
  本项目把 docstring 当作"设计说明"，请保持这个风格。
- **新增指标**：若加了新指标，必须同时给出**计算方式**与**数据来源**，并尽量补一个单测。
- **测试**：新功能请补对应测试。当前基线是 `46 passed`，PR 不应让它下降。
- **密钥**：`data/`、`.env` 已在 `.gitignore` 中，**永远不要**把真实 Key 或客户数据提交上来。

## 五、提交规范

建议使用清晰的 commit message，例如：

```
fix(observability): 修正 wall_latency_ms 误用单节点耗时的问题

原实现取「最长根 span」当作端到端耗时，在串行多节点场景下偏差可达 3.7 倍。
改为 max(end_ts) - min(start_ts)，并补充 sum_latency_ms 作为算力口径。
新增 3 个回归测试锁定该行为。
```

## 六、可复现性检查清单（提 PR 自检）

- [ ] 我新增/修改的数字，都能 `grep <trace_id> data/traces/` 找到出处
- [ ] 我新增的断言/指标有单测覆盖
- [ ] `pytest` 全绿
- [ ] 没有提交任何真实 Key / 私有数据
- [ ] 若涉及性能宣称，附了可复现脚本（而非手写数字）
