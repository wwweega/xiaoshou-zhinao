#!/usr/bin/env bash
# ============================================================
# 运营智脑 · 一键复现脚本
# ============================================================
# 用途：让任何人 clone 本仓库后，用一条命令复现出文档里引用的所有指标。
#
# 用法：
#   bash scripts/reproduce.sh            # 离线（mock）模式，不花钱，验证链路可跑
#   bash scripts/reproduce.sh --real     # 真实模型模式（需先在 .env 配好 Key，会产生费用）
#
# 产出：
#   docs/eval_report.md          —— 离线评测报告
#   docs/measured_metrics.md     —— 从 data/traces 复算的真实指标
#   （--real 时还会跑 scripts/bench_latency.py 生成 docs/bench_latency_result.md）
#
# 设计原则（面试可讲）：
#   文档里的每个数字，都必须能由这个脚本重新算出来。
#   跑不出同一量级的数字，就说明文档在编——这脚本就是用来「打自己脸」的。
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "工作目录：$ROOT"

REAL=0
for arg in "$@"; do
  case "$arg" in
    --real) REAL=1 ;;
    *) echo "未知参数：$arg"; exit 2 ;;
  esac
done

PY="python3"
command -v python >/dev/null 2>&1 && PY="python"

# ---------- 0. 环境自检 ----------
echo ""
echo "==> [0/4] 环境自检"
$PY --version
if [ ! -f ".env" ] && [ ! -f ".env.example" ]; then
  echo "缺少 .env / .env.example"; exit 1
fi

if [ "$REAL" -eq 1 ]; then
  echo "   模式：真实模型（会调用付费 API）"
  export LLM_PROVIDER="${LLM_PROVIDER:-openai_compatible}"
else
  echo "   模式：离线 mock（不调用外部 API）"
  export LLM_PROVIDER="mock"
fi

# ---------- 1. 单元测试 ----------
echo ""
echo "==> [1/4] 单元测试"
$PY -m pytest tests/ -q

# ---------- 2. 离线评测 ----------
echo ""
echo "==> [2/4] 离线评测 -> docs/eval_report.md"
$PY -m scripts.run_eval --limit 6 --no-judge --out docs/eval_report.md

# ---------- 3. 指标复算 ----------
echo ""
echo "==> [3/4] 从 trace 复算真实指标 -> docs/measured_metrics.md"
$PY -m app.eval.metrics_report --out docs/measured_metrics.md

# ---------- 4. 延迟探针（仅真实模式）----------
echo ""
if [ "$REAL" -eq 1 ]; then
  echo "==> [4/4] 延迟探针（真实模型，耗时数分钟）-> docs/bench_latency_result.md"
  $PY scripts/bench_latency.py --out docs/bench_latency_result.md
else
  echo "==> [4/4] 跳过延迟探针（离线模式）。要跑真实探针请加 --real"
fi

echo ""
echo "============================================================"
echo "  复现完成。请检查以下文件里的数字是否与文档一致："
echo "    - docs/eval_report.md"
echo "    - docs/measured_metrics.md"
[ "$REAL" -eq 1 ] && echo "    - docs/bench_latency_result.md"
echo "============================================================"
