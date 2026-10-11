"""
度量报告生成器 · 从 trace 归档真实统计（不手写任何数字）
============================================================
设计原则（这是本模块存在的唯一理由）：
  → 文档里的每个指标都必须能**从原始归档复算**。
    手写数字 = 无法自证 = 面试一问就穿帮。
    所以这个脚本把"算指标"变成一条命令：`python -m app.eval.metrics_report`

统计口径（全部注明，避免"墙钟 vs 累加"这类口径混用）：
  - 墙钟耗时：max(end_ts) - min(start_ts)，即用户实际等待
  - 累加耗时：sum(span.latency_ms)，即算力占用（并行时 > 墙钟）
  - 工具成功率：kind == "tool" 且 status == "ok" 的占比
  - 成本：trace 顶层 total_cost_usd（由 llm client 按 API usage 记账）

用法：
    python -m app.eval.metrics_report
    python -m app.eval.metrics_report --out docs/measured_metrics.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.core.config import settings


def _load_traces(trace_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in sorted(trace_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if d.get("spans"):  # 跳过空 trace
            d["_file"] = f.name
            out.append(d)
    return out


def _recompute_wall_ms(trace: dict[str, Any]) -> float:
    """按修复后的口径重算墙钟（不信任文件里的 wall_latency_ms 旧值）。"""
    sp = trace["spans"]
    now = max(s["end_ts"] for s in sp)
    starts = [s["start_ts"] for s in sp]
    ends = [s["end_ts"] if s["end_ts"] is not None else now for s in sp]
    return round((max(ends) - min(starts)) * 1000, 2)


def _percentile(values: list[float], pct: float) -> float:
    """线性插值百分位（与 app.core.observability.MetricsCollector._pct 同口径）。"""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def compute(traces: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [_recompute_wall_ms(t) for t in traces]

    tool_total = tool_ok = 0
    tool_retried = 0
    agent_calls = 0
    for t in traces:
        for s in t["spans"]:
            if s["kind"] == "tool":
                tool_total += 1
                if s["status"] == "ok":
                    tool_ok += 1
                if int(s.get("attempt", 1) or 1) > 1:
                    tool_retried += 1
            elif s["kind"] == "agent":
                agent_calls += 1

    total_tokens = sum(t["total_tokens"] for t in traces)
    total_cost = sum(t["total_cost_usd"] for t in traces)
    costed = [t for t in traces if t["total_cost_usd"] > 0]

    return {
        "task_count": len(traces),
        "tool_total": tool_total,
        "tool_ok": tool_ok,
        "tool_retried": tool_retried,
        "agent_calls": agent_calls,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd": round(total_cost / len(costed), 6) if costed else 0.0,
        "costed_runs": len(costed),
        "wall_sum_s": round(sum(walls) / 1000, 1),
        "wall_avg_s": round(sum(walls) / len(walls) / 1000, 2) if walls else 0.0,
        "wall_max_s": round(max(walls) / 1000, 1) if walls else 0.0,
        "wall_p50_s": round(_percentile(walls, 0.50) / 1000, 2),
        "wall_p95_s": round(_percentile(walls, 0.95) / 1000, 2),
    }


def render(m: dict[str, Any], traces: list[dict[str, Any]]) -> str:
    tool_rate = (m["tool_ok"] / m["tool_total"] * 100) if m["tool_total"] else 0.0
    lines = [
        "# 度量报告（自动生成，勿手改）",
        "",
        "> 本文件由 `python -m app.eval.metrics_report` 从 `data/traces/*.json` 复算生成。",
        "> **所有数字均可 `grep <trace_id> data/traces/` 回溯**，不存在手写数字。",
        ">",
        "> 说明：仓库只提交 4 次「真实模型长跑」的 trace（见 `.gitignore`）。",
        "> 你本地跑过 mock 后会出现大量瞬时 trace，因此本地统计的 trace 数会大于 4，",
        "> 这是正常的（数字随数据变化，而非写死）—— 这正是「可复算」的体现。",
        "",
        "## 汇总",
        "",
        "| 指标 | 值 | 口径 |",
        "|---|---|---|",
        f"| 有效 trace 数 | {m['task_count']} | 含 span 的归档 |",
        f"| 工具调用总数 | {m['tool_total']} | kind=tool |",
        f"| 工具调用成功率 | {tool_rate:.1f}% | status=ok 占比 |",
        f"| 工具重试次数 | {m['tool_retried']} | attempt>1 |",
        f"| Agent 调用总数 | {m['agent_calls']} | kind=agent |",
        f"| 累计 token | {m['total_tokens']:,} | 所有 trace 求和 |",
        f"| 累计成本 | ${m['total_cost_usd']:.4f} | 顶层 total_cost_usd |",
        f"| 有成本的 run 数 | {m['costed_runs']} | cost>0 的 trace |",
        f"| 平均单次成本 | ${m['avg_cost_usd']:.4f} | 仅计有成本的 run |",
        f"| 墙钟 P50 | {m['wall_p50_s']} s | max(end)-min(start) |",
        f"| 墙钟 P95 | {m['wall_p95_s']} s | 线性插值 |",
        f"| 墙钟最大 | {m['wall_max_s']} s | 单次最长 |",
        "",
        "## 逐条 trace（按墙钟降序）",
        "",
        "| trace_id | 墙钟(s) | token | 成本($) | span | 工具/成功 |",
        "|---|---|---|---|---|---|",
    ]
    rows = []
    for t in traces:
        sp = t["spans"]
        tools = [s for s in sp if s["kind"] == "tool"]
        ok = sum(1 for s in tools if s["status"] == "ok")
        rows.append((
            _recompute_wall_ms(t), t["trace_id"], t["total_tokens"],
            t["total_cost_usd"], len(sp), f"{ok}/{len(tools)}",
        ))
    for wall, tid, tok, cost, nsp, tok_ok in sorted(rows, reverse=True):
        lines.append(
            f"| `{tid}` | {wall/1000:.1f} | {tok:,} | {cost:.6f} | {nsp} | {tok_ok} |"
        )
    lines += [
        "",
        "## 复现命令",
        "",
        "```bash",
        "python -m app.eval.metrics_report --out docs/measured_metrics.md",
        "```",
        "",
        "> 若 `data/traces/` 为空（如全新克隆），先跑一次 `python scripts/demo.py --mock`",
        "> 或 `python -u scripts/demo.py --query \"...\"` 生成 trace。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="从 trace 归档生成度量报告")
    ap.add_argument("--out", default="", help="输出 Markdown 路径（默认仅打印）")
    args = ap.parse_args()

    trace_dir = settings.trace_path
    traces = _load_traces(trace_dir)
    if not traces:
        print(f"未在 {trace_dir} 找到含 span 的 trace。请先运行 demo 生成数据。")
        return 1

    m = compute(traces)
    md = render(m, traces)

    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"\n✅ 已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
