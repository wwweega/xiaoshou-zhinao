"""
演示脚本 · 一键跑通全链路
============================================================
用法：
    python -m scripts.demo                  # 单任务演示
    python -m scripts.demo --full           # 多个任务演示（含记忆复用）
    python -m scripts.demo --mock           # 强制离线 mock 模式
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# 让脚本能直接找到 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.graph.workflow import runner  # noqa: E402
from app.memory.store import long_term_memory  # noqa: E402


BANNER = r"""
   _______  _   _  ___  _   _  _____
  |__   __|| | | ||_ _|| \ | ||  _  |
     | |   | |_| | | | |  \| || |_| |
     | |   |  _  | | | | . ` ||  _  |
     | |   | | | |_| |_| |\  || | | |
     |_|   |_| |_|___/|_| \_||_| |_|

  运营智脑 · 跨境电商 Multi-Agent 运营自动化平台
  ================================================
"""

QUERIES = [
    "帮我调研厨房收纳品类在亚马逊美国站的市场情况，分析 Top5 竞品的定价策略，"
    "并基于评论洞察写一份可直接使用的 Listing 和广告关键词方案",

    "生成本周店铺运营周报，包含营收、订单、转化率、广告表现，并给出下周的优化建议",

    "分析竞品评论里的用户痛点，指出我们的产品应该主打哪三个卖点",
]


async def run_one(query: str, idx: int) -> None:
    print("\n" + "=" * 70)
    print(f"【任务 {idx}】{query}")
    print("=" * 70)

    result = await runner.run(query)
    s = result.summary

    print("\n" + result.trace_text)

    print("\n【执行摘要】")
    print(f"  任务 ID     : {s.task_id}")
    print(f"  状态        : {s.status.value}")
    print(f"  反思轮次    : {s.reflection_rounds}")
    if s.plan:
        print(f"  子任务数    : {len(s.plan.subtasks)}")
        for st in s.plan.subtasks:
            print(f"    · [{st.id}] {st.name:<20} agent={st.agent.value:<12} "
                  f"status={st.status.value}")
    if s.critique:
        d = s.critique.dimensions
        print(f"  质量评分    : {s.critique.score:.2f}  "
              f"(相关 {d.get('relevance', 0):.2f} / 事实 {d.get('factuality', 0):.2f} / "
              f"合规 {d.get('compliance', 0):.2f} / 可读 {d.get('readability', 0):.2f})")
    print(f"  总 token    : {s.total_tokens}")
    print(f"  总成本      : ${s.total_cost_usd:.5f}")
    print(f"  端到端耗时  : {s.latency_ms:.0f} ms")
    if s.errors:
        print(f"  错误        : {s.errors}")

    print("\n【交付物预览（前 1200 字）】")
    content = s.deliverable.content if s.deliverable else "（无）"
    print(content[:1200])
    if len(content) > 1200:
        print(f"\n...（完整内容共 {len(content)} 字，已保存到 data/ 目录与 trace 文件）")


async def main() -> None:
    parser = argparse.ArgumentParser(description="运营智脑演示脚本")
    parser.add_argument("--full", action="store_true", help="跑多个任务，演示记忆复用")
    parser.add_argument("--mock", action="store_true", help="强制离线 mock 模式")
    parser.add_argument("--query", type=str, default="", help="自定义单条需求")
    args = parser.parse_args()

    if args.mock:
        os.environ["LLM_PROVIDER"] = "mock"
        settings.llm_provider = "mock"

    print(BANNER)
    print(f"  LLM provider : {settings.llm_provider}")
    print(f"  Mock 模式    : {settings.use_mock}")
    print(f"  大模型        : {settings.llm_model}")
    print(f"  小模型        : {settings.llm_small_model}")
    print(f"  工具数        : 12")
    print("=" * 70)

    if args.query:
        await run_one(args.query, 1)
    elif args.full:
        for i, q in enumerate(QUERIES, 1):
            await run_one(q, i)
            if i == 1:
                print(f"\n[记忆系统] 长期记忆当前条目数：{long_term_memory.size()}")
                print("[记忆系统] 注意第 3 个任务的召回效果 —— 这就是跨任务复用\n")
    else:
        await run_one(QUERIES[0], 1)

    print("\n" + "=" * 70)
    print("演示完成。下一步：")
    print("  1) 启动服务： python -m uvicorn app.main:app --port 8000")
    print("  2) 打开前端： http://localhost:8000")
    print("  3) 接口文档： http://localhost:8000/docs")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
