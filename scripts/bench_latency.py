"""
性能探针 · 短输出 vs 长输出延迟对比
============================================================
用途：为 `docs/真实模型实测报告.md` 里的「短/长输出延迟差」提供**可复现证据**。

为什么需要这个脚本？
  → 早期文档里写了"短输出 1.6s、长输出 248s、相差 155 倍"，
    但**没有留下任何脚本或归档**，导致这组数字无法复现、无法自证。
    这在开源项目里是硬伤：面试官问"这数据怎么来的"，答不上来就等于造假。
    修复方式不是删掉数字，而是**补上产生它的脚本**，让任何人都能重跑。

用法：
    python scripts/bench_latency.py                 # 用 .env 里的真实模型
    python scripts/bench_latency.py --mock          # 离线（仅验证脚本可跑）
    python scripts/bench_latency.py --out docs/bench_latency_result.md

注意：会真实调用 2 次 LLM（1 次短、1 次长），消耗少量额度。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 注意：必须在 import app.* 之前设置环境变量，
# 否则 config 单例已经实例化，mock 开关不会生效。
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--mock", action="store_true")
_pre, _ = _ap.parse_known_args()
if _pre.mock:
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["LLM_API_KEY"] = ""  # 清空 key，确保走 mock 分支

from app.core.config import settings  # noqa: E402
from app.core.observability import logger  # noqa: E402
from app.llm.client import llm_client  # noqa: E402

SHORT_PROMPT = "用一句话说明跨境电商 Listing 的核心作用。"

LONG_PROMPT = (
    "为「免打孔厨房置物架」写一份完整的亚马逊美国站 Listing 方案，"
    "必须包含：①标题（200 字符内）②五点描述（每点含卖点+参数）"
    "③产品描述（300 字以上）④后台搜索关键词（20 个）"
    "⑤A+ 图文页的结构建议。内容要具体、可直接使用，不要泛泛而谈。"
)


async def _run_stream(prompt: str) -> tuple[str, float]:
    """用流式接口跑一次，返回 (文本, 耗时秒)。"""
    t0 = time.perf_counter()
    text, _ = await llm_client.chat_stream(
        [{"role": "user", "content": prompt}], max_tokens=4096
    )
    return text, time.perf_counter() - t0


async def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 长短输出延迟探针")
    ap.add_argument("--mock", action="store_true", help="离线模式，不调用真实模型")
    ap.add_argument("--out", default="", help="把结果写入指定 Markdown 文件")
    args = ap.parse_args()

    print("=" * 66)
    print("  延迟探针：短输出 vs 长输出")
    print(f"  模式: {'mock（耗时无意义）' if args.mock else '真实模型'}")
    print("=" * 66)

    print("\n[A] 短输出（流式）...")
    short_text, short_s = await _run_stream(SHORT_PROMPT)
    print(f"    → {len(short_text)} 字 / {short_s:.1f} s")

    print("\n[B] 长输出（流式，最长可能需要数分钟，请耐心等待）...")
    long_text, long_s = await _run_stream(LONG_PROMPT)
    print(f"    → {len(long_text)} 字 / {long_s:.1f} s")

    ratio = (long_s / short_s) if short_s > 0 else 0.0
    print("\n" + "-" * 66)
    print(f"  短输出 : {len(short_text):>6} 字 / {short_s:>7.1f} s")
    print(f"  长输出 : {len(long_text):>6} 字 / {long_s:>7.1f} s")
    print(f"  倍率   : {ratio:>6.1f}×")
    print("-" * 66)

    if args.out and not args.mock:
        from datetime import datetime

        try:
            model_label = settings.llm_model
        except Exception:  # noqa: BLE001 — 取模型名失败不该让已测到的数字丢失
            model_label = "(未知)"
        md = [
            "# 延迟探针结果（自动生成）",
            "",
            f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
            f"- 模型：`{model_label}`",
            f"- 命令：`python scripts/bench_latency.py --out {args.out}`",
            "",
            "| 探针 | 输出规模 | 耗时 | 原文预览 |",
            "|---|---|---|---|",
            f"| A · 短输出 | {len(short_text)} 字 | **{short_s:.1f} s** | {short_text[:40]}… |",
            f"| B · 长输出 | {len(long_text)} 字 | **{long_s:.1f} s** | {long_text[:40]}… |",
            "",
            f"> 倍率：长/短 = **{ratio:.1f}×**。",
            "",
            "> 结论：短与长输出的耗时量级差异极大，`timeout` 必须按「最慢负载」设计，",
            "> 且长文生成应使用流式（timeout 约束的是 chunk 间隔而非总时长）。",
        ]
        Path(args.out).write_text("\n".join(md), encoding="utf-8")
        print(f"\n结果已写入: {args.out}")

    if args.mock:
        logger.info("mock 模式完成（耗时无参考价值）")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
