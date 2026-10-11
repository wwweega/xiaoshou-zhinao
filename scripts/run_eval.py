"""
评测脚本 · 跑一遍完整评测并生成报告
============================================================
用法：
    python -m scripts.run_eval                  # 全量评测（含 LLM judge 抽样）
    python -m scripts.run_eval --no-judge       # 只跑确定性 + 规则指标（快、免费）
    python -m scripts.run_eval --limit 5        # 只跑前 5 条（调试用）
    python -m scripts.run_eval --mock           # 离线 mock 模式评测

输出：docs/eval_report.md（可直接贴进 README 或作为简历附件）
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import BASE_DIR, settings  # noqa: E402
from app.eval.harness import Evaluator, default_test_suite  # noqa: E402
from app.graph.workflow import runner  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description="运营智脑评测")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 裁判")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    parser.add_argument("--mock", action="store_true", help="强制 mock 模式")
    parser.add_argument("--out", type=str, default="docs/eval_report.md")
    args = parser.parse_args()

    if args.mock:
        os.environ["LLM_PROVIDER"] = "mock"
        settings.llm_provider = "mock"

    cases = default_test_suite()
    if args.limit:
        cases = cases[: args.limit]

    print("=" * 70)
    print(f"  运营智脑 · 评测  |  用例 {len(cases)} 条  |  "
          f"judge={'off' if args.no_judge else 'on(抽样30%)'}  |  "
          f"mock={settings.use_mock}")
    print("=" * 70)

    evaluator = Evaluator(enable_judge=not args.no_judge, judge_sample_ratio=0.3)

    async def run_one(query: str):
        r = await runner.run(query)
        return r.summary

    report = await evaluator.evaluate(cases, run_one)

    md = Evaluator.render(report)
    out_path = BASE_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    print("\n" + md)
    print("\n" + "=" * 70)
    print(f"  报告已保存：{out_path}")
    is_mock = (args.mock or os.getenv("LLM_PROVIDER", "").lower() == "mock")
    gate_ok = report.gate_passed()
    if is_mock and not gate_ok:
        print("  CI 门禁：⚠️ mock 模式不达标（占位文本所致，非质量退化）→ 不阻断 CI")
        sys.exit(0)
    print(f"  CI 门禁：{'✅ 通过' if gate_ok else '❌ 未通过'}")
    print("=" * 70)

    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
