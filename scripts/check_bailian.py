"""
百炼连通性自检脚本
-----------------
用途：在正式跑项目前，用最小成本确认「Key 有效 + base_url 正确 + 模型名可用 +
      Function Calling 支持 + Structured Output 支持」四件事。

用法：
    export DASHSCOPE_API_KEY=sk-xxxx
    python scripts/check_bailian.py
    # 或者指定模型
    python scripts/check_bailian.py --model qwen3.8-flash

设计要点（面试可讲）：
    上线前的 "smoke test" 不该只 ping 一下 /models，而要**逐项验证项目真正依赖的能力**。
    本项目依赖三项能力，缺任何一个都会在运行中期才暴露：
      1. 基本对话      → 不通说明 Key/地址/模型名有错
      2. Function Calling → 不通则 Agent 的 Researcher 层直接废掉
      3. JSON 结构化输出  → 不通则 Planner 的计划解析会大面积失败
    所以自检要覆盖这三项，而不是只测连通。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("缺少依赖：pip install openai")
    sys.exit(2)


BASE_URL_DEFAULT = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _client(base_url: str) -> OpenAI:
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
    if not key or not key.startswith("sk-"):
        print("✗ 未检测到有效 API Key。请先执行：export DASHSCOPE_API_KEY=sk-xxxx")
        sys.exit(2)
    return OpenAI(api_key=key, base_url=base_url, timeout=60.0)


def step(title: str):
    print(f"\n{'─' * 56}\n▶ {title}\n{'─' * 56}")


def check_basic(client: OpenAI, model: str) -> bool:
    """1) 基本对话：验证 Key / base_url / model 三者匹配。"""
    step(f"[1/3] 基本对话  ·  model={model}")
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "只回复两个字：可用"}],
            max_tokens=32,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 调用失败：{type(exc).__name__}: {exc}")
        print("  排查：① Key 是否过期 ② 模型名是否在百炼已开通 ③ 地域端点是否正确")
        return False

    cost_ms = (time.perf_counter() - t0) * 1000
    text = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    print(f"✓ 返回内容：{text}")
    print(f"✓ 延迟：{cost_ms:.0f} ms")
    print(f"✓ Token 用量：输入 {usage.prompt_tokens} / 输出 {usage.completion_tokens}")
    return True


def check_function_calling(client: OpenAI, model: str) -> bool:
    """2) Function Calling：Agent 的工具调用能力，项目核心依赖。"""
    step(f"[2/3] Function Calling  ·  model={model}")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_inventory",
                "description": "查询某个 SKU 的当前库存数量",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sku": {"type": "string", "description": "商品 SKU 编码"},
                    },
                    "required": ["sku"],
                },
            },
        }
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "帮我查一下 SKU 为 ABC-123 的库存有多少"}
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 调用失败：{type(exc).__name__}: {exc}")
        return False

    msg = resp.choices[0].message
    calls = getattr(msg, "tool_calls", None)
    if not calls:
        print("⚠ 模型没有发起工具调用（未报错，但也没调用）")
        print("  如果模型名是带 thinking 的版本，可尝试在 extra_body 里关掉深度思考")
        return False

    call = calls[0]
    print(f"✓ 发起工具调用：{call.function.name}")
    print(f"✓ 参数：{call.function.arguments}")
    try:
        args = json.loads(call.function.arguments)
        assert args.get("sku") == "ABC-123", f"参数抽取错误：{args}"
        print("✓ 参数抽取正确（sku=ABC-123）")
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 参数解析异常：{exc}")
        return False
    return True


def check_structured_output(client: OpenAI, model: str) -> bool:
    """3) JSON 结构化输出：Planner 计划解析依赖，项目核心依赖。"""
    step(f"[3/3] JSON 结构化输出  ·  model={model}")
    prompt = (
        "把这句话拆成 JSON：{'subtasks': [{'id': 't1', 'desc': '...'}]}。"
        "内容：先抓取竞品价格，再分析趋势，最后写报告。"
        "只输出 JSON，不要任何解释文字和 markdown 代码块。"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是 JSON 生成器，只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 调用失败（可能该模型不支持 response_format）：{type(exc).__name__}: {exc}")
        print("  说明：项目内置 extract_json() 也能兜底解析，但原生支持更稳。")
        return False

    raw = (resp.choices[0].message.content or "").strip()
    print(f"✓ 原始返回（前 200 字）：{raw[:200]}")
    try:
        data = json.loads(raw)
        tasks = data.get("subtasks", [])
        print(f"✓ JSON 合法，解析出 {len(tasks)} 个子任务")
    except json.JSONDecodeError as exc:
        print(f"✗ JSON 非法：{exc}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="阿里云百炼连通性自检")
    parser.add_argument("--model", default="qwen3.8-flash", help="要检测的模型名")
    parser.add_argument("--base-url", default=BASE_URL_DEFAULT, help="OpenAI 兼容端点")
    args = parser.parse_args()

    print("=" * 56)
    print("  运营智脑 · 阿里云百炼连通性自检")
    print("=" * 56)
    print(f"  base_url : {args.base_url}")
    print(f"  model    : {args.model}")

    client = _client(args.base_url)

    results = {
        "基本对话": check_basic(client, args.model),
        "Function Calling": check_function_calling(client, args.model),
        "JSON 结构化输出": check_structured_output(client, args.model),
    }

    print(f"\n{'=' * 56}")
    print("  自检结果汇总")
    print("=" * 56)
    for name, ok in results.items():
        print(f"  [{'✓' if ok else '✗'}] {name}")

    passed = sum(results.values())
    total = len(results)
    print(f"\n  通过 {passed}/{total}")

    if passed == total:
        print("\n  ✓ 全部通过，可以把模型名写入 .env 正式跑项目了。")
        print("    cp .env.bailian.example .env  →  填 Key  →  python scripts/demo.py")
        return 0

    print("\n  ✗ 存在未通过项，详见上方逐项输出。")
    print("    建议先用 qwen3.8-flash 验证，它是百炼上能力最全、成本最低的型号。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
