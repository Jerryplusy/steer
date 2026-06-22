"""
把 steer_eval 的输出（tmp_generation_results_*.json / all_generation_results_*.json）
转成比赛要求提交的 result.json 格式。

用法：
    python shell/convert.py \\
        --input output/generation/qwen3-4b/SteerEval/personality/caa_baseline/caa/layer_20_multip_3/all_generation_results_valid.json \\
        --output 参赛队伍名称_result.json
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List


def convert_one(item: Dict[str, Any]) -> Dict[str, Any]:
    """单条 concept 的格式转换。"""
    # 字段映射：llm_description → concept_description
    return {
        "concept_id": item.get("concept_id", ""),
        "concept_name": item.get("concept_name", ""),
        "concept_description": item.get("concept_description") or item.get("llm_description") or "",
        "generation_prompt": item.get("generation_prompt", None),
        "generated_results": item.get("generated_results", []),
    }


def validate(submission: List[Dict[str, Any]], min_results: int = 5) -> None:
    """最低限度的格式校验：每个 concept 必须有 min_results 条 generated_results。"""
    for i, c in enumerate(submission):
        cid = c.get("concept_id", f"#{i}")
        n = len(c.get("generated_results", []))
        if n < min_results:
            print(
                f"⚠️  concept {cid} 只有 {n} 条 generated_results（期望 ≥ {min_results}）",
                file=sys.stderr,
            )
        for j, g in enumerate(c.get("generated_results", [])):
            if "input" not in g or "pred" not in g:
                print(
                    f"⚠️  concept {cid} 的第 {j} 条缺少 input/pred 字段",
                    file=sys.stderr,
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", required=True,
        help="steer_eval 输出的 generation results json 路径",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="输出的 result.json 路径（默认：<input>_submission.json）",
    )
    parser.add_argument(
        "--team", "-t", default="steer_team",
        help="参赛队伍名（用于默认输出文件名）",
    )
    parser.add_argument(
        "--min-results", type=int, default=5,
        help="每个 concept 最少需要多少条 generated_results（默认 5）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        print(f"❌ 输入文件格式不是 list（每条 concept 一项），实际: {type(raw)}",
              file=sys.stderr)
        return 1

    submission = [convert_one(item) for item in raw]
    validate(submission, args.min_results)

    if args.output is None:
        args.output = f"{args.team}_result.json"

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成: {len(submission)} concepts -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
