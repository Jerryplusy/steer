"""
Steer 比赛结果打分脚本（CS / IS / FS / HM）

API 配置：项目根目录 config/scorer.yaml
  也可用环境变量覆盖：SCORER_API_KEY / SCORER_API_BASE / SCORER_MODEL 等

用法：
  python shell/score.py \\
      --input 参赛队伍名称_result.json \\
      --output score_results.json \\
      --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import yaml  # type: ignore

# ===========================================================================
# 配置加载（从 config/scorer.yaml，环境变量优先级最高）
# ===========================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "scorer.yaml")

# 代码内默认值
DEFAULTS: Dict[str, Any] = {
    "api_base": "https://api.deepseek.com/v1",
    "api_key": "YOUR_API_KEY_HERE",
    "model": "deepseek-chat",
    "temperature": 0,
    "max_tokens": 256,
    "timeout": 60,
    "max_retries": 3,
    "retry_backoff": 2.0,
}

# 环境变量名 -> yaml key 的映射
ENV_OVERRIDES = {
    "SCORER_API_BASE": ("api_base", str),
    "SCORER_API_KEY": ("api_key", str),
    "SCORER_MODEL": ("model", str),
    "SCORER_TEMPERATURE": ("temperature", float),
    "SCORER_MAX_TOKENS": ("max_tokens", int),
    "SCORER_TIMEOUT": ("timeout", int),
    "SCORER_MAX_RETRIES": ("max_retries", int),
    "SCORER_RETRY_BACKOFF": ("retry_backoff", float),
}


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """加载配置：环境变量 > yaml 文件 > 代码默认值。"""
    cfg: Dict[str, Any] = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    for env, (key, caster) in ENV_OVERRIDES.items():
        if env in os.environ:
            cfg[key] = caster(os.environ[env])
    return cfg


# 加载一次（脚本级常量），也允许命令行 --config 覆盖
_CFG = load_config()
SCORER_API_BASE: str = _CFG["api_base"]
SCORER_API_KEY: str = _CFG["api_key"]
SCORER_MODEL: str = _CFG["model"]
SCORER_TEMPERATURE: float = _CFG["temperature"]
SCORER_MAX_TOKENS: int = _CFG["max_tokens"]
SCORER_TIMEOUT: int = _CFG["timeout"]
SCORER_MAX_RETRIES: int = _CFG["max_retries"]
SCORER_RETRY_BACKOFF: float = _CFG["retry_backoff"]


# ===========================================================================
# 评分 prompt
# ===========================================================================

CS_PROMPT = """你是一个严格的评分员。请按以下标准对「模型的回答」是否准确表达「目标概念」打分（0-4 分）。

评分标准：
- 0 分：完全没有体现目标概念
- 1 分：隐约相关但几乎没有体现
- 2 分：有部分体现但不充分、表达生硬
- 3 分：较清晰地体现了目标概念，融入较自然
- 4 分：准确、充分、自然地体现了目标概念

目标概念：
{concept}

模型的回答：
{answer}

只输出一个 0-4 之间的整数，不要任何解释。
分数："""

IS_PROMPT = """你是一个严格的评分员。请按以下标准对「模型的回答」是否良好遵循了「用户指令」打分（0-4 分）。

评分标准：
- 0 分：答非所问，与指令完全无关
- 1 分：勉强相关但基本没回答问题
- 2 分：有回答但偏离主题或答得不完整
- 3 分：基本回答了问题
- 4 分：完整、准确地回答了问题

用户指令：
{instruction}

模型的回答：
{answer}

只输出一个 0-4 之间的整数，不要任何解释。
分数："""

FS_PROMPT = """你是一个严格的评分员。请按以下标准对「模型的回答」的语言质量打分（0-4 分）。

评分标准（只看语言本身，不管内容是否正确）：
- 0 分：完全不可读、严重重复、语法混乱
- 1 分：多处语法错误或不通顺，可读性差
- 2 分：有少量语法问题，整体可读
- 3 分：基本流畅，偶有小瑕疵
- 4 分：流畅、自然、毫无语言瑕疵

模型的回答：
{answer}

只输出一个 0-4 之间的整数，不要任何解释。
分数："""


# ===========================================================================
# 工具函数
# ===========================================================================
def harmonic_mean(cs: float, is_: float, fs: float) -> float:
    """HM = 3 / (1/CS + 1/IS + 1/FS)，任一项为 0 时返回 0。"""
    if min(cs, is_, fs) <= 0:
        return 0.0
    return 3.0 / (1.0 / cs + 1.0 / is_ + 1.0 / fs)


def parse_score(text: str) -> int:
    m = re.search(r"\b([0-4])\b", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\d+", text)
    if m:
        v = int(m.group(0))
        return max(0, min(4, v))
    return 0


def call_scorer(
    prompt: str,
    api_base: str = SCORER_API_BASE,
    api_key: str = SCORER_API_KEY,
    model: str = SCORER_MODEL,
) -> str:
    """调用大模型裁判（OpenAI 兼容 chat/completions）。

    支持 SDK 存在时走 openai / openai 兼容包；否则用 requests。
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, SCORER_MAX_RETRIES + 1):
        try:
            try:
                from openai import OpenAI  # type: ignore
                client = OpenAI(base_url=api_base, api_key=api_key, timeout=SCORER_TIMEOUT)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=SCORER_TEMPERATURE,
                    max_tokens=SCORER_MAX_TOKENS,
                )
                return (resp.choices[0].message.content or "").strip()
            except ImportError:
                import requests  # type: ignore
                r = requests.post(
                    f"{api_base.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": SCORER_TEMPERATURE,
                        "max_tokens": SCORER_MAX_TOKENS,
                    },
                    timeout=SCORER_TIMEOUT,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < SCORER_MAX_RETRIES:
                time.sleep(SCORER_RETRY_BACKOFF)
    raise RuntimeError(f"scorer 调用失败（重试 {SCORER_MAX_RETRIES} 次后仍失败）: {last_err}")


# ===========================================================================
# 打分核心
# ===========================================================================
def score_one(
    concept: Dict[str, Any],
    api_base: str = SCORER_API_BASE,
    api_key: str = SCORER_API_KEY,
    model: str = SCORER_MODEL,
) -> Dict[str, Any]:
    """对单个 concept 的所有 generated_results 打分，返回带分的结果。"""
    concept_name = concept.get("concept_name") or concept.get("concept_description") or ""
    scored_results: List[Dict[str, Any]] = []

    for g in concept.get("generated_results", []):
        # 取第一条 pred 作为 answer
        answers = g.get("pred") or g.get("complete_output") or []
        answer = answers[0] if answers else ""
        question = g.get("input", "")

        try:
            cs_text = call_scorer(CS_PROMPT.format(concept=concept_name, answer=answer),
                                  api_base, api_key, model)
            cs = parse_score(cs_text)
        except Exception as e:  # noqa: BLE001
            print(f"  [CS 失败] {concept.get('concept_id')}: {e}", file=sys.stderr)
            cs = 0

        try:
            is_text = call_scorer(IS_PROMPT.format(instruction=question, answer=answer),
                                  api_base, api_key, model)
            is_ = parse_score(is_text)
        except Exception as e:  # noqa: BLE001
            print(f"  [IS 失败] {concept.get('concept_id')}: {e}", file=sys.stderr)
            is_ = 0

        try:
            fs_text = call_scorer(FS_PROMPT.format(answer=answer),
                                  api_base, api_key, model)
            fs = parse_score(fs_text)
        except Exception as e:  # noqa: BLE001
            print(f"  [FS 失败] {concept.get('concept_id')}: {e}", file=sys.stderr)
            fs = 0

        scored_results.append({
            **g,
            "_scores": {"cs": cs, "is": is_, "fs": fs, "hm": harmonic_mean(cs, is_, fs)},
        })

    return {**concept, "generated_results": scored_results}


def aggregate(per_concept: List[Dict[str, Any]]) -> Dict[str, float]:
    """汇总所有 concept 的平均 HM。"""
    if not per_concept:
        return {"mean_hm": 0.0, "mean_cs": 0.0, "mean_is": 0.0, "mean_fs": 0.0, "n_concepts": 0}

    cs_list, is_list, fs_list, hm_list = [], [], [], []
    for c in per_concept:
        for g in c.get("generated_results", []):
            s = g.get("_scores", {})
            cs_list.append(s.get("cs", 0))
            is_list.append(s.get("is", 0))
            fs_list.append(s.get("fs", 0))
            hm_list.append(s.get("hm", 0))
    return {
        "mean_cs": sum(cs_list) / len(cs_list) if cs_list else 0,
        "mean_is": sum(is_list) / len(is_list) if is_list else 0,
        "mean_fs": sum(fs_list) / len(fs_list) if fs_list else 0,
        "mean_hm": sum(hm_list) / len(hm_list) if hm_list else 0,
        "n_concepts": len(per_concept),
        "n_samples": len(hm_list),
    }


# ===========================================================================
# CLI
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="convert.py 的输出")
    parser.add_argument("--output", "-o", default=None, help="带分数的结果 json 路径")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="并发 concept 数")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个 concept（调试用）")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help=f"评分模型配置文件路径（默认 {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--api-base", default=SCORER_API_BASE,
                        help="覆盖配置文件中的 api_base")
    parser.add_argument("--api-key", default=SCORER_API_KEY,
                        help="覆盖配置文件中的 api_key")
    parser.add_argument("--model", default=SCORER_MODEL,
                        help="覆盖配置文件中的 model")
    args = parser.parse_args()

    # 如果用户显式指定 --config，重新加载
    if args.config != DEFAULT_CONFIG_PATH and os.path.exists(args.config):
        cfg = load_config(args.config)
        if args.api_base == SCORER_API_BASE:  # 没显式覆盖
            args.api_base = cfg["api_base"]
        if args.api_key == SCORER_API_KEY:
            args.api_key = cfg["api_key"]
        if args.model == SCORER_MODEL:
            args.model = cfg["model"]

    if args.api_key == "YOUR_API_KEY_HERE":
        print(
            "⚠️  请设置 SCORER_API_KEY（环境变量）或在 config/scorer.yaml 里填 api_key",
            file=sys.stderr,
        )
        # 不直接退出，允许 dry-run / 看看其他东西

    with open(args.input, "r", encoding="utf-8") as f:
        submission = json.load(f)
    if args.limit:
        submission = submission[: args.limit]

    print(f"开始打分：{len(submission)} concepts, 并发={args.concurrency}, "
          f"model={args.model}, base={args.api_base}")

    scored: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(score_one, c, args.api_base, args.api_key, args.model): c
            for c in submission
        }
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001
                c = futures[fut]
                print(f"❌ concept {c.get('concept_id')} 打分失败: {e}", file=sys.stderr)
                continue
            scored.append(result)
            # 局部聚合
            agg = aggregate(scored)
            print(f"  [{n}/{len(submission)}] {result.get('concept_id')}: "
                  f"HM={agg['mean_hm']:.3f} (CS={agg['mean_cs']:.2f} "
                  f"IS={agg['mean_is']:.2f} FS={agg['mean_fs']:.2f})")

    # 保持原顺序
    order = {c.get("concept_id"): i for i, c in enumerate(submission)}
    scored.sort(key=lambda x: order.get(x.get("concept_id"), 0))

    final_agg = aggregate(scored)
    print("\n===== 总体结果 =====")
    print(f"  Concepts:   {final_agg['n_concepts']}")
    print(f"  Samples:    {final_agg['n_samples']}")
    print(f"  Mean CS:    {final_agg['mean_cs']:.4f}")
    print(f"  Mean IS:    {final_agg['mean_is']:.4f}")
    print(f"  Mean FS:    {final_agg['mean_fs']:.4f}")
    print(f"  Mean HM:    {final_agg['mean_hm']:.4f}")

    out_data = {
        "summary": final_agg,
        "config": {
            "api_base": args.api_base,
            "model": args.model,
            "concurrency": args.concurrency,
        },
        "results": scored,
    }
    out_path = args.output or "score_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细结果写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
