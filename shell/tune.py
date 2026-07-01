"""
Steer 调参脚本：三阶段 Grid Search
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHELL_DIR = PROJECT_ROOT / "shell"
STEER_EVAL_SH = SHELL_DIR / "steer_eval.sh"
CONVERT_PY = SHELL_DIR / "convert.py"
SCORE_PY = SHELL_DIR / "score.py"
DATA_UTILS = SHELL_DIR / "data_utils.py"

def ensure_full_data_has_llm_description() -> None:

    import sys
    sys.path.insert(0, str(SHELL_DIR))
    from data_utils import patch_file

    data_dir = PROJECT_ROOT / "data" / "SteerEval" / "personality"
    for split in ("train", "valid"):
        path = data_dir / f"{split}.json"
        if path.exists():
            patch_file(str(path), verbose=True)

ensure_full_data_has_llm_description()

# 默认数据集
DEFAULT_DATASET = "SteerEval/personality"
DEFAULT_DEVICE = "mps"
DEFAULT_DTYPE = "float16"
DEFAULT_EXP = "valid"
DEFAULT_MAX_NEW_TOKENS = 128
GROUP_TIMEOUT = 60 * 60

# ---- 阶段定义 ----
PHASE1_LAYERS = [18, 22, 26, 30]
PHASE1_MULTS = [1.5, 2.0, 2.5, 3.0]

PHASE3_LAYERS = [20, 22, 24]
PHASE3_MULTS = [2, 3, 4]


# ===========================================================================
# 结果存储
# ===========================================================================
def tune_dir(method: str) -> Path:
    return PROJECT_ROOT / "output" / "tune" / method


def results_csv_path(method: str) -> Path:
    return tune_dir(method) / "results.csv"


def results_json_path(method: str) -> Path:
    return tune_dir(method) / "results.json"


def phase_state_path(method: str, phase: str) -> Path:
    return tune_dir(method) / f"phase_{phase}.json"


def load_results(method: str) -> List[Dict[str, Any]]:
    """读取历史所有 (L, M) 跑过的结果。"""
    p = results_json_path(method)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def save_results(method: str, rows: List[Dict[str, Any]]) -> None:
    tune_dir(method).mkdir(parents=True, exist_ok=True)
    results_json_path(method).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not rows:
        return
    # 同步写 CSV
    p = results_csv_path(method)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def upsert_result(method: str, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """插入或更新一条结果（按 L/M 唯一键）。"""
    rows = load_results(method)
    key = (row["layer"], row["multiplier"])
    rows = [r for r in rows if (r["layer"], r["multiplier"]) != key]
    rows.append(row)
    rows.sort(key=lambda r: -(r.get("mean_hm") or 0))
    save_results(method, rows)
    return rows


# ===========================================================================
# 路径生成
# ===========================================================================
def generation_dir(method: str, layer: int, mult: float, gen_out_path: str) -> Path:
    """steer_eval.sh 输出的 generation 目录。"""
    return (
        PROJECT_ROOT
        / "output"
        / "generation"
        / "qwen3-4b"
        / DEFAULT_DATASET
        / gen_out_path
        / method
        / f"layer_{layer}_multip_{mult}"
    )


def score_json_path(method: str, layer: int, mult: float, gen_out_path: str) -> Path:
    """score.py 的输出。"""
    return PROJECT_ROOT / "output" / "evaluation" / "qwen3-4b" / DEFAULT_DATASET / gen_out_path / method / f"layer_{layer}_multip_{mult}_scores.json"


# ===========================================================================
# 单组 (L, M) 的运行 + 评分
# ===========================================================================
def _kill_proc_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    proc.kill()
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    subprocess.run(
        "pkill -9 -f 'examples/steer_eval.py'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        "pkill -9 -f 'shell/steer_eval.sh'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(3)


def run_one(
    method: str,
    layer: int,
    mult: float,
    gen_out_path: str,
    device: str = DEFAULT_DEVICE,
    dtype: str = DEFAULT_DTYPE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    skip_run: bool = False,
    score_concurrency: int = 8,
) -> Dict[str, Any]:
    """跑一组 (L, M)：steer_eval → convert → score。返回 results 字典。"""
    out_dir = generation_dir(method, layer, mult, gen_out_path)
    score_path = score_json_path(method, layer, mult, gen_out_path)
    result_json = (
        PROJECT_ROOT / "output" / "submission" / "qwen3-4b" / DEFAULT_DATASET
        / gen_out_path / f"L{layer}_M{mult}_result.json"
    )

    # ---- 1) steer_eval ----
    if not skip_run:
        cmd = [
            "bash", str(STEER_EVAL_SH),
            f"--device={device}",
            f"--dtype={dtype}",
            f"--method={method}",
            "--generate_vector=true",
            "--generate_response=true",
            "--generate_orig_output=false",
            "--evaluate=false",
            f"--layers={layer}",
            f"--multipliers={mult}",
            f"--gen_out_path={gen_out_path}",
            f"--exp={DEFAULT_EXP}",
            f"--max_new_tokens={max_new_tokens}",
        ]
        print(f"\n[RUN] L={layer} M={mult} -> steer_eval.sh")
        print("      " + " ".join(cmd[:6]) + " ...")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, start_new_session=True)
        try:
            rc = proc.wait(timeout=GROUP_TIMEOUT)
        except subprocess.TimeoutExpired:
            dt = time.time() - t0
            print(f"超时({dt/60:.1f} min)，杀进程组并跳过该组")
            _kill_proc_tree(proc)
            return {"layer": layer, "multiplier": mult, "status": "timeout"}
        dt = time.time() - t0
        print(f"      done in {dt/60:.1f} min, rc={rc}")
        if rc != 0:
            return {"layer": layer, "multiplier": mult, "status": "steer_eval_failed", "rc": rc}

    # 找 generation 产物
    gen_files = list(out_dir.glob("all_generation_results_*.json"))
    if not gen_files:
        return {"layer": layer, "multiplier": mult, "status": "no_generation_file"}
    src = gen_files[0]

    # ---- 2) convert ----
    print(f"[CONVERT] {src.name} -> {result_json.name}")
    rc = subprocess.run([
        "python", str(CONVERT_PY),
        "--input", str(src),
        "--output", str(result_json),
        "--team", f"{gen_out_path}_L{layer}_M{mult}",
    ]).returncode
    if rc != 0 or not result_json.exists():
        return {"layer": layer, "multiplier": mult, "status": "convert_failed"}

    # ---- 3) score ----
    score_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SCORE] -> {score_path.relative_to(PROJECT_ROOT)}")
    rc = subprocess.run([
        "python", str(SCORE_PY),
        "--input", str(result_json),
        "--output", str(score_path),
        "--concurrency", str(score_concurrency),
    ]).returncode
    if rc != 0 or not score_path.exists():
        return {"layer": layer, "multiplier": mult, "status": "score_failed"}

    # ---- 4) 读分数 ----
    score_data = json.loads(score_path.read_text(encoding="utf-8"))
    summary = score_data.get("summary", {})
    return {
        "layer": layer,
        "multiplier": mult,
        "status": "ok",
        "mean_hm": summary.get("mean_hm", 0),
        "mean_cs": summary.get("mean_cs", 0),
        "mean_is": summary.get("mean_is", 0),
        "mean_fs": summary.get("mean_fs", 0),
        "n_concepts": summary.get("n_concepts", 0),
        "n_samples": summary.get("n_samples", 0),
        "ts": datetime.now().isoformat(timespec="seconds"),
        "gen_out_path": gen_out_path,
        "score_path": str(score_path.relative_to(PROJECT_ROOT)),
    }


# ===========================================================================
# 阶段实现
# ===========================================================================
def phase_grid(
    method: str,
    layers: List[int],
    mults: List[float],
    phase_name: str,
    gen_out_path: str,
    device: str = DEFAULT_DEVICE,
    dtype: str = DEFAULT_DTYPE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    skip_run: bool = False,
) -> List[Dict[str, Any]]:
    """通用 grid search：遍历所有 (L, M) 组合。"""
    todo = [(L, M) for L in layers for M in mults]
    # 去重 + 过滤掉已有结果的
    existing = {(r["layer"], r["multiplier"]): r for r in load_results(method)}
    pending = [(L, M) for L, M in todo if (L, M) not in existing]
    print(f"\n===== {phase_name} for {method} =====")
    print(f"  total: {len(todo)} | already done: {len(todo) - len(pending)} | pending: {len(pending)}")
    if not pending:
        print("  (all done, skip)")
        return list(existing.values())

    # 保存本阶段的网格
    phase_state_path(method, phase_name).parent.mkdir(parents=True, exist_ok=True)
    phase_state_path(method, phase_name).write_text(
        json.dumps({"layers": layers, "mults": mults, "gen_out_path": gen_out_path}, indent=2),
        encoding="utf-8",
    )

    for i, (L, M) in enumerate(pending, 1):
        print(f"\n----- [{i}/{len(pending)}] L={L} M={M} -----")
        row = run_one(method, L, M, gen_out_path, device=device, dtype=dtype,
                      max_new_tokens=max_new_tokens, skip_run=skip_run)
        rows = upsert_result(method, row)
        if row["status"] == "ok":
            print(f"  → HM={row['mean_hm']:.4f}  (CS={row['mean_cs']:.2f} "
                  f"IS={row['mean_is']:.2f} FS={row['mean_fs']:.2f})")
        else:
            print(f"  ✗ status={row['status']}")

    return load_results(method)


def phase1(method: str, skip_run: bool = False, device: str = DEFAULT_DEVICE,
           dtype: str = DEFAULT_DTYPE, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> List[Dict[str, Any]]:
    return phase_grid(
        method, PHASE1_LAYERS, PHASE1_MULTS, "phase1",
        gen_out_path=f"tune_{method}_phase1", skip_run=skip_run,
        device=device, dtype=dtype, max_new_tokens=max_new_tokens,
    )


def phase2(method: str, skip_run: bool = False, device: str = DEFAULT_DEVICE,
           dtype: str = DEFAULT_DTYPE, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
           seeds: str = "") -> List[Dict[str, Any]]:
    if seeds:
        expanded = []
        for part in seeds.split(","):
            part = part.strip()
            if not part:
                continue
            L_s, M_s = part.split(":")
            expanded.append({"layer": int(L_s), "multiplier": float(M_s)})
        print(f"  phase2 手动 seeds: {[(r['layer'], r['multiplier']) for r in expanded]}")
    else:
        rows = load_results(method)
        valid = [r for r in rows if r.get("status") == "ok"]
        if len(valid) < 1:
            print("❌ phase2 需要 phase1 至少 1 个 ok 结果（或用 --seeds 手动指定）")
            return rows
        top3 = sorted(valid, key=lambda r: -r["mean_hm"])[:3]
        seen_layers = set()
        expanded = []
        for r in top3:
            if r["layer"] in seen_layers:
                continue
            seen_layers.add(r["layer"])
            expanded.append(r)
        print(f"  phase2 top(去重后): {[(r['layer'], r['multiplier']) for r in expanded]}")

    layers: List[int] = []
    mults: List[float] = []
    for r in expanded:
        L0, M0 = r["layer"], r["multiplier"]
        for dL in (-1, 0, 1):
            L = L0 + dL
            if L not in layers and 8 <= L <= 34:
                layers.append(L)
        for dM in (-0.5, 0, 0.5):
            M = round(M0 + dM, 1)
            if M > 0 and M not in mults and M <= 10:
                mults.append(M)

    layers.sort()
    mults.sort()
    print(f"  phase2 top(去重后): {[(r['layer'], r['multiplier']) for r in expanded]}")
    print(f"  phase2 grid: layers={layers}  mults={mults}  (= {len(layers)*len(mults)} 组)")
    return phase_grid(
        method, layers, mults, "phase2",
        gen_out_path=f"tune_{method}_phase2", skip_run=skip_run,
        device=device, dtype=dtype, max_new_tokens=max_new_tokens,
    )


def phase3(method: str, skip_run: bool = False) -> List[Dict[str, Any]]:
    """per-concept 细搜：每个 concept 单独选最优 (L, M)。
    """
    print("\n===== phase3 (per-concept 细搜) =====")
    return load_results(method)


# ===========================================================================
# 分析
# ===========================================================================
def analyze(method: str) -> None:
    """找最差 10 个 concept，给出按 concept 排序的坏 case 报告。"""
    rows = [r for r in load_results(method) if r.get("status") == "ok"]
    if not rows:
        print("❌ 没有可分析的结果")
        return
    best = max(rows, key=lambda r: r["mean_hm"])
    print(f"\n===== Analyze: {method} =====")
    print(f"  best: L={best['layer']} M={best['multiplier']} HM={best['mean_hm']:.4f}")
    print(f"  score file: {best.get('score_path')}")

    score_data = json.loads(
        (PROJECT_ROOT / best["score_path"]).read_text(encoding="utf-8")
    )
    # 聚合每个 concept 的平均 HM
    per_concept: Dict[str, List[float]] = {}
    for c in score_data.get("results", []):
        cid = c.get("concept_id", "?")
        hms = [g.get("_scores", {}).get("hm", 0) for g in c.get("generated_results", [])]
        per_concept.setdefault(cid, []).extend(hms)
    agg = [
        (cid, sum(hms) / len(hms), len(hms))
        for cid, hms in per_concept.items()
    ]
    agg.sort(key=lambda x: x[1])

    print(f"\n  最差 10 个 concept (在 best (L, M) 下):")
    print(f"  {'concept_id':<10} {'HM':>6}  {'n':>4}")
    print(f"  {'-'*10} {'-'*6}  {'-'*4}")
    for cid, hm, n in agg[:10]:
        print(f"  {cid:<10} {hm:>6.3f}  {n:>4}")

    print(f"\n  最好 5 个 concept:")
    for cid, hm, n in agg[-5:][::-1]:
        print(f"  {cid:<10} {hm:>6.3f}  {n:>4}")

    print(f"\n  启示：")
    print(f"  - 极难的 concept（HM<0.5）可能需要单独选 layer/M（见 phase3）")
    print(f"  - 极容易的 concept（HM>1.5）可以减 multiplier 省 FS 损耗")
    print(f"  - 整体都低 → 换方法（caa → reps）")


def best(method: str) -> None:
    """打印 best (L, M) 摘要。"""
    rows = [r for r in load_results(method) if r.get("status") == "ok"]
    if not rows:
        print("❌ 没有 ok 结果")
        return
    rows.sort(key=lambda r: -r["mean_hm"])
    print(f"\n===== Best for {method} =====")
    print(f"  {'rank':<5} {'L':>4} {'M':>5}  {'HM':>6}  {'CS':>5}  {'IS':>5}  {'FS':>5}")
    print(f"  {'-'*5} {'-'*4} {'-'*5}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}")
    for i, r in enumerate(rows[:10], 1):
        print(f"  {i:<5} {r['layer']:>4} {r['multiplier']:>5}  "
              f"{r['mean_hm']:>6.3f}  {r['mean_cs']:>5.2f}  "
              f"{r['mean_is']:>5.2f}  {r['mean_fs']:>5.2f}")


def compare(methods: List[str]) -> None:
    """横向对比多个方法。"""
    print(f"\n===== Compare: {', '.join(methods)} =====")
    print(f"  {'method':<10}  {'best L':>6} {'best M':>6}  {'best HM':>8}  {'n runs':>6}")
    print(f"  {'-'*10}  {'-'*6} {'-'*6}  {'-'*8}  {'-'*6}")
    for m in methods:
        rows = [r for r in load_results(m) if r.get("status") == "ok"]
        if not rows:
            print(f"  {m:<10}  {'-':>6} {'-':>6}  {'-':>8}  {0:>6}")
            continue
        b = max(rows, key=lambda r: r["mean_hm"])
        print(f"  {m:<10}  {b['layer']:>6} {b['multiplier']:>6}  "
              f"{b['mean_hm']:>8.4f}  {len(rows):>6}")


# ===========================================================================
# 入口
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--method", default="caa")
        p.add_argument("--device", default=DEFAULT_DEVICE)
        p.add_argument("--dtype", default=DEFAULT_DTYPE)
        p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
        p.add_argument("--skip-run", action="store_true",
                       help="跳过 steer_eval，只读已有产物 + 跑 convert/score")

    p1 = sub.add_parser("phase1", help="粗搜：5L × 4M = 20 组")
    add_common(p1)

    p2 = sub.add_parser("phase2", help="局部搜：围绕 phase1 top-3 邻居")
    add_common(p2)
    p2.add_argument("--seeds", default="",
                    help="手动指定起点 L:M,L:M,...（如 16:4.0,18:3.0,26:2.5）"
                         "，覆盖默认 top-3；子集变化旧分数不可比时用")

    p3 = sub.add_parser("phase3", help="per-concept 细搜（需支持单 concept 模式）")
    add_common(p3)

    pa = sub.add_parser("analyze", help="在 best (L, M) 下找最差 10 个 concept")
    pa.add_argument("--method", default="caa")

    pb = sub.add_parser("best", help="打印 best 排名")
    pb.add_argument("--method", default="caa")

    pc = sub.add_parser("compare", help="横向对比多个方法")
    pc.add_argument("--methods", required=True, help="逗号分隔，如 caa,reps")

    args = parser.parse_args()

    skip_run = getattr(args, "skip_run", False) or os.environ.get("TUNE_SKIP_RUN") == "true"

    if args.cmd == "phase1":
        phase1(args.method, skip_run=skip_run, device=args.device,
               dtype=args.dtype, max_new_tokens=args.max_new_tokens)
        best(args.method)
    elif args.cmd == "phase2":
        phase2(args.method, skip_run=skip_run, device=args.device,
               dtype=args.dtype, max_new_tokens=args.max_new_tokens,
               seeds=getattr(args, "seeds", ""))
        best(args.method)
    elif args.cmd == "phase3":
        phase3(args.method, skip_run=skip_run)
    elif args.cmd == "analyze":
        analyze(args.method)
    elif args.cmd == "best":
        best(args.method)
    elif args.cmd == "compare":
        compare(args.methods.split(","))
    return 0


if __name__ == "__main__":
    sys.exit(main())
