"""
准备 N 个测试训练样本 + 对应 N 个验证样本。
"""
import argparse
import json
import os
import shutil

from data_utils import ensure_llm_description

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "data", "SteerEval", "personality")
DST_DIR = os.path.join(ROOT, "data_test", "SteerEval", "personality")


def filter_split(data, n, concept_id):
    subset = [d for d in data if d["concept_id"] == concept_id][:n]
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="每 split 保留条数")
    parser.add_argument("--concept", type=str, default="L1_1", help="保留哪个 concept_id")
    args = parser.parse_args()

    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)
    os.makedirs(DST_DIR, exist_ok=True)

    for split in ("train", "valid"):
        src = os.path.join(SRC_DIR, f"{split}.json")
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
        subset = filter_split(data, args.n, args.concept)
        subset = ensure_llm_description(subset)
        dst = os.path.join(DST_DIR, f"{split}.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"  [ok] {split}: {len(subset)} samples -> {dst}")
        if subset:
            print(f"        concept_id={subset[0]['concept_id']}, "
                  f"concept='{subset[0]['concept']}'")
            has_field = "llm_description" in subset[0]
            print(f"        llm_description: {has_field}")

    print(f"\n测试数据目录: {DST_DIR}")


if __name__ == "__main__":
    main()
