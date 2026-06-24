import argparse
import json
import os
import shutil

from data_utils import ensure_llm_description

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "data", "SteerEval", "personality")
DST_DIR = os.path.join(ROOT, "data_test", "SteerEval", "personality")


def filter_split(data, n_concepts, n_per_concept, concept_id):
    if concept_id:
        return [d for d in data if d["concept_id"] == concept_id][:n_per_concept]
    by_concept = {}
    for d in data:
        by_concept.setdefault(d["concept_id"], []).append(d)
    subset = []
    for cid in list(by_concept.keys())[:n_concepts]:
        subset.extend(by_concept[cid][:n_per_concept])
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="每个 concept 保留条数")
    parser.add_argument("--n_concepts", type=int, default=1, help="保留多少个 concept（concept 未指定时生效）")
    parser.add_argument("--concept", type=str, default=None, help="只保留指定 concept_id（优先于 n_concepts）")
    args = parser.parse_args()

    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)
    os.makedirs(DST_DIR, exist_ok=True)

    for split in ("train", "valid"):
        src = os.path.join(SRC_DIR, f"{split}.json")
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
        subset = filter_split(data, args.n_concepts, args.n, args.concept)
        subset = ensure_llm_description(subset)
        dst = os.path.join(DST_DIR, f"{split}.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        cids = sorted({d["concept_id"] for d in subset})
        print(f"  [ok] {split}: {len(subset)} samples, {len(cids)} concepts -> {dst}")
        print(f"        concepts: {cids}")

    print(f"\n测试数据目录: {DST_DIR}")


if __name__ == "__main__":
    main()
