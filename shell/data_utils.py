from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def ensure_llm_description(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """为 list of items 中的每条加 `llm_description` 字段
    """
    for it in items:
        if "llm_description" not in it or not it["llm_description"]:
            it["llm_description"] = (
                it.get("concept_description")
                or it.get("domain_description")
                or it.get("concept")
                or ""
            )
    return items


def patch_file(path: str, *, verbose: bool = True) -> int:
    """读 json → 加 llm_description → 写回。返回被修改的条数"""
    if not os.path.exists(path):
        if verbose:
            print(f"  [skip] 不存在: {path}")
        return 0
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        if verbose:
            print(f"  [skip] 不是 list: {path}")
        return 0

    modified = 0
    for it in items:
        if "llm_description" not in it or not it["llm_description"]:
            it["llm_description"] = (
                it.get("concept_description")
                or it.get("domain_description")
                or it.get("concept")
                or ""
            )
            modified += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"  [ok] {path}: {modified}/{len(items)} items augmented")
    return modified


if __name__ == "__main__":
    """CLI 用法：python data_utils.py file1.json file2.json ..."""
    import sys
    for p in sys.argv[1:]:
        patch_file(p)
