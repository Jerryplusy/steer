import os
import subprocess
import sys

from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "SteerEval")


def download_personality_dataset() -> None:
    """下载 SteerEval/personality 的 train.json 与 valid.json。
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    for split in ("train.json", "valid.json"):
        rel_path = f"personality/{split}"
        local_path = os.path.join(DATA_DIR, rel_path)
        if os.path.exists(local_path):
            print(f"[skip] 已存在: {local_path}")
            continue
        print(f"[download] {rel_path}")
        hf_hub_download(
            repo_id="zjunlp/SteerEval",
            filename=rel_path,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )


def main() -> int:
    try:
        download_personality_dataset()
    except Exception as e:  # noqa: BLE001
        print(f"[error] 数据下载失败: {e}", file=sys.stderr)
        return 1

    print("\n数据准备完成。")
    print(f"  - 数据: {os.path.join(DATA_DIR, 'personality')}")
    try:
        sys.path.insert(0, os.path.join(ROOT, "shell"))
        from shell.data_utils import patch_file
        print("\n[patch] 注入 llm_description 字段...")
        for split in ("train", "valid"):
            patch_file(os.path.join(DATA_DIR, "personality", f"{split}.json"))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] patch llm_description 失败: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
