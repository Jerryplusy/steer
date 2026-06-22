import os
import subprocess
import sys

from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.abspath(__file__))
EASYEDIT_DIR = os.path.join(ROOT, "EasyEdit")
DATA_DIR = os.path.join(ROOT, "data", "SteerEval")
REPO_URL = "https://github.com/zjunlp/EasyEdit.git"


def clone_easyedit() -> None:
    """克隆 EasyEdit 仓库；若已存在则跳过。"""
    if os.path.isdir(os.path.join(EASYEDIT_DIR, ".git")):
        print(f"[skip] EasyEdit 已存在: {EASYEDIT_DIR}")
        return
    print(f"[clone] {REPO_URL} -> {EASYEDIT_DIR}")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, EASYEDIT_DIR],
        check=True,
    )


def download_personality_dataset() -> None:
    """下载 SteerEval/personality 的 train.json 与 valid.json。

    注意：传给 hf_hub_download 的 filename 是 'personality/train.json'，
    若把 local_dir 设成 './data/SteerEval/personality' 会得到
    './data/SteerEval/personality/personality/train.json' 这种嵌套路径。
    所以 local_dir 必须是 './data/SteerEval'。
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
        clone_easyedit()
        download_personality_dataset()
    except subprocess.CalledProcessError as e:
        print(f"[error] git clone 失败: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[error] 数据下载失败: {e}", file=sys.stderr)
        return 1

    print("\n数据准备完成。")
    print(f"  - 仓库: {EASYEDIT_DIR}")
    print(f"  - 数据: {os.path.join(DATA_DIR, 'personality')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
