#!/bin/bash
# ===========================================================================
# 测试脚本
#
# 用法：
#   ./shell/test_pipeline.sh           # 默认 mps 设备，caa 方法
#   ./shell/test_pipeline.sh --device=cuda:0
#   ./shell/test_pipeline.sh --method=reps
# ===========================================================================
set -e

# ===========================
# 解析参数
# ===========================
EXTRA_ARGS=()
for arg in "$@"; do
    EXTRA_ARGS+=("$arg")
done

# ===========================
# 路径
# ===========================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EASYEDIT_DIR="$PROJECT_ROOT/EasyEdit"
DATA_LINK="$EASYEDIT_DIR/data"
TEST_DATA_DIR="$PROJECT_ROOT/data_test"
REAL_DATA_DIR="$PROJECT_ROOT/data"

echo "===== [1/4] 准备 5 个测试样本 ====="
python "$PROJECT_ROOT/shell/prepare_test_data.py" --n 5 --concept L1_1

echo ""
echo "===== [2/4] 切换 EasyEdit/data -> data_test/ ====="
if [ -L "$DATA_LINK" ] || [ -e "$DATA_LINK" ]; then
    # 把现有软链/目录改名备份（不是删，是改名以便恢复）
    BACKUP_PATH="$DATA_LINK.bak.$(date +%s)"
    mv "$DATA_LINK" "$BACKUP_PATH"
    echo "  备份原链接: $BACKUP_PATH"
fi
ln -s "$TEST_DATA_DIR" "$DATA_LINK"
echo "  $DATA_LINK -> $TEST_DATA_DIR"

echo ""
echo "===== [3/4] 跑 baseline ====="
set +e
"$PROJECT_ROOT/shell/steer_eval.sh" \
    --method=caa \
    --generate_vector=true \
    --generate_response=false \
    --evaluate=false \
    --layers=20 \
    --multipliers=3 \
    --gen_out_path=test_5samples \
    "${EXTRA_ARGS[@]}"
TEST_STATUS=$?
set -e

echo ""
echo "===== [4/4] 恢复 EasyEdit/data -> data ====="
if [ -L "$DATA_LINK" ]; then
    rm "$DATA_LINK"
fi

LATEST_BACKUP=$(ls -t "$DATA_LINK".bak.* 2>/dev/null | head -1)
if [ -n "$LATEST_BACKUP" ]; then
    mv "$LATEST_BACKUP" "$DATA_LINK"
    echo "  恢复: $DATA_LINK"
else
    ln -s "$REAL_DATA_DIR" "$DATA_LINK"
    echo "  重建: $DATA_LINK -> $REAL_DATA_DIR"
fi

echo ""
if [ $TEST_STATUS -eq 0 ]; then
    echo "✅ 测试通过"
else
    echo "❌ 测试失败 (exit $TEST_STATUS)"
fi
exit $TEST_STATUS
