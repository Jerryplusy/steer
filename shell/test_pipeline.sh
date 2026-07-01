#!/bin/bash
# ===========================================================================
#
# 用法：
#   ./shell/test_pipeline.sh                              # mps + caa, multip=1（smoke）
#   ./shell/test_pipeline.sh --device=cuda:0              # 改设备
#   ./shell/test_pipeline.sh --method=reps                # 当前仅支持 caa;非 caa 会被拒绝
#   ./shell/test_pipeline.sh --layers=22 --multipliers=2  # 改超参
#   SKIP_SCORE=true ./shell/test_pipeline.sh              # 跳过打分
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
TEST_DATA_DIR="$PROJECT_ROOT/data_test"
REAL_DATA_DIR="$PROJECT_ROOT/data"
GEN_OUT_PATH="test_5samples"

# 输出路径（与 steer_eval.sh 的 generation_output_dir 拼接规则一致）
GEN_DIR="$PROJECT_ROOT/output/generation/qwen3-4b/SteerEval/personality/${GEN_OUT_PATH}"
SUBMISSION_JSON="$PROJECT_ROOT/output/submission/qwen3-4b/SteerEval/personality/${GEN_OUT_PATH}_result.json"
SCORE_JSON="$PROJECT_ROOT/output/evaluation/qwen3-4b/SteerEval/personality/${GEN_OUT_PATH}_scores.json"

SKIP_SCORE="${SKIP_SCORE:-false}"

# ===========================
# 1. 准备测试数据
# ===========================
echo "===== [1/6] 准备 5 个测试样本 ====="
python "$PROJECT_ROOT/shell/prepare.py" --n 5 --concept L1_1

# ===========================
# 2. 通过 STEER_DATA_DIR 让 steer_eval 用 data_test/
# ===========================
echo ""
echo "===== [2/6] 设置 STEER_DATA_DIR -> data_test/ ====="
export STEER_DATA_DIR="$TEST_DATA_DIR"
echo "  STEER_DATA_DIR=$STEER_DATA_DIR"

# ===========================
# 3. 跑 steer_eval（生成 vector + 回复）
# ===========================
echo ""
echo "===== [3/6] 跑 steer_eval（vector + response） ====="
set +e
"$PROJECT_ROOT/shell/steer_eval.sh" \
    --method=caa \
    --generate_vector=true \
    --generate_response=true \
    --generate_orig_output=false \
    --evaluate=false \
    --layers=20 \
    --multipliers=1 \
    --gen_out_path="$GEN_OUT_PATH" \
    "${EXTRA_ARGS[@]}"
TEST_STATUS=$?
set -e
if [ $TEST_STATUS -ne 0 ]; then
    echo "❌ steer_eval 失败 (exit $TEST_STATUS)"
    exit $TEST_STATUS
fi

# 找 generation 产物
GEN_FILE=$(ls -1 "$GEN_DIR"/*/*/all_generation_results_*.json 2>/dev/null | head -1)
if [ -z "$GEN_FILE" ]; then
    echo "❌ 找不到 generation 输出: $GEN_DIR/*/*/all_generation_results_*.json"
    exit 1
fi
echo "  generation: $GEN_FILE"

# ===========================
# 4. 转比赛格式
# ===========================
echo ""
echo "===== [4/6] convert.py: 转比赛格式 ====="
python "$PROJECT_ROOT/shell/convert.py" \
    --input "$GEN_FILE" \
    --output "$SUBMISSION_JSON" \
    --team test_team
if [ ! -f "$SUBMISSION_JSON" ]; then
    echo "❌ convert 失败: $SUBMISSION_JSON 未生成"
    exit 1
fi
echo "  submission: $SUBMISSION_JSON"

# ===========================
# 5. 打分
# ===========================
if [ "$SKIP_SCORE" = "true" ]; then
    echo ""
    echo "===== [5/6] score.py: 跳过（SKIP_SCORE=true） ====="
    echo "  手动跑：python shell/score.py --input $SUBMISSION_JSON"
else
    echo ""
    echo "===== [5/6] score.py: CS/IS/FS 打分 ====="
    if [ ! -f "$PROJECT_ROOT/config/scorer.yaml" ]; then
        echo "❌ 缺少 config/scorer.yaml"
        exit 1
    fi
    if grep -q "YOUR_API_KEY_HERE" "$PROJECT_ROOT/config/scorer.yaml"; then
        echo "  ⚠️  config/scorer.yaml 还是占位 api_key，会 CS/IS/FS 全 0 分"
        echo "  配好后再跑：export SCORER_API_KEY=sk-xxx"
    fi
    python "$PROJECT_ROOT/shell/score.py" \
        --input "$SUBMISSION_JSON" \
        --output "$SCORE_JSON" \
        --concurrency 4 || true
    # 打分失败不退出（API 没配就 0 分继续看）
    if [ -f "$SCORE_JSON" ]; then
        echo ""
        echo "===== [6/6] 最终结果 ====="
        python -c "
import json
d = json.load(open('$SCORE_JSON'))
s = d.get('summary', {})
print(f'  concept: 1 (L1_1 only)')
print(f'  samples: {s.get(\"n_samples\", 0)}')
print(f'  Mean CS: {s.get(\"mean_cs\", 0):.4f}')
print(f'  Mean IS: {s.get(\"mean_is\", 0):.4f}')
print(f'  Mean FS: {s.get(\"mean_fs\", 0):.4f}')
print(f'  Mean HM: {s.get(\"mean_hm\", 0):.4f}')
"
    fi
fi

# Real data dir for reference (no symlink dance anymore).

echo "✅ 测试 pipeline 跑通"
echo "   vector:  output/vectors/qwen3-4b/SteerEval/personality/${GEN_OUT_PATH}/"
echo "   result:  $SUBMISSION_JSON"
echo "   scores:  $SCORE_JSON"
