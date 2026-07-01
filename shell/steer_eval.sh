#!/bin/bash
# ===========================================================================
# Steer Eval 启动脚本 — Qwen3-4b/mps CAA
#
#
# 用法：
#   ./shell/steer_eval.sh \
#       --device=mps \
#       --method=caa \
#       --layers=20 --multipliers=3 \
#       --generate_vector=true --generate_response=true \
#       --generate_orig_output=false --evaluate=false \
#       --exp=valid
# ===========================================================================

set -e

# ===========================
# 路径
# ===========================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ===========================
# 默认值
# ===========================
device=mps          # mps / cuda:0 / cpu
dtype=float16       # mps 上 bfloat16 兼容性差，使用 float16
gpu=0               # 仅当 device=cuda:* 时有效
vllm_enable=false   # mac/mps 不支持 vllm

model=qwen3-4b
method=caa
use_pca=false
dataset=SteerEval/personality

generate_vector=true
gen_out_path=baseline_v1

generate_response=true
generate_orig_output=false
evaluate=false

layers=20
multipliers=3

use_best_multip=false

max_new_tokens=512

clean=false

exp=valid           # 在 validation split 上评估

# ===========================
# Argument Parsing
# ===========================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --device=*) device="${1#*=}"; shift ;;
        --device)   device="$2"; shift; shift ;;

        --dtype=*) dtype="${1#*=}"; shift ;;
        --dtype)   dtype="$2"; shift; shift ;;

        --gpu=*) gpu="${1#*=}"; shift ;;
        --gpu)   gpu="$2"; shift; shift ;;

        --vllm_enable=*) vllm_enable="${1#*=}"; shift ;;
        --vllm_enable)   vllm_enable="$2"; shift; shift ;;

        --model=*) model="${1#*=}"; shift ;;
        --model)   model="$2"; shift; shift ;;

        --method=*) method="${1#*=}"; shift ;;
        --method)   method="$2"; shift; shift ;;

        --use_pca=*) use_pca="${1#*=}"; shift ;;
        --use_pca)   use_pca="$2"; shift; shift ;;

        --dataset=*) dataset="${1#*=}"; shift ;;
        --dataset)   dataset="$2"; shift; shift ;;

        --generate_vector=*) generate_vector="${1#*=}"; shift ;;
        --generate_vector)   generate_vector="$2"; shift; shift ;;

        --gen_out_path=*) gen_out_path="${1#*=}"; shift ;;
        --gen_out_path)   gen_out_path="$2"; shift; shift ;;

        --generate_response=*) generate_response="${1#*=}"; shift ;;
        --generate_response)   generate_response="$2"; shift; shift ;;

        --generate_orig_output=*) generate_orig_output="${1#*=}"; shift ;;
        --generate_orig_output)   generate_orig_output="$2"; shift; shift ;;

        --evaluate=*) evaluate="${1#*=}"; shift ;;
        --evaluate)   evaluate="$2"; shift; shift ;;

        --layers=*) layers="${1#*=}"; shift ;;
        --layers)   layers="$2"; shift; shift ;;

        --multipliers=*) multipliers="${1#*=}"; shift ;;
        --multipliers)   multipliers="$2"; shift; shift ;;

        --use_best_multip=*) use_best_multip="${1#*=}"; shift ;;
        --use_best_multip)   use_best_multip="$2"; shift; shift ;;

        --max_new_tokens=*) max_new_tokens="${1#*=}"; shift ;;
        --max_new_tokens)   max_new_tokens="$2"; shift; shift ;;

        --clean=*) clean="${1#*=}"; shift ;;
        --clean)   clean="$2"; shift; shift ;;

        --exp=*) exp="${1#*=}"; shift ;;
        --exp)   exp="$2"; shift; shift ;;
        *) echo "unknown: $1"; exit 1 ;;
    esac
done

# ===========================
# Verify Inputs
# ===========================
echo "--------------------------------"
echo "Device:          $device"
echo "Dtype:           $dtype"
echo "GPU ID:          $gpu (only used when device=cuda:*)"
echo "vLLM Enable:     $vllm_enable"
echo "Model:           $model"
echo "Method:          $method"
echo "use PCA:         $use_pca"
echo "Dataset:         $dataset"
echo "Generate Vector: $generate_vector"
echo "Output Path:     $gen_out_path"
echo "Generate Resp:   $generate_response"
echo "Orig Output:     $generate_orig_output"
echo "Evaluate:        $evaluate"
echo "Layers:          $layers"
echo "Multipliers:     $multipliers"
echo "Best Multip:     $use_best_multip"
echo "Max New Tokens:  $max_new_tokens"
echo "Clean:           $clean"
echo "Experiment:      $exp"
echo "STEER_DATA_DIR:  ${STEER_DATA_DIR:-<unset, uses ./data>}"
echo "--------------------------------"

# Early warnings for flags we don't support natively but accept for back-compat.
if [ "$vllm_enable" != "false" ]; then
    echo "[warn] --vllm_enable=$vllm_enable ignored (vLLM not wired in this local driver)"
fi
if [ "$use_pca" != "false" ]; then
    echo "[warn] --use_pca=$use_pca ignored (PCA pathway not implemented locally)"
fi
if [ "$use_best_multip" != "false" ]; then
    echo "[warn] --use_best_multip not implemented in this driver; using --multipliers as-is"
fi
if [ "$evaluate" != "false" ]; then
    echo "[warn] --evaluate ignored; use 'python shell/score.py' after this finishes"
fi
if [ "$method" != "caa" ]; then
    echo "[error] only --method=caa is supported by this driver (got '$method')"
    exit 1
fi

# ===========================
# 模型路径
# ===========================
model_name_or_path="$PROJECT_ROOT/$model"

# ===========================
# 输出目录
# ===========================
OUTPUT_ROOT="$PROJECT_ROOT/output"
steer_vector_output_dirs="$OUTPUT_ROOT/vectors/$model/${dataset}/${gen_out_path}"
steer_vector_load_dir="$OUTPUT_ROOT/vectors/$model/${dataset}/${gen_out_path}"
generation_output_dir="$OUTPUT_ROOT/generation/$model/${dataset}/${gen_out_path}/${method}/layer_${layers}_multip_${multipliers}"

mkdir -p "$OUTPUT_ROOT/logs/${model}/${dataset}/${gen_out_path}/${method}"
logdir="$OUTPUT_ROOT/logs/${model}/${dataset}/${gen_out_path}/${method}/layer_${layers}_multip_${multipliers}.log"

# ===========================
# Optional Clean
# ===========================
if [ "$clean" = "true" ]; then
    CLEAN_DIR="$OUTPUT_ROOT/generation/${model}/${dataset}/${gen_out_path}"
    if [ -d "$CLEAN_DIR" ]; then
        rm -rf "$CLEAN_DIR"
        echo "[clean] removed $CLEAN_DIR"
    fi
fi

# ===========================
# 设备处理（CUDA only）
# ===========================
ENV_PREFIX=""
if [[ "$device" == cuda* ]]; then
    ENV_PREFIX="CUDA_VISIBLE_DEVICES=$gpu"
fi

# ===========================
# Run
# ===========================
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

RUNNER="$PROJECT_ROOT/src/run_steer.py"
if [ ! -f "$RUNNER" ]; then
    echo "[error] $RUNNER not found"
    exit 1
fi

# --generate_vector: only pass when true; let the runner default to false when missing.
$ENV_PREFIX python "$RUNNER" \
    --model_name_or_path="$model_name_or_path" \
    --device="$device" \
    --dtype="$dtype" \
    --method="$method" \
    --dataset="$dataset" \
    --generate_vector="$generate_vector" \
    --gen_out_path="$gen_out_path" \
    --generate_response="$generate_response" \
    --generate_orig_output="$generate_orig_output" \
    --evaluate="$evaluate" \
    --layers="$layers" \
    --multipliers="$multipliers" \
    --max_new_tokens="$max_new_tokens" \
    --exp="$exp" \
    $([ "$clean" = "true" ] && echo "--clean") \
    2>&1 | tee "$logdir"

# ===========================
# example
# ===========================
# 1) MPS 跑 CAA baseline 只生成向量
# ./shell/steer_eval.sh --device=mps --method=caa --generate_vector=true --generate_response=false --evaluate=false
#
# 2) 生成向量 + 生成回复
# ./shell/steer_eval.sh --device=mps --method=caa --generate_vector=true --generate_response=true --generate_orig_output=false --evaluate=false --layers=20 --multipliers=3
#
# 3) Linux + CUDA
# ./shell/steer_eval.sh --device=cuda:0 --gpu=0 --dtype=bfloat16 --method=caa
