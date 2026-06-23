#!/bin/bash
# ===========================================================================
# Steer Eval 启动脚本 Qwen3-4b/mps 模型优化
#
#
# 用法：
#   ./shell/steer_eval.sh \
#       --device=mps \
#       --method=caa \
#       --layers=20 --multipliers=3 \
#       --generate_vector=true --generate_response=true \
#       --generate_orig_output=true --evaluate=true \
#       --exp=valid
# ===========================================================================

set -e

# ===========================
# 路径准备
# ===========================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EASYEDIT_DIR="$PROJECT_ROOT/EasyEdit"

if [ ! -e "$EASYEDIT_DIR/data" ]; then
    ln -s "$PROJECT_ROOT/data" "$EASYEDIT_DIR/data"
    echo "[setup] linked $PROJECT_ROOT/data -> $EASYEDIT_DIR/data"
fi

# 项目本地的 hparams（针对 qwen3-4b 的 CAA/RePS/Prompt 配置）放在
# 项目根目录的 hparams/qwen3-4b/，软链到 EasyEdit 里 SteerEval 期望的位置，
# 这样不污染 EasyEdit 仓库本体（pull 时不会冲突）。
HPARAM_TARGET_DIR="$EASYEDIT_DIR/hparams/Steer/experiment_hparams/steer_eval"
if [ ! -e "$HPARAM_TARGET_DIR/qwen3-4b" ]; then
    ln -s "$PROJECT_ROOT/hparams/qwen3-4b" "$HPARAM_TARGET_DIR/qwen3-4b"
    echo "[setup] linked $PROJECT_ROOT/hparams/qwen3-4b -> $HPARAM_TARGET_DIR/qwen3-4b"
fi

# ===========================
# Default Values
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
echo "Experiment:      $exp"
echo "--------------------------------"

# ===========================
# best_multip_path 相对路径（指向项目根目录 output/ 下生成的 best multipliers）
# 注：脚本会在第 188 行 `cd "$EASYEDIT_DIR"`，所以这里用 `../output/...` 才是项目根。
# ===========================
best_multip_path=None
if [ "$use_best_multip" = true ] ; then
    if [ "$use_pca" = true ] ; then
        best_multip_path="../output/generation/$model/${dataset}/${gen_out_path}/pca/best_multipliers.json"
    else
        best_multip_path="../output/generation/$model/${dataset}/${gen_out_path}/${method}/best_multipliers.json"
    fi
    echo "Using best multipliers from: $best_multip_path"
    multipliers=0
fi

# ===========================
# 模型路径
# ===========================
model_name_or_path="$PROJECT_ROOT/$model"

# ===========================
# Hparam 路径（指向项目根 hparams/，已被脚本软链到 EasyEdit 内对应位置）
# ===========================
steer_train_hparam_paths="[hparams/Steer/experiment_hparams/steer_eval/$model/${method}/generate_${method}.yaml]"
apply_steer_hparam_paths="[hparams/Steer/experiment_hparams/steer_eval/$model/${method}/apply_${method}.yaml]"

# ===========================
# 输出目录（项目根的 output/，不污染 EasyEdit 仓库）
# 注：相对路径在 `cd $EASYEDIT_DIR` 之后解析为 $PROJECT_ROOT/output/...
# ===========================
OUTPUT_ROOT="$PROJECT_ROOT/output"
steer_vector_output_dirs="[../output/vectors/$model/${dataset}/${gen_out_path}]"
steer_vector_load_dir="[../output/vectors/$model/${dataset}/${gen_out_path}]"

if [ "$use_pca" = true ] ; then
    generation_output_dir=../output/generation/$model/${dataset}/${gen_out_path}/pca/layer_${layers}_multip_${multipliers}
else
    generation_output_dir=../output/generation/$model/${dataset}/${gen_out_path}/${method}/layer_${layers}_multip_${multipliers}
fi

logdir=$OUTPUT_ROOT/logs/${model}/${dataset}/${gen_out_path}/${method}/layer_${layers}_multip_${multipliers}.log
mkdir -p "$OUTPUT_ROOT/logs/${model}/${dataset}/${gen_out_path}/${method}"

# ===========================
# 设备处理
# ===========================
ENV_PREFIX=""
if [[ "$device" == cuda* ]]; then
    ENV_PREFIX="CUDA_VISIBLE_DEVICES=$gpu"
fi

# ===========================
# 进入 EasyEdit 目录并执行
# ===========================
cd "$EASYEDIT_DIR"

# 把 EasyEdit 加到 PYTHONPATH，让 steer_eval.py 能 `import steer`（steer 包在 EasyEdit/steer/）
export PYTHONPATH="$EASYEDIT_DIR:${PYTHONPATH:-}"

HPARMS_LINK="examples/hparams"
HPARMS_BACKUP=""
if [ ! -e "$HPARMS_LINK" ]; then
    ln -s "../hparams" "$HPARMS_LINK"
    HPARMS_TEMP_CREATED=true
fi
trap 'if [ "$HPARMS_TEMP_CREATED" = "true" ] && [ -L "'"$HPARMS_LINK"'" ]; then rm "'"$HPARMS_LINK"'"; echo "  清理临时软链: '"$HPARMS_LINK"'"; fi' EXIT

eval "$ENV_PREFIX" python examples/steer_eval.py \
    model_name_or_path="${model_name_or_path}" \
    device="${device}" \
    dtype="${dtype}" \
    +method="${method}" \
    +use_pca="${use_pca}" \
    +dataset="${dataset}" \
    steer_train_hparam_paths="$steer_train_hparam_paths" \
    apply_steer_hparam_paths="$apply_steer_hparam_paths" \
    +generate_vector="${generate_vector}" \
    steer_vector_output_dirs="$steer_vector_output_dirs" \
    +generate_response="$generate_response" \
    steer_vector_load_dir="$steer_vector_load_dir" \
    generation_output_dir="$generation_output_dir" \
    generate_orig_output="$generate_orig_output" \
    +evaluate="$evaluate" \
    +vllm_enable="$vllm_enable" \
    +layers=["$layers"] \
    +multipliers=["$multipliers"] \
    +best_multip_path="$best_multip_path" \
    +exp="$exp" \
    2>&1 | tee "$logdir"


# ===========================
# example
# ===========================
# 1) 使用 MPS 跑 CAA baseline 只生成向量
# ./shell/steer_eval.sh --device=mps --method=caa --generate_vector=true --generate_response=false --evaluate=false
#
# 2) 生成向量 + 生成回复 + 评测
# ./shell/steer_eval.sh --device=mps --method=caa --generate_vector=true --generate_response=true --generate_orig_output=true --evaluate=true --layers=20 --multipliers=3
#
# 3) Linux + CUDA
# ./shell/steer_eval.sh --device=cuda:0 --gpu=0 --dtype=bfloat16 --method=caa
#
# 4) 只跑评测
# ./shell/steer_eval.sh --device=mps --generate_vector=false --generate_response=false --evaluate=true
