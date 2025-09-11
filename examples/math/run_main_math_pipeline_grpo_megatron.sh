#! /bin/bash
set -x

tabs 4
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
export PYTHONPATH=${REPO_PATH}:/mnt/public/hujunhao:/mnt/public/hujunhao/ElasticMegatron:$PYTHONPATH
# export PYTHONPATH=${REPO_PATH}:/mnt/public/hujunhao:/mnt/public/hujunhao/params_resharding_release:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="qwen2.5-1.5b-grpo-megatron-pipeline"
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/math/main_math_pipeline.py --config-path ${CONFIG_PATH}/config/  --config-name $CONFIG_NAME