#!/bin/bash
# USAGE: bash server.sh

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm_env

model_path="./model/Qwen/Qwen2___5-0___5B-Instruct"
CUDA_VISIBLE_DEVICES=0 API_PORT=8621 llamafactory-cli api \
    --model_name_or_path $model_path \
    --template qwen \
    --infer_backend vllm \
    --vllm_gpu_util 0.99 \
    --vllm_maxlen 512 \
    --vllm_enforce_eager
