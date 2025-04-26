model_path="./model/Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
    --model_name_or_path $model_path\
    --template qwen