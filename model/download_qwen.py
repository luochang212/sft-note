# USAGE: python download_qwen.py

from modelscope import snapshot_download

model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='./')
# model_dir = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='./')

print(f"model_dir: {model_dir}")
