import os
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
print(f"Attempting to load .env from: {dotenv_path}")

load_dotenv(dotenv_path=dotenv_path)


DATA_ROOT = "/home/mindrank/fuli/delirium-rag/Data_v7_0509" # 容器内数据根目录
RERANK_PATH= os.getenv("RERANKER_PATH")

EMBED_MODEL = "nomic-embed-text:latest"

# 这个字典相对静态，可以保留在 config.py 中，但也可以通过更复杂的配置管理方式外部化
MODEL_DICT = {
    "gemma3": "gemma3:27b",
    "DeepSeek-R1": "deepseek-r1:32b",
    "Qwen2.5": "Qwen2.5:0.5b",
    "MindGPT": "Qwen2.5:0.5b"
}
