import os


DATA_ROOT = os.getenv("DATA_ROOT", "/app/data") # 容器内数据根目录
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") 
RERANK_PATH= os.getenv("RERANK_PATH", "/app/bge-reranker-large")

RERANKER_PATH = os.getenv("RERANKER_PATH", "/app/bge-reranker-large")
EMBED_MODEL = "nomic-embed-text:latest"

# 这个字典相对静态，可以保留在 config.py 中，但也可以通过更复杂的配置管理方式外部化
MODEL_DICT = {
    "gemma3": "gemma3:27b",
    "DeepSeek-R1": "deepseek-r1:32b",
    "Qwen2.5": "Qwen2.5:0.5b"
}
