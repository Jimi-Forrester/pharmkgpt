# 数据路径
DATA_PATH = "/home/mindrank/fuli/delirium-rag/Data_test_v3"  

PERSIST_DIR = f"{DATA_PATH}/derilirum_index"  # 索引持久化目录

# 知识图谱路径
KG_PATH = f"{DATA_PATH}/delirium_kg.pkl"  # 知识图谱路径

# 默认reranker模型
DEFAULT_RERANKER = "bge-reranker-large"  

# Gemini API密钥 (如果使用Gemini)
GOOGLE_API_KEY = "AIzaSyCQaHZ0YOhVMqTw7XkWVhcR6pBMfZdeArg"  

# Ollama API地址
OLLAMA_BASE_URL = "http://127.0.0.1:11434"  

# Embedding 模型
EMBED_MODEL = "nomic-embed-text:latest" 


MODEL_DICT = {
    "gemma3": "gemma3:27b",
    "DeepSeek-R1": "deepseek-r1:32b",
    "Qwen2.5": "Qwen2.5:0.5b"
}
