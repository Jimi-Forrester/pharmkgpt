# delirium-rag

## 1. Model Directory

Install [ollama](https://ollama.com/)
Download deepseek-r1:1.5b with ollama

```sh
ollama run deepseek-r1:1.5b
ollama run nomic-embed-text:latest
ollama serve  #启动 ollama
```

## 2. Download `bge-reranker-large`

Command to download with [hf-mirror](https://hf-mirror.com/):

```sh
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh BAAI/bge-reranker-large
```

## 3.Data

```sh
 unzip Data.zip
```

## 4. ENV

```sh
Conda create -n rag python=3.10
pip install -r requirements.txt

# 单独安装
pip install llama-index-llms-ollama
pip install llama-index-embeddings-ollama
pip install llama-index-llms-gemini
```

## 5.Run

```sh
python run.py  # 主函数 RAG，返回字典
```
