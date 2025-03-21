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

pip install flask
pip install itext2kg    

# 安装了itext2kg 可能需要更新
conda uninstall numpy scipy scikit-learn
conda install numpy scipy scikit-learn
```

## 5.Run

```sh
# 启动
python app.py  

curl -X POST -H "Content-Type: application/json" -d '{"question": "how does kynurenic acid contribute to dilirium?"}' http://127.0.0.1:5000/query
```
