# delirium-rag
![alt text](image.png)
## 1. Model Directory

Install [ollama](https://ollama.com/)

```sh
ollama pull deepseek-r1:32b
ollama pull nomic-embed-text:latest
ollama pull gemma3:27b
ollama pull Qwen2.5:0.5b 
ollama serve  #启动 ollama
```
## 2 Installation

#### Download `bge-reranker-large`

Command to download with [hf-mirror](https://hf-mirror.com/):
```sh
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh BAAI/bge-reranker-large
```

## 2.1 Docker
```
docker pull deepnote/python:3.10-conda
docker build -t mindresilience:v1 .

docker run -d -p 5000:5000 \
    -v "/home/mindrank/fuli/delirium-rag/Data_v4:/app/data" \
    -e OLLAMA_BASE_URL="http://127.0.0.1:11434" \
    --name mindresilience-app mindresilience:v1
```



## 2.2 Conda

```
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export RERANKER_PATH=/home/mindrank/fuli/delirium-rag/bge-reranker-large
export DATA_ROOT=/home/mindrank/fuli/delirium-rag/Data_test_v3 #数据位置
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
pip install itext2kg    

# 安装了itext2kg 可能需要更新
conda uninstall numpy scipy scikit-learn
conda install numpy scipy scikit-learn
```

## 6.Run

```sh
# 测试
# 修改config.py 里面的DATA_PATH 为 Data_test 快速跑通流程
python run.py
# 启动
# 多卡 gpu 的时候，指定 GPU
export CUDA_VISIBLE_DEVICES=0
export 


python app.py  

curl -X POST -H "Content-Type: application/json" -d '{"question": "How does kynurenic acid contribute to dilirium?"}' http://127.0.0.1:5000/query
```

## 6. 测试

20250328测试文件Data_test_v3
```
pytest test/test_QA.py 
```

## 7. 数据记录
Data_v1：初始版本为了跑通 demo
Data_v2: 去掉一些空的abstract
Data_v3: 重构数据结构，导致检索出错
Data_v4: 在 v2 的基础上重构数据，对应测试数据 Data_test_v3