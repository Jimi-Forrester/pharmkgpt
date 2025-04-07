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


### 2.1 Docker

```sh
docker pull deepnote/python:3.10-conda
docker build -t mindresilience:v1 .
```


### 解压重排脚本和数据

```sh
tar -zxvf bge-reranker-large.tar.gz 
tar -zxvf Data_test_v3.tar.gz # 测试数据
tar -zxvf Data_v4.tar.gz  # delirium 数据
```


### 运行 docker
```
# 先测试测试集是否能成功，测试集载入时间短
docker run -d -p 5000:5000 \
    -v "/home/mindrank/fuli/delirium-rag/Data_test_v3:/app/data" \
    -v "/home/mindrank/fuli/delirium-rag/bge-reranker-large:/app/bge-reranker-large" \
    -e OLLAMA_BASE_URL="http://host.docker.internal:11434" \
    --name mindresilience-app-test mindresilience:v1

# 设置模型参数
curl -X POST http://localhost:5000/api/update_config \
     -H "Content-Type: application/json" \
     -d '{
           "model_type": "gemma3",
           "api_key": "NA",
           "top_k": 5,
           "hops": 1
         }'

# chat
curl -X POST http://localhost:5000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the main symptoms of delirium?"}' \
     -N 



# 跑所有数据
docker run -d -p 5000:5000 \
    -v "/home/mindrank/fuli/delirium-rag/Data_v4:/app/data" \
    -v "/home/mindrank/fuli/delirium-rag/bge-reranker-large:/app/bge-reranker-large" \
    -e OLLAMA_BASE_URL="http://host.docker.internal:11434" \
    --name mindresilience-app mindresilience:v1
```


### 2.2 Conda
```
Conda create -n rag python=3.10
pip install -r requirements.txt

# 单独安装
pip install llama-index-llms-ollama
pip install llama-index-embeddings-ollama
pip install llama-index-llms-gemini
pip install itext2kg    
pip install --upgrade gradio

# 安装了itext2kg 可能需要更新
conda uninstall numpy scipy scikit-learn
conda install numpy scipy scikit-learn

```


## 2.3 Local Run

```sh
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export RERANKER_PATH=/home/mindrank/fuli/delirium-rag/bge-reranker-large
export DATA_ROOT=/home/mindrank/fuli/delirium-rag/Data_test_v3 #数据位置
# 测试

python run.py
# 启动
# 多卡 gpu 的时候，指定 GPU
export CUDA_VISIBLE_DEVICES=0


python app.py  

curl -X POST -H "Content-Type: application/json" -d '{"question": "How does kynurenic acid contribute to dilirium?"}' http://172.17.0.1:5000/query
```

## 3. 测试

20250328测试文件Data_test_v3
```
pytest test/test_QA.py 
```

## 4. 数据记录

Data_v1：初始版本为了跑通 demo
Data_v2: 去掉一些空的abstract
Data_v3: 重构数据结构，导致检索出错
Data_v4: 在 v2 的基础上重构数据，对应测试数据 Data_test_v3