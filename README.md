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
pip install -qU langchain-google-genai
pip install langchain_community

# 安装了itext2kg 可能需要更新
conda uninstall numpy scipy scikit-learn
conda install numpy scipy scikit-learn
```

## 5.Run

```sh
# 测试
# 修改config.py 里面的DATA_PATH 为 Data_test 快速跑通流程
python run.py
# 启动
# 多卡 gpu 的时候，指定 GPU
export CUDA_VISIBLE_DEVICES=0

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