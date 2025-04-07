FROM rapidsai/miniforge-cuda:cuda12.8.0-base-ubuntu22.04-py3.10

COPY ./ /app
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

RUN pip install llama-index-llms-ollama 
RUN pip install llama-index-embeddings-ollama
RUN pip install llama-index-llms-gemini
RUN pip install itext2kg

VOLUME /app/bge-reranker-large
VOLUME /app/data

CMD ["python", "/app/app.py"]