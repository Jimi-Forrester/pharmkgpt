FROM deepnote/python:3.10-conda


RUN pip install itext2kg \
    llama-index-llms-ollama \
    llama-index-embeddings-ollama \
    llama-index-llms-gemini

COPY ./ /app
WORKDIR /app

RUN pip install -r requirements.txt

RUN wget https://hf-mirror.com/hfd/hfd.sh
RUN chmod a+x hfd.sh
ENV HF_ENDPOINT=https://hf-mirror.com
RUN ./hfd.sh BAAI/bge-reranker-large

VOLUME /app/data

CMD ["python", "/app/app.py"]