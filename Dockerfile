FROM deepnote/python:3.10-conda


RUN pip install itext2kg \
    llama-index-llms-ollama \
    llama-index-embeddings-ollama \
    llama-index-llms-gemini

COPY ./ /app
WORKDIR /app

RUN pip install -r requirements.txt


VOLUME /app/data
VOLUME /app/bge-reranker-large

CMD ["python", "/app/app.py"]