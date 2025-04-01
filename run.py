from src.rag import RAGEngine


engine = RAGEngine()
engine.setup_query_engine(
    model_type='gemma3',
    api_key=None,
    top_k=5,
    hops=1,
)
# output = engine.query("How does kynurenic acid contribute to dilirium?")
output = engine.query("How does kynurenic acid contribute to dilirium?")
print(output)
