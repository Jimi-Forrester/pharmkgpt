from src.rag import RAGEngine


engine = RAGEngine(
    data_root="/home/mindrank/fuli/delirium-rag/Data_test_v4"
)
engine.setup_query_engine(
    model_type='gemma3',
    api_key=None,
    top_k=5,
    hops=1,
)

response_generator = engine.query("How does kynurenic acid contribute to dilirium?")
collected_results_loop = []
for piece in response_generator:
    collected_results_loop.append(piece)

for term in response_generator:
    print(term)
