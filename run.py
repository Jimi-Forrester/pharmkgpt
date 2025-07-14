from src.rag import RAGEngine


engine = RAGEngine(
    data_root="/home/mindrank/fuli/delirium-rag/benchmark_data2",
)
engine.setup_query_engine(
    model_type='gemma3',
    api_key=None,
    top_k=5,
    hops=1,
)

response_generator = engine.query("Did the patient experience symptoms resembling delirium after receiving low-dose cyclophosphamide chemotherapy?")
collected_results_loop = []
for piece in response_generator:
    collected_results_loop.append(piece)

for term in response_generator:
    print(term)
