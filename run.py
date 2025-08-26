
from src.rag import RAGEngine



engine = RAGEngine(
    data_root="APOE_benchmark"
)

engine.setup_query_engine(
    model_type='VLLM',
    api_key=None,
    top_k=10,
    hops=1,
)

response_generator = engine.query(
        question= "How does APOE genotype affect the risk of Alzheimer's disease?",
    )
        
for data in response_generator:
    if data['type'] == 'result':
        print(data['data'])