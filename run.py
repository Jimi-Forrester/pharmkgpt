from src.rag import RAGEngine


engine = RAGEngine(top_k=10, hops=2)
# output = engine.query("How does kynurenic acid contribute to dilirium?")
output = engine.query("How about dementia assay in patients with delirium?")
# print(output)
