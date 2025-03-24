from src.rag import RAGEngine


engine = RAGEngine()
output = engine.query("How does kynurenic acid contribute to dilirium?")
print(output)
