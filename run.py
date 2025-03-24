from scr.rag import RAGEngine


rag_engine = RAGEngine()
output = rag_engine.query("How does kynurenic acid contribute to dilirium?")
print(output)