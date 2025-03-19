import pickle   
import logging
from FlagEmbedding import FlagReranker

from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex,PromptTemplate,StorageContext,load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.ollama import OllamaEmbedding
from src.util.kg_post_processor import NaivePostprocessor,KGRetrievePostProcessor,ngram_overlap,GraphFilterPostProcessor
from src.util.kg_response_synthesizer import get_response_synthesizer

DATA_PATH = "/Users/fl/Desktop/my_code/delirium-rag/Data_v2"

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
)


def RAG(
    question, 
    top_k=5, 
    hops=1, 
    model_type='gemini', # gemini or ollama
    reranker='bge-reranker-large', 
    persist_dir = f'{DATA_PATH}/derilirum_index'
    ):
    if model_type == "gemini":
        GOOGLE_API_KEY="AIzaSyCQaHZ0YOhVMqTw7XkWVhcR6pBMfZdeArg"
        Settings.llm = Gemini(
            api_key=GOOGLE_API_KEY, 
            model='models/gemini-2.0-flash'
            )
        
    elif model_type == "ollama":
        Settings.llm = Ollama(
            model="deepseek-r1:1.5b",
            base_url="http://127.0.0.1:11434"
            )
        
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text:latest", 
        base_url="http://127.0.0.1:11434"
        )

    logging.info("Loading entities and doc2kg")
    with open(f'{DATA_PATH}/entities_doc2kg.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    with open(f'{DATA_PATH}/chunk_index.pkl', 'rb') as f:
        chunks_index = pickle.load(f)

    ents = loaded_dict['ents']
    doc2kg = loaded_dict['doc2kg']

    logging.info("Loaded index")
    sc = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(sc)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    # 定义 RAG 问答模板
    qa_rag_template_str = 'Context information is below.\n{context_str}\nQ: {query_str}\nA: '
    qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT, text_qa_template=qa_rag_prompt_template)

    # 初始化知识图谱增强的后处理器
    kg_post_processor1 = KGRetrievePostProcessor(
        ents=ents, 
        doc2kg=doc2kg, 
        chunks_index=chunks_index, 
        hops=hops)

    bge_reranker = FlagReranker(model_name_or_path=reranker)
    kg_post_processor2 = GraphFilterPostProcessor(topk=top_k, ents=ents, doc2kg=doc2kg, chunks_index=chunks_index, reranker=bge_reranker)

    logging.info("Loaded query engine")
    engine = RetrieverQueryEngine(
        retriever=retriever,  #向量检索器
        response_synthesizer=response_synthesizer,   # 生成答案的组件, 定义如何组合检索到的信息
        node_postprocessors=[
            kg_post_processor1, # 利用知识图谱增强答案
            kg_post_processor2, # 使用 bge-reranker 重新排序答案。
            NaivePostprocessor()  # 个简单的文本清理后处理器。
            ]
        )

    # 运行查询
    logging.info("Ready to query")
    try:
        response = engine.query(question)
        answer = response.response
    except Exception as e:
        logging.info("Query failed, retrying")
        response = engine.query(question)
        answer = response.response

    except Exception as e:
        logging.info("Query failed again, returning empty answer")
        response = engine.query(question)
        answer = response.response

        
    sps = [source_node.node.id_ for source_node in response.source_nodes]
    context = {
        
    }
    for s in sps:
        with open(f'{DATA_PATH}/delirium_text/{s.replace("pmid", "")}.txt', 'r') as f:
            text_line = f.readlines()
            title = text_line[0].strip().split('|')[-1]
            abstract = text_line[1].strip().split('|')[-1]
            context[s] = f"Title: {title}\n Abstract: {abstract}"

    output_dict = {
        "Question":question,
        "Answer":answer,
        "Supporting literature":sps,
        "Context":context
    }
    logging.info(f"**Question:** {question}")
    logging.info(f"**Answer:** {answer}")
    logging.info(f"**Supporting literature:** {sps}", )
    return output_dict

if __name__ == '__main__':
    question = "how does kynurenic acid contribute to dilirium?"
    output = RAG(
        question,
        top_k=5, 
        hops=1, 
        model_type ='ollama', # gemini or ollama
        )
    print(output)