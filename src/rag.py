import pickle
import logging
import re

from FlagEmbedding import FlagReranker
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama

from llama_index.core import Settings, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.ollama import OllamaEmbedding
from src.util.kg_post_processor import (
    NaivePostprocessor,
    KGRetrievePostProcessor,
    GraphFilterPostProcessor,
)
from src.util.kg_response_synthesizer import get_response_synthesizer
from config import (
    DATA_PATH,
    DEFAULT_RERANKER,
    PERSIST_DIR,
    GOOGLE_API_KEY,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    KG_PATH
    )

from src.kgvisual import kg_visualization

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 设置日志格式
)


# --- RAG引擎类 ---
class RAGEngine:
    def __init__(
        self,
        model_type='DeepSeek-R1',
        top_k=5,
        hops=1,
        kg_path=KG_PATH,
        reranker=DEFAULT_RERANKER,
        persist_dir=PERSIST_DIR,
    ):
        self.model_type = model_type
        self.reranker = reranker
        self.persist_dir = persist_dir
        self.kg_path = kg_path
        self.top_k = top_k
        self.hops = hops
        self.engine = None  # 引擎将在 initialize 方法中初始化
        self.initialize()
        self.load_kg()

    def load_kg(self):
        logging.info("Loading KG...")
        with open(self.kg_path, "rb") as f:
            self.kg_dict = pickle.load(f)
            
    def initialize(self):
        """初始化RAG引擎"""
        try:
            if self.model_type == "gemini":
                logging.info("Initializing Gemini...")
                Settings.llm = Gemini(api_key=GOOGLE_API_KEY, model="models/gemini-2.0-flash")

            elif self.model_type == "DeepSeek-R1":
                logging.info("Initializing DeepSeek-R1...")
                llm = Ollama(
                model =OLLAMA_MODEL, 
                base_url = OLLAMA_BASE_URL
            )
                llm.complete("hello world!")
                
                Settings.llm =llm
            
            logging.info("Initializing OllamaEmbedding...")
            emd = OllamaEmbedding(
                model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
            emd.get_query_embedding("hello world!")
            
            Settings.embed_model = emd

            logging.info("Loading entities and doc2kg...")
            with open(f"{DATA_PATH}/entities_doc2kg.pkl", "rb") as f:
                loaded_dict = pickle.load(f)

            with open(f"{DATA_PATH}/chunk_index.pkl", "rb") as f:
                chunks_index = pickle.load(f)

            ents = loaded_dict["ents"]
            doc2kg = loaded_dict["doc2kg"]

            logging.info("Loading index...")
            sc = StorageContext.from_defaults(persist_dir=self.persist_dir)
            index = load_index_from_storage(sc)
            retriever = VectorIndexRetriever(index=index, similarity_top_k=self.top_k)

            qa_rag_template_str = (
                "Context information is below.\n{context_str}\nQ: {query_str}\nA: "
            )
            qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT, text_qa_template=qa_rag_prompt_template
            )

            kg_post_processor1 = KGRetrievePostProcessor(
                ents=ents, doc2kg=doc2kg, chunks_index=chunks_index, hops=self.hops
            )
            bge_reranker = FlagReranker(model_name_or_path=self.reranker)
            kg_post_processor2 = GraphFilterPostProcessor(
                topk=self.top_k,
                ents=ents,
                doc2kg=doc2kg,
                chunks_index=chunks_index,
                reranker=bge_reranker,
            )

            self.engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[
                    kg_post_processor1,
                    kg_post_processor2,
                    NaivePostprocessor(),
                ],
            )
            logging.info("RAG engine initialized successfully.")

        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise  # 重新抛出异常，以便上层处理
        except Exception as e:
            logging.error(f"An error occurred during initialization: {e}")
            raise
        
    def _remove_brackets(self, text: str) -> str:
        cleaned_text = re.sub(r"[\[\]]", "", text)
        return cleaned_text
    
    def query_kg(self, pmid_list):
        return kg_visualization(pmid_list, self.kg_dict)
    
    def _query(self, question):
        """执行查询"""
        if self.engine is None:
            raise Exception("RAG engine is not initialized.")

        response = self.engine.query(question)
        answer = response.response
        sps = [source_node.node.id_ for source_node in response.source_nodes]
        context = {}
        for s in sps:
            with open(f"{DATA_PATH}/delirium_text/{s.replace('pmid', '')}.txt", "r") as f:
                text_line = f.readlines()
                title = self._remove_brackets(text_line[0].strip().split("|")[-1])
                abstract = text_line[1].strip().split("|")[-1]
                context[s] = {"title": title, "abstract": abstract}

        output_dict = {
            "Question": question,
            "Answer": answer + "\n**Supporting literature**: " + ", ".join(sps).upper(),
            "Supporting literature": sps,
            "Context": context,
            "KG": self.query_kg(sps),
        }
        logging.info(f"**Question:** {question}")
        logging.info(f"**Answer:** {answer}")
        logging.info(f"**Supporting literature:** {sps}")
        return output_dict

    def query(self, question):
        """查询"""
        try:
            return self._query(question)
        except Exception as e:
            logging.info(f"the error is {e}")
            return self._query(question)
        except Exception as e:
            logging.error(f"An error occurred during query: {e}")
            return {
            "Question": None,
            "Answer": None,
            "Supporting literature": None,
            "Context":  None,
            "KG":  None,
        }

