import pickle
import logging
import re
import os
from FlagEmbedding import FlagReranker
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama

from llama_index.core.postprocessor import SimilarityPostprocessor
from langchain_community.llms import Ollama as lc_Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.core import Settings, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.ollama import OllamaEmbedding
from .util.kg_post_processor import (
    NaivePostprocessor,
    KGRetrievePostProcessor,
    GraphFilterPostProcessor,
)
from .util.kg_response_synthesizer import get_response_synthesizer
from config import (
    DATA_PATH,
    DEFAULT_RERANKER,
    PERSIST_DIR,
    MODEL_DICT,
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    KG_PATH
    )

from src.kgvisual import kg_visualization
from src.hightlight import HightLight_context, format_docs, highlight_segments

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 设置日志格式
)


# --- RAG引擎类 ---
class RAGEngine:
    def __init__(
        self,
        kg_path=KG_PATH,
        reranker=# `DEFAULT_RERANKER` is a constant variable that likely holds the default reranker model used in the
        # system. It is used in the code snippet you provided to initialize the `RAGEngine` class. The
        # `RAGEngine` class is responsible for initializing and managing the RAG (Retrieval-Augmented Generation)
        # engine, which is used for querying and retrieving information from a knowledge graph.
        DEFAULT_RERANKER,
        persist_dir=PERSIST_DIR,
    ):
        self.reranker = reranker
        self.persist_dir = persist_dir
        self.kg_path = kg_path
        self.engine = None  # 引擎将在 initialize 方法中初始化
        self.load_kg()

    def load_kg(self):
        logging.info("Loading KG...")
        with open(self.kg_path, "rb") as f:
            self.kg_dict = pickle.load(f)
            
    def initialize(self,         
                model_type='gemma3',
                api_key=None,
                top_k=5,
                hops=1,
                device="cuda:0"
                ):
        """初始化RAG引擎"""
        self.model_type=model_type
        self.api_key=api_key
        self.top_k=top_k
        self.hops=hops
        try:
            if model_type == "gemini":
                logging.info("Initializing Gemini...")
                llm = Gemini(api_key=api_key, model="models/gemini-2.0-flash")
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash", 
                    google_api_key=api_key, 
                    temperature=0
                    )

            elif model_type in MODEL_DICT:
                logging.info(f"Initializing {model_type}...")
                llm = Ollama(
                model = MODEL_DICT[model_type],
                base_url = OLLAMA_BASE_URL 
                )
                self.llm = lc_Ollama(
                model = MODEL_DICT[model_type],
                base_url = OLLAMA_BASE_URL
            )
            
            self.llm.invoke("hello world!")
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
            
            retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

            qa_rag_template_str = (
                "Context information is below.\n{context_str}\nQ: {query_str}\nA: "
            )
            qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT, text_qa_template=qa_rag_prompt_template
            )
            
            bge_reranker = FlagReranker(model_name_or_path=self.reranker, device=device)
            bge_reranker.model.to(device)
            
            # 基于pmid 和关系的图检索
            kg_post_processor1 = KGRetrievePostProcessor(
                ents=ents, 
                doc2kg=doc2kg, 
                chunks_index=chunks_index, 
                hops=hops
            )
            
            
            # 基于节点和 query 覆盖率的图过滤
            kg_post_processor2 = GraphFilterPostProcessor(
                topk=top_k,
                ents=ents,
                doc2kg=doc2kg,
                chunks_index=chunks_index,
                reranker=bge_reranker,
            )

            postprocessor = SimilarityPostprocessor(similarity_cutoff=0.6)

            self.engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[
                    kg_post_processor1,
                    kg_post_processor2,
                    postprocessor,
                    NaivePostprocessor(),
                ],
            )
            logging.info("RAG engine initialized successfully.")

        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise  # 重新抛出异常，以便上层处理
        except Exception as e:
            logging.error(f"An error occurred during initialization: {e}")
    
    def get_parameters(self):
        return {
            "model_type": self.model_type,
            "api_key_provided": bool(self.api_key),
            "top_k": self.top_k,
            "hops": self.hops
        }      
        
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
        
        if len(response.source_nodes) == 0:
            return {
                "Question": question,
                "Answer": "I'm sorry, I cannot answer this question based on the information currently available in the knowledge graph.",
                "Supporting literature": None,
                "Context":  None,
                "KG":  None,
            }
        else:  
            num_source_nodes = len(response.source_nodes) 
            logging.info(f"源节点数量：{num_source_nodes}") 

            # 循环遍历源节点并打印元数据
            for s in response.source_nodes: 
                logging.info(f"节点分数：{s.score}") 
                logging.info(s.node)
            
            answer = response.response
            sps = [source_node.node.id_ for source_node in response.source_nodes]
            sps_score = [source_node.score for source_node in response.source_nodes]
            context = {}
            for s, _score in zip(sps, sps_score):
                with open(f"{DATA_PATH}/delirium_text/{s.replace('pmid', '')}.txt", "r") as f:
                    text_line = f.readlines()
                    title = self._remove_brackets(text_line[0].strip().split("|")[-1])
                    abstract = text_line[1].strip().split("|")[-1]
                context[s] = {"title": title, "abstract": abstract, "score": _score, "pmid": s.replace('pmid', '')}

            # 高亮
            logging.info("Highlighting documents...")
            HighlightDocuments_dict = HightLight_context(
                input_data ={
                "documents": format_docs(context),
                "question": question,
                "generation": answer
                            }, 
                llm_model=self.llm
                ), 
            
            logging.info(f"HighlightDocuments_dict: {HighlightDocuments_dict}")
            context_ = highlight_segments(
                context, HighlightDocuments_dict
            )
            
            output_dict = {
                "Question": question,
                "Answer": answer + "\n**Supporting literature**: " + ", ".join(sps).upper(),
                "Supporting literature": sps,
                "Context": context_,
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