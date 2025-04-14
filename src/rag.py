import pickle
import logging
import re
import os
from FlagEmbedding import FlagReranker
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import SimilarityPostprocessor
from langchain_community.llms import Ollama as lc_Ollama

from llama_index.core import Settings, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.ollama import OllamaEmbedding

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from .util.kg_post_processor import (
    NaivePostprocessor,
    KGRetrievePostProcessor,
    GraphFilterPostProcessor,
)
from .util.kg_response_synthesizer import get_response_synthesizer
from .detect import is_likely_junk_input
from config import (
    MODEL_DICT,
    EMBED_MODEL, 
    DATA_ROOT,
    RERANK_PATH
    )

from src.kgvisual import kg_visualization
from src.hightlight import detect_all_entity_name, format_docs, highlight_segments_prioritized, hallucination_test
from src.hightlight import format_scientific_text_with_colon 

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
        data_root=DATA_ROOT,
        embed_model=EMBED_MODEL,
        reranker_path=RERANK_PATH,
    ):
        self.embed_model=embed_model
        logging.info(f"data_root: {data_root}")
        self.data_root = data_root
        self.reranker_path=reranker_path
        self.ollama_url="http://127.0.0.1:11434"
        self.load_kg()
        self.load_index()
        
        self._configured = False
        self.model_type=None
        self.api_key=None
        self.hops=None
        self.top_k=None
        self.engine = None  # 引擎将在 initialize 方法中初始化

    def load_kg(self):
        logging.info("Loading KG...")
        with open(f"{self.data_root}/delirium_kg.pkl", "rb") as f:
            self.kg_dict = pickle.load(f)
        
        logging.info("Loading entities and doc2kg...")
        with open(f"{self.data_root}/entities_doc2kg.pkl", "rb") as f:
            loaded_dict = pickle.load(f)

        with open(f"{self.data_root}/chunk_index.pkl", "rb") as f:
            self.chunks_index = pickle.load(f)

        with open(f"{self.data_root}/chunk_index_nodes.pkl", "rb") as f:
            self.chunk_index_embed = pickle.load(f)
            
        self.ents = loaded_dict["ents"]
        self.doc2kg = loaded_dict["doc2kg"]

    def load_index(self):
        logging.info("Initializing OllamaEmbedding...")
        emd = OllamaEmbedding(
            model_name=self.embed_model, 
            base_url=self.ollama_url
            )
        emd.get_query_embedding("hello world!")
        Settings.embed_model = emd
        
        logging.info("Loading index...")
        sc = StorageContext.from_defaults(persist_dir=f"{self.data_root}/derilirum_index")
        self.index = load_index_from_storage(sc)
        
        self.retriever = VectorIndexRetriever(
            index=self.index, 
            similarity_top_k=20)

        qa_rag_template_str = (
            "Context information is below.\n{context_str}\nQ: {query_str}\nA: "
        )
        self.qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)

        device="cuda:0"
        self.bge_reranker = FlagReranker(model_name_or_path=self.reranker_path, device=device)
        self.bge_reranker.model.to(device)

        
    def setup_query_engine(self,         
                model_type,
                api_key,
                top_k,
                hops,
                model_map=MODEL_DICT,
                ):
        """初始化RAG引擎"""
        self.model_type=model_type
        self.api_key=api_key
        self.top_k=top_k
        self.hops=hops
    
        llm = None 
        self.llm = None
        
        try:
            if model_type == "gemini":
                logging.info("Initializing Gemini...")
                llm = Gemini(api_key=api_key, model="models/gemini-2.0-flash")
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash", 
                    google_api_key=api_key, 
                    temperature=0
                    )
                
            elif model_type == "openai":
                logging.info("Initializing Openai...")
                llm = OpenAI(api_key=api_key, model="gpt-4o")
                
                self.llm = ChatOpenAI(
                    api_key=api_key,
                    model="gpt-4o", 
                    temperature=0,
                    max_retries=2
                    )

            elif model_type in MODEL_DICT:
                logging.info(f"Initializing {model_type}...")
                llm = Ollama(
                model = model_map[model_type],
                base_url = self.ollama_url
                )
                self.llm = lc_Ollama(
                model = model_map[model_type],
                base_url = self.ollama_url
            )
            else:
                logging.info(f"Initializing {model_type}...")
                llm = Ollama(
                    model = "gemma3:27b",
                    base_url = self.ollama_url
                )
                
                self.llm = lc_Ollama(
                    model = "gemma3:27b",
                    base_url = self.ollama_url
                )
                
            self.llm.invoke("hello world!")

            llm.complete("hello world!")
            
            Settings.llm =llm


            self.response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT, text_qa_template=self.qa_rag_prompt_template
            )
        
            # 基于pmid 和关系的图检索
            kg_post_processor1 = KGRetrievePostProcessor(
                ents=self.ents, 
                doc2kg=self.doc2kg, 
                chunks_index=self.chunks_index,
                chunk_index_embed=self.chunk_index_embed,
                hops=hops,
                top_k=top_k,
            )
            
            # 基于节点和 query 覆盖率的图过滤
            kg_post_processor2 = GraphFilterPostProcessor(
                topk=top_k,
                ents=self.ents,
                doc2kg=self.doc2kg,
                use_tpt=True,
                chunks_index=self.chunks_index,
                reranker=self.bge_reranker,
            )

            Simpostprocessor = SimilarityPostprocessor(
                similarity_cutoff=0.5
                )

            self.engine = RetrieverQueryEngine(
                retriever=self.retriever,
                response_synthesizer=self.response_synthesizer,
                node_postprocessors=[
                    kg_post_processor1,
                    kg_post_processor2,
                    Simpostprocessor,
                    NaivePostprocessor(),
                ],
            )
            
            self._configured = True
            logging.info("RAG engine initialized successfully.")
            
            
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise  # 重新抛出异常，以便上层处理
        except Exception as e:
            logging.error(f"An error occurred during initialization: {e}")
    
    def is_configured(self):
        return self._configured
    
    def get_status(self):
        status_data = {
            'configured': self._configured, # 使用 'configured' 字符串键
            "params": None                 # 初始化 params 为 None
            }
        
        if self._configured:
            status_data["params"] = {
                "model_type": self.model_type,
                "api_key_provided": bool(self.api_key),
                "top_k": self.top_k,
                "hops": self.hops
            }
        return status_data
        
    def _remove_brackets(self, text: str) -> str:
        cleaned_text = re.sub(r"[\[\]]", "", text)
        return cleaned_text
    
    def query_kg(self, pmid_list):
        return kg_visualization(pmid_list, self.kg_dict)
    
    
    def entity_dict(self,pmid_list):
        return detect_all_entity_name(pmid_list, self.kg_dict)
    
    def format_context(self, context):
        return format_scientific_text_with_colon(context)
    
    def _query(self, question):
        """执行查询"""
        if self.engine is None:
            raise Exception("RAG engine is not initialized.")
        
        yield {"type": "progress", "message": "Retrieving Knowledge"}
        
        response = None
        try:
            response = self.engine.query(question)

        except Exception as e: # 捕获其他所有在 query 中可能发生的错误
            # 记录详细错误供调试
            logging.error(f"Unexpected error processing question '{question}': {e}", exc_info=True)
            yield {
                "type": "result",
                "data": {
                    "Question": question,
                    # 提供通用的、友好的用户消息
                    "Answer": "I'm sorry, an unexpected error occurred while processing your question. Please try again later.",
                    "Supporting literature": None,
                    "Context": None,
                    "KG": None,
                }
            }
            return

        if not response or not getattr(response, 'response', None) or len(response.source_nodes) == 0: # 检查 response 是否存在且有实际内容
            logging.warning(f"Engine query for '{question}' returned an empty or null response.")
            yield {
                "type": "result", # 或者可以考虑用 "no_answer" 类型
                "data": {
                    "Question": question,
                    "Answer": "I'm sorry, I cannot find a specific answer based on the information currently available.",
                    "Supporting literature": getattr(response, 'source_nodes', None),
                    "Context": None,
                    "KG": None,
                }
            }
            return # 如果响应为空，也退出

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
                with open(f"{self.data_root}/delirium_text/{s.replace('pmid', '')}.txt", "r") as f:
                    text_line = f.readlines()
                    title = self._remove_brackets(text_line[0].strip().split("|")[-1])
                    abstract = text_line[1].strip().split("|")[-1]
                    if 'BACKGROUND' in abstract:
                        abstract = self.format_context(abstract)
                context[s] = {"title": title, "abstract": abstract, "score": _score, "pmid": s.replace('pmid', '')}
            
            yield {"type": "progress", "message": "Knowledge Retrieved.\nVerifying Facts"}
            
            # 幻觉检测
            logging.info(">>>hallucination detect>>>>>")
            logging.info(f"answer: {answer}")
            try:
                faithfulness_score = hallucination_test(
                    llm_model=self.llm,
                    input_data={'documents':format_docs(context), "generation": answer}
                ).Faithfulness_score
                logging.info(f"Faithfulness_score: {faithfulness_score}")
                if faithfulness_score <= 2:
                    pass
    
                elif faithfulness_score == 3:
                    answer +="\n While based on relevant documents, this answer might only partially address your specific query or could include details beyond the direct scope of the provided context."
                else:
                    yield {
                            "type": "result", # Keep type result, content indicates failure
                            "data": {
                                "Question": question,
                                "Answer": "Sorry, the initial answer I generated did not fully pass verification when checked against the reference documents. To ensure accuracy, I cannot provide a reliable answer right now.",
                                "Supporting literature": None, # Or maybe provide sps but indicate failure?
                                "Context": None,
                                "KG": None,
                            }
                                    }
                    return # Stop the generator

            except:
                logging.info(">>>hallucination detect has some problem!")
            
            # 高亮
            yield {"type": "progress", "message": "Knowledge Retrieved.\nFacts Verified.\nGenerating Answer"}
            
            logging.info(">>>>>Highlighting documents...>>>>>")
            # HighlightDocuments_dict = hightLight_context(
            #     input_data ={
            #     "documents": format_docs(context),
            #     "question": question,
            #     "generation": answer
            #                 }, 
            #     llm_model=self.llm
            #     )
            try:
                HighlightDocuments_dict = self.entity_dict(sps)

                context_ = highlight_segments_prioritized(
                    context, HighlightDocuments_dict
                )
            except:
                logging.error(">>>Highlighting has some problem!")
                context_ = context
            
            try:
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
                yield {"type": "result", "data": output_dict}
                return
            
            except Exception as e:
                logging.error(f"An error occurred while generating the output: {e}")
                yield {
                    "type": "result",
                    "data": {
                        "Question": question,
                        "Answer": answer,
                        "Supporting literature": sps,
                        "Context": context_,
                        "KG": None,
                    }
                }
                return

    def query(self, question):
        """查询"""
        if is_likely_junk_input(question, self.ents):
            yield {"type": "result", 
                        "data":{
                            "Question": question,
                            "Answer": f"It looks like you entered some random characters:\n\n{question}\n\nThis doesn't seem to be a specific question or request.\n\nCould you please clarify what you need help with or ask your question again?",
                            "Supporting literature": None,
                            "Context":  None,
                            "KG":  None,
                            }
                        }

        else:
            try:
                output_dict = self._query(question)
                yield from self._query(question)
            except Exception as e:
                logging.info(f"the error is {e}")
                output_dict = self._query(question)
                yield {"type": "result", "data": output_dict}
            except Exception as e:
                logging.error(f"An error occurred during query: {e}")
                yield {"type": "result", 
                        "data":{
                            "Question": None,
                            "Answer": None,
                            "Supporting literature": None,
                            "Context":  None,
                            "KG":  None,
                            }
                        }