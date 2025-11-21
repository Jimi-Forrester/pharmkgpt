import os
import pytest
import time

if os.getenv("RUN_HEAVY_RAG_TESTS", "0") != "1":
    pytest.skip(
        "Skipping RAG integration tests because RUN_HEAVY_RAG_TESTS is disabled.",
        allow_module_level=True,
    )

from src.rag import RAGEngine

TEST_DATA_PATH = "/home/mindrank/fuli/delirium-rag/Data_test_v4"
model_type_list = ["gemma3", "DeepSeek-R1", "Qwen2.5"]

engine = RAGEngine(data_root=TEST_DATA_PATH)
    

def test_qa():

    api_key = None
    
    for model_type in model_type_list:
        
        for top_k in range(1, 16):

            engine.setup_query_engine(
                model_type=model_type,
                api_key=api_key,
                top_k=top_k,
            )
            response_generator = engine.query("How does kynurenic acid contribute to dilirium?")
            collected_results_loop = []
            for piece in response_generator:
                collected_results_loop.append(piece)

            for term in response_generator:
                assert 'pmid36409557' in [i for i in term['data']['Context']]

            response_generator = engine.query("xsdfdfdsdfe")
            collected_results_loop = []
            for piece in response_generator:
                collected_results_loop.append(piece)

            for term in response_generator:
                assert None == term['data']['Context']
            
            response_generator = engine.query("Is there research on the protein SCUBE2?")
            collected_results_loop = []
            for piece in response_generator:
                collected_results_loop.append(piece)

            for term in response_generator:
                assert None == term['data']['Context']
                

def test_gemini_qa():
    model_type = 'gemini'
    api_key = 'AIzaSyCQaHZ0YOhVMqTw7XkWVhcR6pBMfZdeArg'
                
    for top_k in range(1, 16):
        engine.setup_query_engine(
            model_type=model_type,
            api_key=api_key,
            top_k=top_k,
        )
        
        time.sleep(60)
        response_generator = engine.query("How does kynurenic acid contribute to dilirium?")
        collected_results_loop = []
        for piece in response_generator:
            collected_results_loop.append(piece)

        for term in response_generator:
            assert 'pmid36409557' in [i for i in term['data']['Context']]

        
        response_generator = engine.query("xsdfdfdsdfe")
        collected_results_loop = []
        for piece in response_generator:
            collected_results_loop.append(piece)

        for term in response_generator:
            assert None == term['data']['Context']
        
        time.sleep(60)
        response_generator = engine.query("Is there research on the protein SCUBE2?")
        collected_results_loop = []
        for piece in response_generator:
            collected_results_loop.append(piece)

        for term in response_generator:
            assert None == term['data']['Context']