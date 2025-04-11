import pytest

from src.rag import RAGEngine

TEST_DATA_PATH = "/home/mindrank/fuli/delirium-rag/Data_test_v4"  


def test_qa():
    engine = RAGEngine(
        data_root=TEST_DATA_PATH
    )

    engine.setup_query_engine(
        model_type='gemma3',
        api_key=None,
        top_k=5,
        hops=1,
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