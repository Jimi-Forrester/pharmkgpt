import pytest

from src.rag import RAGEngine

TEST_DATA_PATH = "/home/mindrank/fuli/delirium-rag/Data_test_v3"  

f"{TEST_DATA_PATH}/delirium_kg.pkl"


def test_qa():
    engine = RAGEngine(
        kg_path=f"{TEST_DATA_PATH}/delirium_kg.pkl",
        persist_dir=f"{TEST_DATA_PATH}/derilirum_index"
    )

    engine.setup_query_engine(
        model_type='gemma3',
        api_key=None,
        top_k=5,
        hops=1,
    )
    # output = engine.query("How does kynurenic acid contribute to dilirium?")
    output = engine.query("How about dementia assay in patients with delirium?")
    assert 'pmid36409557' in [i for i in output['Context']]
        
