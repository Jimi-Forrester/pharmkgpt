from src.rag import RAGEngine
import json
import pandas as pd
from tqdm import tqdm

engine = RAGEngine(
    data_root="/home/mindrank/fuli/delirium-rag/Data_v7_0509"
)

engine.setup_query_engine(
    model_type='VLLM',
    api_key=None,
    top_k=10,
    hops=1,
)

df = pd.read_csv('metabolites_all.csv')
metabolite_list = df['metabolite']

output_query = {}
for metabolite in tqdm(metabolite_list):
    response_generator = engine.query(f"Is there research on the metabolites {metabolite}?")
    for term in response_generator:
        if term['type'] =='result':
            results_dict = term['data']
            if results_dict["Context"]:
                output_query[f"{metabolite}_Q1"] = {
                    "Question":results_dict["Question"],
                    "Answer":results_dict["Answer"],
                    "Context":results_dict["Context"],
                    "Supporting literature":results_dict["Supporting literature"]
                    }
    
    response_generator = engine.query(f"What is the role of {metabolite} in the delirium?")
    for term in response_generator:
        if term['type'] =='result':
            results_dict = term['data']
            if results_dict["Context"]:
                output_query[f"{metabolite}_Q2"] = {
                    "Question":results_dict["Question"],
                    "Answer":results_dict["Answer"],
                    "Context":results_dict["Context"],
                    "Supporting literature":results_dict["Supporting literature"]
                    }

    response_generator = engine.query(f"What is the role of {metabolite} in the alzheimer's disease?")
    for term in response_generator:
        if term['type'] =='result':
            results_dict = term['data']
            if results_dict["Context"]:
                output_query[f"{metabolite}_Q3"] = {
                    "Question":results_dict["Question"],
                    "Answer":results_dict["Answer"],
                    "Context":results_dict["Context"],
                    "Supporting literature":results_dict["Supporting literature"]
                    }

with open(f'case_study/metabolites_all.json', 'w', encoding='utf-8')as f:
    json.dump(output_query, f, ensure_ascii=False, indent=4)
