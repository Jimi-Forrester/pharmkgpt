from src.rag import RAGEngine
import json
import pandas as pd
from tqdm import tqdm

engine = RAGEngine(
    data_root="/home/mindrank/fuli/delirium-rag/Data_v7_0509"
)

engine.setup_query_engine(
    model_type='MindGPT',
    api_key=None,
    top_k=10,
    hops=1,
)

df = pd.read_csv('/home/mindrank/fuli/delirium-rag/shap_summary.csv')

for group_name, gene_list in df.items():
    gene_list = list(set(gene_list))
    output_query = {}
    for gene in tqdm(gene_list):
        response_generator = engine.query(f"Is there research on the protein {gene}?")
        for term in response_generator:
            if term['type'] =='result':
                results_dict = term['data']
                if results_dict["Context"]:
                    output_query[f"{gene}_Q1"] = {
                        "Question":results_dict["Question"],
                        "Answer":results_dict["Answer"],
                        "Context":results_dict["Context"],
                        "Supporting literature":results_dict["Supporting literature"]
                        }
        
        response_generator = engine.query(f"What is the role of {gene} in the delirium?")
        for term in response_generator:
            if term['type'] =='result':
                results_dict = term['data']
                if results_dict["Context"]:
                    output_query[f"{gene}_Q2"] = {
                        "Question":results_dict["Question"],
                        "Answer":results_dict["Answer"],
                        "Context":results_dict["Context"],
                        "Supporting literature":results_dict["Supporting literature"]
                        }

        response_generator = engine.query(f"What is the role of {gene} in the alzheimer's disease?")
        for term in response_generator:
            if term['type'] =='result':
                results_dict = term['data']
                if results_dict["Context"]:
                    output_query[f"{gene}_Q3"] = {
                        "Question":results_dict["Question"],
                        "Answer":results_dict["Answer"],
                        "Context":results_dict["Context"],
                        "Supporting literature":results_dict["Supporting literature"]
                        }

        with open(f'case_study/{group_name}.json', 'w', encoding='utf-8')as f:
            json.dump(output_query, f, ensure_ascii=False, indent=4)
