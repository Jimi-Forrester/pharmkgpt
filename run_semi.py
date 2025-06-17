from src.rag import RAGEngine


import json
import requests
import os
import re
from tqdm import tqdm
import logging


engine = RAGEngine(
    data_root="/home/mindrank/fuli/delirium-rag/benchmark_data2"
)
engine.setup_query_engine(
    model_type='MindGPT',
    api_key=None,
    top_k=10,
    hops=1,
)


def process_pmid(pmid_string):
    pmid_string = pmid_string.strip()  # 去除空格
    return [s.replace("pmid", "") for s in pmid_string.split(", ")]
    
def extract_specific_answer_option(text: str) -> str:
    """
    从文本中提取表示答案选项的单个大写字母 (A, B, C, 或 D)。
    具有不同的优先级来处理不同格式的答案指示。
    """
    answer_prefix_pattern = r"(?i)\bAnswer(?: is)?\s*[:\s]*\W*([A-D])(?:[\.:\s]|\b)"
    match = re.search(answer_prefix_pattern, text)
    if match:
        return match.group(1)

    explicit_phrase_pattern = r"(?i)(?:the answer is|the correct option is|option is|the choice is|choice is|is:)\s+\W*([A-D])(?:[\.:\s]|\b)"
    match = re.search(explicit_phrase_pattern, text)
    if match:
        return match.group(1)

    general_pattern = r"(?<![A-Za-z0-9])\W*([A-D])[\.:](?![A-Za-z0-9])"
    match = re.search(general_pattern, text)
    if match:
        return match.group(1)

    # Priority 4: 单独的大写字母 A, B, C, 或 D (作为一个完整的词，前后是单词边界)
    standalone_letter_pattern = r"\b([A-D])\b"
    match = re.search(standalone_letter_pattern, text)
    if match:
        return match.group(1)
        
    return None



with open(f'/home/mindrank/fuli/mcq_generator/QA_Data/QA.json', 'r', encoding='utf-8') as f:
    QA_list = json.load(f)
    
answer_dict = {}
for group_name, group_dict in QA_list.items():
    answer_dict[group_name] = {}
    for k, qa_dict in tqdm(group_dict.items()):
        response_generator = engine.query(
            question= f"{qa_dict['question']} select the best answer from the following options",
            option= qa_dict['options']
            )
        
        for data in response_generator:
            if data['type'] == 'result':
                answer_dict[group_name][k] = {
                    "question": qa_dict['question'],
                    "answer": data['data']['Answer'],
                    "pmid": data['data']['Supporting literature']
                }

with open(f'pharmkgpt_smi_answer2.json', 'w', encoding='utf-8') as f:
    json.dump(answer_dict, f, ensure_ascii=False, indent=4)






# response_generator = engine.query(
#     question="What is the relationship between cholinergic drugs used for treating neuroleptic-induced tardive dyskinesia and Alzheimer's disease?", 
#     option="A. Cholinergic drugs are effective in treating both conditions.; B. Cholinergic drugs have been primarily developed for Alzheimers disease., C. Cholinergic drugs used for tardive dyskinesia are the same as those used for Alzheimers disease., D. There is potential for new cholinergic agents used for Alzheimers disease to be investigated for treating tardive dyskinesia.")

# collected_results_loop = []
# for piece in response_generator:
#     collected_results_loop.append(piece)
#     print(piece)

