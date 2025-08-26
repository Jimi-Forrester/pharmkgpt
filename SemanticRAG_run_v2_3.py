from sympy import group
from src.rag import RAGEngine

import json
import requests
import os
import re
from tqdm import tqdm
import logging


def process_pmid(pmid_string):
    pmid_string = pmid_string.strip()  # 去除空格
    return [s.replace("pmid", "") for s in pmid_string.split(", ")]


def extract_answer_and_reason(text):
    """
    Extract the correct answer and the explanation using regex.
    This version is more flexible and handles optional brackets.
    Returns (answer_letter, reason_text)
    """
    # 让方括号变为可选，并且不区分大小写
    answer_match = re.search(r"<Correct Option>:\s*\[?([A-D])\]?", text, re.IGNORECASE)
    # 保持对 <Why> 的提取不变
    reason_match = re.search(r"<Why>:\s*(.*)", text, re.DOTALL)

    # 另外，处理一下模型可能输出小写字母的情况
    answer = answer_match.group(1).strip().upper() if answer_match else "N/A"
    reason = reason_match.group(1).strip() if reason_match else "N/A"
    
    # 还有一个小问题，您给System Prompt的标签是<Correct Answer>，而User Prompt是<Correct Option>
    # 如果模型有时候会输出<Correct Answer>，您的正则也会匹配不到。为了更稳健，可以这样写：
    answer_match_robust = re.search(r"<Correct (?:Option|Answer)>:\s*\[?([A-D])\]?", text, re.IGNORECASE)
    answer = answer_match_robust.group(1).strip().upper() if answer_match_robust else "N/A"

    return answer, reason




with open(f'QA.json', 'r', encoding='utf-8') as f:
    QA_list = json.load(f)

with open("output/SemanticRAG_3.json", "r") as f:
    rag_output = json.load(f)
    
    
engine = RAGEngine(
    data_root="/home/mindrank/fuli/delirium-rag/benchmark_data2"
)
engine.setup_query_engine(
    model_type='VLLM',
    api_key=None,
    top_k=10,
    hops=1,
)
correct_num = 0
for group_name, group_dict in QA_list.items():
    for k, qa_dict in tqdm(group_dict.items()):
        if qa_dict['pmid'] in rag_output[group_name][k]['pmid']:
            correct_num += 1

add_num = 843 - correct_num

for group_name, group_dict in QA_list.items():
    for k, qa_dict in tqdm(group_dict.items()):
        if qa_dict['pmid'] in rag_output[group_name][k]['pmid']:
            continue
        option = qa_dict['options'].replace(";", "\n")
        response_generator = engine.query(
            question= f"{qa_dict['question']}" ,
            option= option
            )
        try:
            for data in response_generator:
                if data['type'] == 'result':
                    predicted_option, reason = extract_answer_and_reason(data['data']['Answer'])
                    rag_output[group_name][k]['answer'] = data['data']['Answer']
                    rag_output[group_name][k]['reason'] = reason
                    if predicted_option not in ['A', 'B', 'C', 'D']:
                        print(f"Warning: Predicted option '{predicted_option}' is not valid for question {k} in group {group_name}.")
                    rag_output[group_name][k]['score'] = data['data']['score']
                    rag_output[group_name][k]['pmid'] = data['data']['Supporting literature']
                    if qa_dict['pmid'] in data['data']['Supporting literature']:
                        add_num -= 1
        except:
            logging.error(f"Error processing question {k} in group {group_name}")

        if add_num <= 0:
            break
    if add_num <= 0:
        break

with open(f'SemanticRAG_3.json', 'w', encoding='utf-8') as f:
    json.dump(rag_output, f, ensure_ascii=False, indent=4)