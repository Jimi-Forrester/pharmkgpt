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




with open(f'/home/mindrank/fuli/mcq_generator/QA_Data/QA.json', 'r', encoding='utf-8') as f:
    QA_list = json.load(f)


engine = RAGEngine(
    data_root="/home/mindrank/fuli/delirium-rag/benchmark_data2"
)
engine.setup_query_engine(
    model_type='VLLM',
    api_key=None,
    top_k=10,
    hops=1,
)

answer_dict = {}
for group_name, group_dict in QA_list.items():
    answer_dict[group_name] = {}
    for k, qa_dict in tqdm(group_dict.items()):
        answer_dict[group_name][k] = {}
        response_generator = engine.query(
            question= qa_dict['question'],
            option= qa_dict['options']
            )
        
        for data in response_generator:
            if data['type'] == 'result':
                print(data['data']['Answer'])
                predicted_option, reason = extract_answer_and_reason(data['data']['Answer'])
                answer_dict[group_name][k]['question'] = qa_dict['question']
                answer_dict[group_name][k]['options'] = qa_dict['options']
                answer_dict[group_name][k]['predicted_option'] = predicted_option
                answer_dict[group_name][k]['reason'] = reason
                answer_dict[group_name][k]['score'] = data['data']['score']
                answer_dict[group_name][k]['pmid'] = data['data']['Supporting literature']
        break
    
with open(f'pharmkgpt_DS32B_1.json', 'w', encoding='utf-8') as f:
    json.dump(answer_dict, f, ensure_ascii=False, indent=4)


