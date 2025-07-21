from sympy import group
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
    
import re

def extract_specific_answer_option(text: str) -> str:
    """
    从文本中提取单个大写字母选项 (A, B, C, 或 D)，优先处理 'Answer:' 和 'Answer is' 等格式。
    只使用 '</think>' 标签之后的部分作为有效输入。
    匹配顺序如下：
    1. 'Answer:' 后的选项
    2. 'Answer is' 后的选项
    3. 显式描述句式（例如 'the correct option is'）
    4. 格式如 A.、B:、C)
    5. 独立大写选项字母
    """

    # Step 0: 仅保留 </think> 后的文本（若存在）
    # if "</think>" in text:
    #     text = text.split("</think>", 1)[-1].strip()

    # Priority 1: 明确的 "Answer:"
    match = re.search(r"(?i)\bCorrect Answer\s*>:\s*\W*([A-D])\b", text)
    if match:
        return match.group(1).upper()

    # Priority 2: "Answer is" 或 "Answer is:"
    # match = re.search(r"(?i)\bAnswer\s+is\s*[:\-]?\s*\W*([A-D])\b", text)
    # if match:
    #     return match.group(1).upper()

    # # Priority 3: 显式描述
    # match = re.search(r"(?i)\b(?:the answer is|the correct option is|option is|the choice is|choice is|is)\s*[:\-]?\s*\W*([A-D])\b", text)
    # if match:
    #     return match.group(1).upper()

    # # Priority 4: 常见格式 A. B: C)
    # match = re.search(r"(?<![A-Za-z0-9])\b([A-D])[\.:)](?![A-Za-z0-9])", text)
    # if match:
    #     return match.group(1).upper()

    # # Priority 5: 独立的大写选项字母
    # match = re.search(r"\b([A-D])\b", text)
    # if match:
    #     return match.group(1).upper()

    return None

with open(f'/home/mindrank/fuli/delirium-rag/pharmkgpt_answer_processor1_and_2_path_DR_gemma3_only_answer.json', 'r', encoding='utf-8') as f:
    answer_dict = json.load(f)
    
with open(f'/home/mindrank/fuli/mcq_generator/QA_Data/QA.json', 'r', encoding='utf-8') as f:
    QA_list = json.load(f)

answer_dict = {}


for group_name, group_dict in QA_list.items():
    answer_dict[group_name] = {}
    for k, qa_dict in tqdm(group_dict.items()):
        # if extract_specific_answer_option(answer_dict[group_name][k]['answer']) != qa_dict['correct_option']:
        #if extract_specific_answer_option(answer_dict[group_name][k]['answer']) == None:
        # if QA_list[group_name][k]["pmid"] in qa_dict['pmid'] and extract_specific_answer_option(answer_dict[group_name][k]['answer']) == None:
        response_generator = engine.query(
            question= qa_dict['question'],
            option= qa_dict['options']
            )
        
        answer_dict[group_name][k] = {
            'question': qa_dict['question'],
            'options': qa_dict['options'],
            'pmid': qa_dict['pmid'],
            'answer': None
        }
        
        for data in response_generator:
            if data['type'] == 'result':
                if extract_specific_answer_option(data['data']['Answer']) == None:
                    response_generator = engine.query(
                        question= qa_dict['question'],
                        option= qa_dict['options']
                        )
                    for data in response_generator:
                        if data['type'] == 'result':
                            answer_dict[group_name][k]['answer'] = data['data']['Answer']
                            answer_dict[group_name][k]['pmid'] = data['data']['Supporting literature']
                            answer_dict[group_name][k]['pharmkgpt_option'] = extract_specific_answer_option(data['data']['Answer'])
                else:
                    answer_dict[group_name][k]['answer'] = data['data']['Answer']
                    answer_dict[group_name][k]['pmid'] = data['data']['Supporting literature']
                    answer_dict[group_name][k]['pharmkgpt_option'] = extract_specific_answer_option(data['data']['Answer'])

with open(f'pharmkgpt_processor1_and_2_gemma3.json', 'w', encoding='utf-8') as f:
    json.dump(answer_dict, f, ensure_ascii=False, indent=4)


# response_generator = engine.query(
#     question="What is the relationship between cholinergic drugs used for treating neuroleptic-induced tardive dyskinesia and Alzheimer's disease?", 
#     option="A. Cholinergic drugs are effective in treating both conditions.; B. Cholinergic drugs have been primarily developed for Alzheimers disease., C. Cholinergic drugs used for tardive dyskinesia are the same as those used for Alzheimers disease., D. There is potential for new cholinergic agents used for Alzheimers disease to be investigated for treating tardive dyskinesia.")

# collected_results_loop = []
# for piece in response_generator:
#     collected_results_loop.append(piece)
#     print(piece)

