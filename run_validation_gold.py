
import json
import requests
import json
# 请求的 URL 和 headers
url = "http://localhost:5000/api/query"
headers = {
    "Content-Type": "application/json"
}

def process_pmid(pmid_string):
    pmid_string = pmid_string.strip()  # 去除空格
    return [s.replace("PMID", "") for s in pmid_string.split(", ")]
    

with open(f'/home/mindrank/fuli/vlidation_rag/pathway_gold.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
QA_list = data
rag_dict  = {
    "question": {},
    "answer": {},
    "sp": {}
}

for qa_dict in QA_list:
    payload = {
        "question": qa_dict['question']
    }

    # 发送 POST 请求
    response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)

    # 读取并打印响应内容
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                if data['type'] == 'result' and data['data']['Context'] is not None:
                        rag_dict["question"][qa_dict["_id"]] = qa_dict['question']
                        rag_dict['answer'][qa_dict["_id"]] = data['data']['Answer'].split('**Supporting literature**:')[0]
                        rag_dict['sp'][qa_dict["_id"]] = process_pmid(data['data']['Answer'].split('**Supporting literature**:')[1])

    else:
        print(f"Error {response.status_code}: {response.text}")

with open(f'/home/mindrank/fuli/vlidation_rag/gemini_api_prediciton_pathway.json','w', encoding='utf-8')as f:
    json.dump(rag_dict, f, ensure_ascii=False, indent=4)



