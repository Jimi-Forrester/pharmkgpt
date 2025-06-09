
import json
import requests
import json
# 请求的 URL 和 headers
url = "http://localhost:5001/api/query"
headers = {
    "Content-Type": "application/json"
}


with open(f'/home/mindrank/fuli/vlidation_rag/AD_QA_all_DS_start_metabolism_0507_3000_5000.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
gene_list = data
new_query = []

for gene_dict in gene_list:
    if gene_dict["category"] =="metabolism":
        # 构造 payload
        payload = {
            "question": gene_dict['question']
        }

        # 发送 POST 请求
        response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)

        # 读取并打印响应内容
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if data['type'] == 'result' and data['data']['Context'] is not None:
                        if str('pmid'+gene_dict["pmid"]) in [i for i in data['data']['Context']]:
                            gene_dict['answer-rag'] = data['data']['Answer']
                            gene_dict['Context'] = data['data']['Context']
                            new_query.append(gene_dict)
                            print('yes')
        else:
            print(f"Error {response.status_code}: {response.text}")

print(len(new_query))


with open(f'/home/mindrank/fuli/vlidation_rag/delirirum/AD_QA_all_DS_start_metabolism_RAG_0507_3000_5000.json','w', encoding='utf-8')as f:
    json.dump(new_query, f, ensure_ascii=False, indent=4)



